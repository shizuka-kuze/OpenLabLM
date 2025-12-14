import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.model.embeddings import RotaryEmbedding, apply_rotary_emb
from modules.model.norm import RMSNorm
from modules.config import ModelConfig

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim = config.dim
        
        # Dimensions
        self.nope_dim = config.num_heads * config.nope_head_dim
        self.rope_dim = config.num_heads * config.rope_head_dim
        self.value_dim = config.num_heads * config.v_head_dim
        
        # Compression (Latent)
        self.compress_q = nn.Linear(config.dim, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank)
        
        self.compress_kv = nn.Linear(config.dim, config.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(config.kv_lora_rank)
        
        # Decompression
        self.decompress_q_nope = nn.Linear(config.q_lora_rank, self.nope_dim, bias=False)
        self.decompress_q_rope = nn.Linear(config.q_lora_rank, self.rope_dim, bias=False)
        
        self.decompress_k_nope = nn.Linear(config.kv_lora_rank, self.nope_dim, bias=False)
        self.decompress_v = nn.Linear(config.kv_lora_rank, self.value_dim, bias=False)
        
        # Decoupled RoPE Key
        self.k_rope_linear = nn.Linear(config.dim, config.rope_head_dim, bias=False)
        
        self.out_proj = nn.Linear(self.value_dim, config.dim, bias=False)
        self.rotary = RotaryEmbedding(config.rope_head_dim, config.context_len)

    def forward(self, x, mask=None, past_kv=None):
        B, T, _ = x.shape
        
        # 1. Compress
        q_lat = self.q_norm(self.compress_q(x))
        kv_lat = self.kv_norm(self.compress_kv(x))
        
        # 2. Decompress Q 
        q_nope = self.decompress_q_nope(q_lat).view(B, T, self.num_heads, -1)
        q_rope = self.decompress_q_rope(q_lat).view(B, T, self.num_heads, -1)
        
        # 3. Decompress KV
        k_nope = self.decompress_k_nope(kv_lat).view(B, T, self.num_heads, -1)
        v = self.decompress_v(kv_lat).view(B, T, self.num_heads, -1)
        
        # 4. RoPE Key
        k_rope = self.k_rope_linear(x).view(B, T, 1, -1)
        k_rope = k_rope.expand(-1, -1, self.num_heads, -1)
        
        # 5. Apply RoPE
        # If we have past_kv, we are at a later position, not 0
        start_pos = 0
        if past_kv is not None:
            # past_kv is (k, v). k is [B, Heads, SeqLen, Dim]
            start_pos = past_kv[0].shape[2]
            
        pos = torch.arange(start_pos, start_pos + T, device=x.device)
        freqs = self.rotary(pos)  # Shape: [T, HeadDim]
        freqs = freqs[None, :, None, :] # Broadcast
        
        q_rope = apply_rotary_emb(freqs, q_rope)
        k_rope = apply_rotary_emb(freqs, k_rope)
        
        # 6. Concat Q and K parts
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        
        # 7. Transpose for Attention: [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 8. KV Cache Management
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        # Store current KV for next step
        # (For MLA we technically only need to cache the compressed latent, 
        # but caching decompressed heads is easier for standard SDPA usage)
        current_kv = (k, v) 
        
        # 9. Flash Attention
        # If mask is provided (combined padding mask), we use it. 
        # If no mask, we assume causal training.
        # If past_kv exists, we are generating 1 token attending to all previous, 
        # so is_causal=False (since we manually concatenated inputs).
        
        dropout_p = 0.0 # not done
        
        if mask is not None:
            # mask is (B, 1, T, T) - SDPA broadcasts
            #out = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask, dropout_p=dropout_p)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        else:
            # If we are training (T > 1) and no explicit mask, use causal
            # If we are generating (T=1) and have past_kv, is_causal must be False
            is_causal = (T > 1 and past_kv is None)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
        
        # 10. Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.out_proj(out), current_kv