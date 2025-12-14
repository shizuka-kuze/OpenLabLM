"""
Small LLM (Muon + MLA + RoPE + RMSNorm + ReLU^2)

12/14/24
"""
from __future__ import annotations
import os
import time
import random
import numpy
import argparse
#from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter #tensorboard --logdir=./runs

from datasets import load_dataset
import tiktoken

from rich.console import Console
from rich.table import Table
from rich.progress import track

from modules.model.attention import MultiHeadLatentAttention
from modules.model.norm import RMSNorm
from modules.optimizer.muon import SingleDeviceMuonWithAuxAdam
from modules.config import ModelConfig

#Make results deterministic
random.seed(0)
torch.manual_seed(0)
numpy.random.seed(0)

# Data Pipeline
class LLMDataset(Dataset):
    def __init__(self, hf_data, tokenizer, context_len, mode="dolly"):
        # We hold a reference to the HF dataset. HF datasets are memory-mapped by default, so this supports datasets larger than RAM.
        self.data = hf_data
        self.enc = tokenizer
        self.context_len = context_len
        self.mode = mode
        self.eot = self.enc._special_tokens.get('<|endoftext|>', 50256)

    def __len__(self):
        return len(self.data)

    def format_dolly(self, item):
        text = f"Instruction: {item['instruction']}\n"

        if item.get('context') and str(item['context']).strip():
            text += f"Context: {item['context']}\n"
                
        text += f"Response: {item['response']}<|endoftext|>"
        
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.mode == "dolly":
            text = self.format_dolly(item)
        else:
            text = item.get('text', '') + "<|endoftext|>"

        # Encode ONCE.
        tokens = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        n_tokens = len(tokens) # Capture length immediately

        # Prepare logic for Truncation vs Padding
        target_len = self.context_len + 1 
        
        if n_tokens > target_len:
            # Truncate
            tokens = tokens[:target_len]
            # If truncated, the valid data fills the whole context
            valid_len = self.context_len 
            padding_len = 0
        else:
            # Pad
            padding_len = target_len - n_tokens
            tokens = tokens + [self.eot] * padding_len
            # If padded, the valid data is the original length (n_tokens) Note: We clamp this to context_len because x is 1 token shorter than input tokens
            valid_len = min(n_tokens - 1, self.context_len)

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        mask = torch.zeros(self.context_len, dtype=torch.long)
        
        # We want to attend to everything up to valid_len.
        mask[:valid_len] = 1 
        
        # Set padding targets to -100 so CrossEntropy ignores them
        if padding_len > 0:
            if valid_len < self.context_len:
                y[valid_len:] = -100

        return x, y, mask

def get_validation_prompt(dataset, tokenizer, mode="dolly"):
    """Generates a prompt from the held-out validation set."""
    
    idx = random.randint(0, len(dataset) - 1)
    item = dataset[idx]
    
    if mode == "dolly":
        prompt = f"Instruction: {item['instruction']}\n"
        
        if item.get('context') and str(item['context']).strip():
            prompt += f"Context: {item['context']}\n"
            
        prompt += "Response:"
        target_response = item['response']
    else:
        # For Shakespeare/Raw text, grab the first few words as prompt
        text = item.get('text', '')
        words = text.split()

        # Grab first 8-32 words as prompt, next 20 as target preview...
        if len(words) > 0:
            word_count = torch.randint(8, 32, (1,)).item()
            # If text is shorter than word_count, then... just grabs the whole thing!
            prompt = " ".join(words[:word_count])
            target_response = " ".join(words[word_count:word_count+20]) + "..."
        else:
            prompt = "The"
            target_response = "..."
    
    tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    return torch.tensor(tokens).unsqueeze(0), prompt, target_response

"""class FeedForward(nn.Module):
    #SwiGLU Feed-Forward Network.
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, multiple_of: int = 256, use_bias: bool = False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=use_bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x), inplace=True) * self.w3(x))"""

# Model Components
class ReluSquaredMLP(nn.Module): 
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # I find bias wastes params and hurts performance... idk...
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.relu(self.w1(x)).square() * self.w2(x))
        #return self.w3(F.leaky_relu(self.w1(x), 0.025).square() * self.w2(x))

# I read ReLU^2 outperforms other choices including SwiGLU but have not tested it. It doesn't solve dying gradient problem so might be worth trying something else.

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadLatentAttention(config)
        self.mlp = ReluSquaredMLP(config.dim, int(3.5 * config.dim))
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)

    def forward(self, x, mask=None, past_kv=None):
        # Pass mask and kv to attention
        attn_out, new_kv = self.attn(self.norm1(x), mask=mask, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv

# Maybe "'effecient'" would make it sound fancier.
class SmallLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight 

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    #It would seem I do not know how to implement KV-cahcing as both speed and performance have tanked. Sorry...
    def forward(self, idx, targets=None, mask=None, past_kv=None):
        b, t = idx.size()
        
        if past_kv is not None:
            combined_mask = None
        else:
            # Create Causal Mask, tril gives us lower triangle as 1s
            causal_mask = torch.tril(torch.ones(t, t, device=idx.device, dtype=torch.bool))
            causal_mask = causal_mask.view(1, 1, t, t)
            
            if mask is not None:
                # Mask is (B, T) with 1s and 0s. Convert to Bool.
                mask_bool = mask.view(b, 1, 1, t) > 0.5 
                # Combine: Must be Causal AND Valid(Mask)
                combined_mask = causal_mask & mask_bool
            else:
                combined_mask = causal_mask

        x = self.token_emb(idx)
        
        new_kvs = []
        for i, block in enumerate(self.blocks):
            # Get the cache for this specific layer if it exists
            layer_past = past_kv[i] if past_kv is not None else None
            
            x, layer_new_kv = block(x, mask=combined_mask, past_kv=layer_past)
            new_kvs.append(layer_new_kv)
            
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, new_kvs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        
        # Prepare KV Cache container
        past_kv = None 
        
        for _ in range(max_new_tokens):
            # If we have cache, only pass the VERY LAST token
            if past_kv is not None:
                idx_cond = idx[:, -1:] # (B, 1)
            else:
                idx_cond = idx # First pass (Prefill) processes whole prompt
                
            # Forward pass needs to return the new cache
            logits, _, past_kv = self(idx_cond, past_kv=past_kv)
            
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            if idx_next.item() == 50256: 
                break
                
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="shakespeare", choices=["dolly", "smollm", "shakespeare"], help="Dataset choice")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./runs", exist_ok=True)
    
    console = Console()
    writer = SummaryWriter(log_dir=f"runs/{args.dataset}_{int(time.time())}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig()

    console.print(f"[bold yellow]NOTE: Cold start time might take about a minute and first epoch/validation is slowest![/bold yellow]")
    console.print(f"[bold green]Initializing Small LLM on {device}[/bold green]")
    console.print(f"Dataset: [cyan]{args.dataset}[/cyan] | Config: {config}")

    tokenizer = tiktoken.get_encoding("gpt2")
    
    epochs = 80 # Around epoch 8 with tiny shakesphere it starts overfitting... so by epoch 80~ it should Grok... right?! ;-; (it doesn't seem to.)
    global_step = 0
    
    if args.dataset == "shakespeare":
        # tiny_shakespeare scripts are deprecated in recent datasets versions. We load the raw text directly using the 'text' builder and split. it. manually.
        ds = load_dataset("text", data_files={"train": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"})
        
        # 'text' builder splits by line, so we join it back
        full_text = "\n".join(ds['train']['text'])
        
        # 0.95 Train, 0.05 Validation
        split_idx = int(len(full_text) * 0.95)
        train_text = full_text[:split_idx]
        val_text = full_text[split_idx:]
        
        def chunk_data(text_data, chunk_size=2000):
            chunks = []
            for i in range(0, len(text_data), chunk_size):
                chunks.append({'text': text_data[i:i+chunk_size]})
            return chunks

        train_raw = chunk_data(train_text)
        val_raw = chunk_data(val_text)
        
        console.print(f"Processed Shakespeare: {len(train_raw)} train chunks, {len(val_raw)} val chunks")

    elif args.dataset == "dolly":
        hf_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        split_dataset = hf_dataset.train_test_split(test_size=0.05, seed=0)
        train_raw = split_dataset['train']
        val_raw = split_dataset['test']
        
    else: # smollm or cosmopedia
        hf_dataset = load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train")
        split_dataset = hf_dataset.train_test_split(test_size=0.05, seed=0)
        train_raw = split_dataset['train']
        val_raw = split_dataset['test']

    console.print(f"[bold green]Loading dataset...[/bold green]")
    train_ds = LLMDataset(train_raw, tokenizer, config.context_len, mode=args.dataset)
    val_ds = LLMDataset(val_raw, tokenizer, config.context_len, mode=args.dataset)

    console.print(f"[bold green]Loading dataloader...[/bold green]")
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    console.print(f"[bold green]Loading model...[/bold green]")
    model = SmallLLM(config).to(device)
    console.print(f"[yellow]Model Size:[/yellow] [cyan]{sum(p.numel() for p in model.parameters())/1e6:.2f}m params[/cyan]")
    
    console.print(f"[bold green]Loading optimizer groups...[/bold green]")
    muon_params = []
    adam_params = []
    for n, p in model.named_parameters():
        if p.ndim >= 2 and "token_emb" not in n and "lm_head" not in n:
            muon_params.append(p)
        else:
            adam_params.append(p)
            
    optim_groups = [
        {"params": muon_params, "lr": config.lr_muon, "momentum": 0.95, "use_muon": True, "weight_decay": config.weight_decay},
        {"params": adam_params, "lr": config.lr_adam, "betas": (0.9, 0.95), "use_muon": False, "weight_decay": config.weight_decay}
    ]
    optim = SingleDeviceMuonWithAuxAdam(optim_groups)

    # Scheduler seems to help with validation loss even if train loss is a 50/50 gacha pull...
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, 
        T_max=total_steps, 
        eta_min=config.lr_adam * 0.025
    )

    best_val_loss = float('inf')

    console.print(f"[bold green]Starting training, to check results try `[/bold green][bold yellow]tensorboard --logdir=./runs[/bold yellow][bold green]`![/bold green]")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        progress = track(train_loader, description=f"Epoch {epoch+1}/{epochs}")
        for x, y, mask in progress:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss, _ = model(x, targets=y, mask=mask)
                
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optim.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            if global_step % 50 == 0:
                writer.add_scalar("Loss/train", loss.item(), global_step)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                _, loss, _ = model(x, targets=y, mask=mask)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        console.print(f"Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)

        # Only save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            console.print(f"[bold yellow]New best model found! Loss: {best_val_loss:.4f}[/bold yellow]")
            torch.save(model.state_dict(), f"./models/{args.dataset}_best.pt")

        # Optional: Save latest model every epoch to resume training if it crashes or just to archive
        #torch.save(model.state_dict(), f"./models/{args.dataset}_last.pt")

        input_tensor, prompt_text, target_resp = get_validation_prompt(val_raw, tokenizer, mode=args.dataset)
        input_tensor = input_tensor.to(device)
        
        generated = model.generate(input_tensor, max_new_tokens=50, top_k=50, temperature=1)
        decoded = tokenizer.decode(generated[0].tolist())

        # For raw text, we just want to see the continuation
        response_only = decoded[len(prompt_text):] if args.dataset == "dolly" else decoded
        
        console.print(f"\n[bold]Epoch {epoch+1}[/bold] | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        tbl = Table(title=f"Validation Generation ({args.dataset})")
        tbl.add_column("Prompt", style="cyan", width=40)
        tbl.add_column("Model Response", style="green")
        tbl.add_row(prompt_text, response_only)
        console.print(tbl)
        
        writer.add_text("Generation", f"Prompt: {prompt_text}\nResponse: {decoded}\nTarget: {target_resp}", epoch+1)

    writer.close()

if __name__ == "__main__":
    main()