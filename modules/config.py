from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257 # From the tokernizer, if you change it change this.

    #Main model config
    dim: int = 512 
    depth: int = 8
    num_heads: int = 8
    
    # MLA Config
    q_lora_rank: int = 128
    kv_lora_rank: int = 64
    nope_head_dim: int = 32
    rope_head_dim: int = 32
    v_head_dim: int = 64
    
    # Context Config
    context_len: int = 512
    dropout: float = 0.025 # Not implemneted
    batch_size: int = 8 # Higher batch sizes use more data more quickly and help with accurate estimation of the gradient.
    
    # Optimization Config
    lr_muon: float = 0.02
    lr_adam: float = 5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0