from dataclasses import dataclass 

@dataclass 
class GPTconfig:
    block_size: int = 1024 
    vocab_size: int = 50304 # use GPT2 bpe later, reserve size
    n_layer: int = 12 
    n_head: int = 12
    n_embd: int = 768 # 124M parameters
    dropout: float = 0.1
    bias: bool = False # True = LLaMA style, False = GPT-2 

@dataclass
class TrainConfig:
    batch_size: int = 64 # adjust down if OOM
    learning_rate: float = 6e-4
    max_iters: int = 50000 # 50000 steps = decent quality
    warmup_iters: int = 2000
    lr_decay_iters: int = 50000
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interal: int = 1000
    log_interval: int = 100
    