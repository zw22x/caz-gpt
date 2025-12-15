import torch
import math
import time
from model import GPT
from config import GPTconfig
from data import FineWebEdu
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Training on {device}")

config = GPTconfig()
model = GPT(config).to(device)

# AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.95))

# data
dataloader = iter(FineWebEdu(block_size=config.block_size))

# mix precision 
scaler = torch.amp.GradScaler(device) if device != 'cpu' else None 

def get_lr(step): # cosine learning rate decay 
    warmup = 2000
    if step < warmup:
        return 6e-4 * step / warmup 
    decay_ratio = (step - warmup) / (50000 - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return 6e-5 * coeff * (6e-4 - 6e-5)

for step in tqdm(range(50000)): # training loop
    x, y = next(dataloader)
    x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device) # add batch dimension

    lr = get_lr(step)
    for g in optimizer.param_groups:
        g['lr'] = lr

    with torch.amp.autocast(device_type=device.split(':')[0] if ':' in device else device):
        _, loss = model(x, y)

    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    optimizer.zero_grad()

    if step % 500 == 0:
        print(f"step {step} | loss {loss.item():.4f} | lr {lr:.2e}")

    if step % 5000 == 0 and step > 0:
        torch.save(model.state_dict(), f"caz_gpt_124M_step{step}.pth")

torch.save(model.state_dict(), "caz_gpt_124M_final.pth")
print("Training complete")
