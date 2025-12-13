<div align="center">
    <h1>caz-gpt</h1>
    <p>a 124M-parameter GPT I’m building from scratch in my bedroom 
    This is my take on “let’s actually build GPT, not just watch YouTube videos about it.”
    I’m training it on FineWeb-Edu 100BT — the same cleaned, high-quality internet text that Llama-3, Qwen-2, and every serious 2025 model uses.</p>   
    <h2>Current status (Dec 2025):</h2>
    <p>124 million parameters (12 layers, 12 heads, 768 dim) — exactly GPT-2 small size
    Streaming FineWeb-Edu straight from Hugging Face (zero disk usage)
    FlashAttention-2 + mixed precision + cosine LR + all the 2025 tricks
    Character-level for now (yes, on purpose — I want to understand every single byte first)
    Loss is already dropping fast</p>
    <h2>What’s in this repo right now</h2>
    textdata.py        → infinite streaming dataloader (the part I’m most proud of)
    config.py      → clean hyper-params for the 124M model
    model.py       → the actual transformer (coming tomorrow)
    train.py       → fast training loop with AMP & checkpointing
    generate.py    → talk to the model once it’s alive
    <h2>Next steps (doing them one by one):</h2>
    <p>Switch to BPE (tiktoken) — char-level is cute but I want real intelligence
    Train until loss < 2.0
    Make it sound like Grok (yes, I’m going to feed it Elon tweets eventually)
    Maybe scale to 350M or 1.3B when I get bored</p>
    <p>Feel free to star, fork, roast, or steal ideas.<p>
    <p>I’m documenting everything because one day I want to look back and say “damn, I actually did this.”</p>
    <p>— will (zw22x)</p>
    <p>dec 2025, somewhere on earth, probably 3am</p>
    <p>p.s. if you’re reading this in 2026 and this model is actually good… lmk, I’ll buy you coffee</p>
