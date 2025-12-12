import torch 
from datasets import load_dataset
from torch.utils.data import IterableDataset

class FineWebEdu(IterableDataset):
    def __init__(self, split='train', block_size=1024, num_workers=4):
        self.block_size = block_size
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-100BT", # 100 billion tokens, highest quality
            split=split,
            streaming=True,
        )
        self.dataset = self.dataset.shuffle(buffer_size=10_000, seed=42)
        self.tokenizer = None # use raw text first (char level for simplicity)
        # later we upgrade to BPE with tiktoken or sentencepiece

    def __iter__(self):
        buffer = ""
        for sample in self.dataset:
            text = sample['text'].strip()
            if len(text) < 100: # skip short sample (ads, nav menus, etc.)
                continue
            buffer += " " + text 

            while len(buffer) >= self.block_size + 1:
                chunk = buffer[: self.block_size + 1]
                # simple char level tokenization for now 
                # upgrade to BPE later
                data = torch.tensor([ord(c) for c in chunk], dtype=torch.long)
                x = data[:-1]
                y = data[1:]
                yield x, y 
                buffer = buffer[self.block_size // 2 :] # 50% overlap = more data
