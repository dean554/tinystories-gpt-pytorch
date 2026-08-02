import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import numpy as np

# 1. Load the BPE tokenizer
tokenizer = Tokenizer.from_file("/content/drive/MyDrive/data/tinystories_vocab.json")

# 5. Get the vocab size dynamically from the tokenizer
vocab_size = tokenizer.get_vocab_size()


# Hyperparameters

emb_dim = 512
block_size = 256
batch_size = 64
num_heads = 8
head_size = emb_dim // num_heads
num_layers = 12
learning_rate = 1e-4
dropout = 0.1



device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 2. Split the data
train_data = np.memmap('drive/MyDrive/data/train_encoded.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('drive/MyDrive/data/val_encoded.bin', dtype=np.uint16, mode='r')

# 3. Update the get_batch function to cast the memmap slices back to PyTorch tensors
def get_batch(split):

    d = train_data if split == 'train' else val_data
    # Pick random starting indices
    ix = torch.randint(len(d) - block_size, (batch_size,))
    # Slice the numpy memmap and convert to torch.long (int64) for the embedding layer
    x = torch.stack([torch.from_numpy((d[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((d[i + 1 : i + block_size + 1]).astype(np.int64)) for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y


class Head(nn.Module):

    def __init__(self, emb_dim, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)


        wei = q@k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei@v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,  num_heads, emb_dim, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class Block(nn.Module):
    def __init__(self, num_heads, emb_dim, head_size, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.mha = MultiHeadAttention(num_heads, emb_dim, head_size, block_size)

        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffwd = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb_table = nn.Embedding(block_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.Block = nn.Sequential(
            *[Block(num_heads, emb_dim, head_size, block_size) for _ in range(num_layers)]
        )
        self.ln1 = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)
        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)  # <-- re-init everything to a sane scale

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        char_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_emb_table(torch.arange(T, device=idx.device))

        x = char_emb + pos_emb
        x = self.dropout(x)

        x = self.Block(x)
        x = self.ln1(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.2, top_k=None):
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :]

            # Apply Temperature
            logits = logits / temperature

            # Apply Top-K Sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
