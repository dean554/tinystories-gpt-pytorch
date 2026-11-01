import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tokenizers import Tokenizer
import numpy as np

# ============================================================
# Config
# Architecture constants MUST match the pretraining run exactly,
# because we're about to load a state_dict that was saved with
# these exact shapes. Change any of these (emb_dim, num_layers,
# num_heads, block_size) and load_state_dict will throw a shape
# mismatch error.
# ============================================================
tokenizer = Tokenizer.from_file("/kaggle/working/tinystories_vocab.json")
vocab_size = tokenizer.get_vocab_size()

emb_dim = 1024
block_size = 512
batch_size = 64
num_heads = 8
head_size = emb_dim // num_heads
num_layers = 12
dropout = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- CHANGED: point these at your fine-tuning dataset, not the pretraining one.
# Fine-tuning implies a new (usually much smaller / more specific) corpus.
# Update these two paths to wherever your fine-tune .bin files live.
FINETUNE_TRAIN_PATH = '/kaggle/working/finetune_train_encoded.bin'
FINETUNE_VAL_PATH   = '/kaggle/working/finetune_val_encoded.bin'

train_data = np.memmap(FINETUNE_TRAIN_PATH, dtype=np.uint16, mode='r')
val_data   = np.memmap(FINETUNE_VAL_PATH, dtype=np.uint16, mode='r')

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((d[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((d[i + 1 : i + block_size + 1]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# ============================================================
# Model — copied verbatim from the pretraining script.
# This is not optional boilerplate: PyTorch's load_state_dict
# matches parameters by name, and the names come from how the
# class is written (self.query, self.key, self.blocks[i]...).
# If this class definition drifts even slightly from the one
# used to produce the checkpoint, loading will fail or silently
# load into the wrong tensors.
# ============================================================
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
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, torch.finfo(wei.dtype).min)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, emb_dim)
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
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, use_checkpointing=False):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb_table = nn.Embedding(block_size, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [Block(num_heads, emb_dim, head_size, block_size) for _ in range(num_layers)]
        )
        self.use_checkpointing = use_checkpointing

        self.ln1 = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)
        self.lm_head.weight = self.token_embedding_table.weight

        # This still runs on construction and randomly initializes every
        # weight. That's fine and expected — we immediately overwrite all
        # of it a few lines down with load_state_dict(). We are NOT
        # relying on this init for fine-tuning; it's just a leftover
        # side effect of building an nn.Module the normal way.
        self.apply(self._init_weights)

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

        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln1(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.2, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def create_optimizer(model, learning_rate):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:   # NOTE: frozen params (see below) are skipped here automatically
            continue
        if p.dim() < 2:
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': 0.1},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=learning_rate, betas=(0.9, 0.95))


# ============================================================
# Load the pretrained checkpoint
# ============================================================
PRETRAIN_CKPT_PATH = "/kaggle/working/tiny_stories_checkpoint_step12000.pt"  # <-- point at your actual final/best checkpoint

# --- CHANGED: optionally freeze early layers.
# In a GPT, early blocks tend to learn generic, low-level structure
# (local syntax, token co-occurrence) while later blocks specialize
# more toward the task/style. Freezing 0 means "fine-tune everything",
# which is the standard and usually best choice when your fine-tune
# dataset is reasonably sized. Set this > 0 only if your fine-tune
# dataset is tiny and you're worried about catastrophic forgetting /
# overfitting — freezing reduces both the risk and the compute cost,
# at the expense of flexibility.
FREEZE_FIRST_N_BLOCKS = 0

print("Loading pretrained checkpoint...")
model = GPTLanguageModel(use_checkpointing=True).to(device)
ckpt = torch.load(PRETRAIN_CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=True)
print(f"Loaded weights from pretraining step {ckpt['step']}")

if FREEZE_FIRST_N_BLOCKS > 0:
    for p in model.token_embedding_table.parameters():
        p.requires_grad = False
    for p in model.pos_emb_table.parameters():
        p.requires_grad = False
    for block in model.blocks[:FREEZE_FIRST_N_BLOCKS]:
        for p in block.parameters():
            p.requires_grad = False
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Froze embeddings + first {FREEZE_FIRST_N_BLOCKS} blocks ({n_frozen:,} params)")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = nn.DataParallel(model)

# --- CHANGED: much lower learning rate than pretraining.
# Pretraining's max_lr was 3e-4, chosen because the model started from
# random noise and needed large updates to learn anything at all. The
# fine-tuning model already sits in a good region of the loss landscape;
# large updates now would just as easily destroy what it learned
# (catastrophic forgetting) as improve it. A common rule of thumb is
# 1/10th to 1/100th of the pretraining peak LR. Starting at 1/10th here.
finetune_lr = 3e-5
optimizer = create_optimizer(model, finetune_lr)

# --- CHANGED: deliberately NOT loading ckpt['optimizer_state_dict'].
# AdamW keeps running estimates of each parameter's gradient mean and
# variance (its momentum). Those estimates were tuned for a completely
# different learning rate (3e-4) and a completely different data
# distribution (the pretraining corpus). Carrying them over into
# fine-tuning at 3e-5 tends to produce a mismatched, sometimes unstable
# first few hundred steps. Starting the optimizer fresh is the more
# common and more predictable choice. (If you ever need to *resume* a
# fine-tuning run that was interrupted, that's the one case where you
# *would* reload optimizer_state_dict — because then it matches the
# same LR and data you're continuing with.)

print(f"Fine-tuning on device: {device}")
train_losses, val_losses, steps_logged = [], [], []
scaler = torch.amp.GradScaler('cuda')


@torch.no_grad()
def estimate_loss(eval_iters=10):
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        split_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                _, loss = model(xb, yb)
            split_losses[k] = loss.mean().item()
        losses[split] = split_losses.mean()
    model.train()
    return losses


# --- CHANGED: much shorter schedule.
# Pretraining used a 1200-step warmup over a 30000-step run because it
# was covering a huge, diverse corpus from scratch. Fine-tuning datasets
# are almost always far smaller, and the model doesn't need to "relearn
# how to model language" — it just needs to adapt. A short warmup avoids
# an unstable first few steps at the new (much smaller) LR, and a short
# total step count matches a smaller dataset and reduces overfitting risk.
# Treat max_steps as an upper bound — early stopping below will likely
# cut the run short anyway.
eval_interval = 50
warmup_steps = 100
max_steps = 3000
max_lr = finetune_lr
min_lr = max_lr * 0.1

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# --- NEW: early stopping.
# Fine-tune sets are typically small enough that the model can start
# memorizing them (val loss rising while train loss keeps falling) well
# before max_steps. patience * eval_interval is how many steps we
# tolerate without a new best val loss before we stop.
best_val_loss = float('inf')
patience = 5
patience_counter = 0

start_step = 0
for i in range(start_step, max_steps + 1):
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    xb, yb = get_batch('train')
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits, loss = model(xb, yb)
    loss = loss.mean()
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    # --- CHANGED: checkpoint filename says "finetuned" so it can never
    # collide with or overwrite a pretraining checkpoint, and saves more
    # frequently (every 500 vs every 1000 steps) since the whole run is
    # much shorter.
    if i % 500 == 0 and i > 0:
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            'step': i,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"/kaggle/working/tiny_stories_finetuned_step{i}.pt")

    if i % eval_interval == 0:
        losses = estimate_loss()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        steps_logged.append(i)
        print(f"step {i}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {lr:.2e}")

        # --- NEW: track + save the best checkpoint by val loss, and
        # stop once val loss stalls for `patience` evals in a row.
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'step': i,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "/kaggle/working/tiny_stories_finetuned_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at step {i} "
                      f"(val loss hasn't improved in {patience * eval_interval} steps)")
                break
