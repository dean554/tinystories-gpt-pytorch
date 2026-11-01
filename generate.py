import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import numpy

# ==========================================
# HYPERPARAMETERS & CONFIG (Must match training)
# ==========================================
tokenizer_path = "/kaggle/working/tinystories_vocab.json"
checkpoint_path = "/kaggle/working/tiny_stories_checkpoint_step12000.pt" # Update with your desired checkpoint step

emb_dim = 1024
block_size = 512
num_heads = 8
head_size = emb_dim // num_heads
num_layers = 12
dropout = 0.0 # Set dropout to 0 for inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer to get vocab size
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
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
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb_table = nn.Embedding(block_size, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [Block(num_heads, emb_dim, head_size, block_size) for _ in range(num_layers)]
        )
        
        self.ln1 = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        char_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_emb_table(torch.arange(T, device=idx.device))

        x = char_emb + pos_emb
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln1(x)
        logits = self.lm_head(x)
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply Top-K sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# ==========================================
# INFERENCE PIPELINE
# ==========================================
def generate_story(prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    # DUAL GPU KAGGLE FIX: Clear leftover memory from the training loop
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading model...")
    model = GPTLanguageModel().to(device)
    
    # DUAL GPU KAGGLE FIX: Wrap model in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel for inference")
        model = nn.DataParallel(model)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Strict Security Fix
    with torch.serialization.safe_globals([
        numpy._core.multiarray.scalar, 
        numpy.dtype,
        numpy.dtypes.Float64DType
    ]):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
    # Extract the raw model module to load the state dict (handles DataParallel wrapping)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(ckpt['model_state_dict'])
    
    model.eval()
    print("Model loaded successfully!")

    # Encode prompt
    encoded_prompt = tokenizer.encode(prompt).ids
    idx = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
    
    print(f"\nPrompt: {prompt}")
    print("-" * 50)
    
    # Generate
    with torch.amp.autocast('cuda', dtype=torch.float16):
        # We call generate on raw_model because DataParallel doesn't natively expose custom methods
        out_ids = raw_model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode and format output
    generated_text = tokenizer.decode(out_ids[0].tolist())
    print(generated_text)
    print("-" * 50)
    return generated_text

if __name__ == "__main__":
    # Example usage
    prompt_text = "Once upon a time, there was a little boy named Tom."
    generate_story(prompt=prompt_text, max_new_tokens=500, temperature=0.7, top_k=40)