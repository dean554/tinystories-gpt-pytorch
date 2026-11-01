# TinyStories GPT (PyTorch)

A 160M-parameter GPT-style language model written from scratch in PyTorch, trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. The repository covers the whole pipeline: training a custom BPE tokenizer, encoding the dataset, pretraining, fine-tuning on an instruction-style prompt format, and generating stories.

It was built and trained on Kaggle with 2 GPUs.

## Example Output

**Pretrained model**, prompt: *"Once upon a time, there was a little boy named Tom."*

> Tom was very curious and loved to explore. One day he went for a walk in the park and saw a big tree. The tree was very tall and had lots of green leaves. Tom wanted to climb the tree and see what was on top. [...] Tom learned that it's okay to be curious and ask for help when he needs it.

**Fine-tuned model**, prompt:

```
Write a story about: A cat and a mouse who decide they want to be friends instead of fighting.
They sit under a tree and share a piece of cheese.
With: Happy end
### Story:
```

> One day, the cat said, "I want to be friends with the mouse." The mouse said, "Me too, but I don't know if we can be friends." The cat thought for a moment and said, "Let's share the cheese." So, the cat and the mouse sat under the tree and ate the cheese together. They were very happy. After they ate, they played and had fun. The cat and the mouse became best friends.

## Training Results

| Stage | Steps | Final train loss | Final val loss |
|-------|-------|------------------|----------------|
| Pretraining (TinyStories) | 30,000 planned (log shown up to ~9,900; checkpoint at 12,000 used for generation) | 1.3653 (step 9,900) | 1.3783 (step 9,900) |
| Fine-tuning | stopped early at 1,050 | 1.1443 | 1.1514 |

Fine-tuning validation loss dropped from 2.5583 at step 0 to about 1.15, and training stopped automatically once validation loss stopped improving (patience of 5 evaluations).

## Model Architecture

A decoder-only Transformer implemented without any prebuilt Transformer modules:

| Setting | Value |
|---------|-------|
| Parameters | 160M |
| Layers | 12 |
| Embedding size | 1024 |
| Attention heads | 8 (head size 128) |
| Context length (`block_size`) | 512 tokens |
| Feed-forward | `1024 -> 4096 -> 1024`, GELU |
| Normalization | Pre-LayerNorm |
| Dropout | 0.1 (0.0 at inference) |
| Vocabulary | 8,000 (byte-level BPE) |
| Embeddings | Learned positional embeddings, token embedding tied to the output layer |

Implementation details: causal self-attention written by hand (per-head `Head` modules concatenated in `MultiHeadAttention`), GPT-style weight initialization (`std=0.02`), with residual projections scaled by `1/sqrt(2 * num_layers)`.

## Repository Structure

```
tinystories-gpt-pytorch/
├── vocab.py            # Trains the 8,000-token BPE tokenizer
├── encoder.py          # Encodes text files into uint16 .bin token files
├── model_trainloop.py  # Model definition + pretraining loop
├── finetune.py         # Fine-tunes the pretrained checkpoint
└── generate.py         # Loads a checkpoint and generates stories
```

## Pipeline

### 1. Train the tokenizer (`vocab.py`)

Trains a byte-level BPE tokenizer with a vocabulary of 8,000 on the TinyStories training file, with special tokens `<pad>`, `<unk>` and `<|endoftext|>`. The file is streamed in 1M-character chunks, so it never has to fit in memory. Output: `tinystories_vocab.json`.

### 2. Encode the data (`encoder.py`)

Tokenizes the train and validation text files story by story, appends `<|endoftext|>` after each story, and writes the token IDs as `uint16` to `train_encoded.bin` and `val_encoded.bin`, flushing to disk in 500k-token blocks. The training scripts read these files with `np.memmap`.

### 3. Pretrain (`model_trainloop.py`)

| Setting | Value |
|---------|-------|
| Batch size | 64 sequences of 512 tokens |
| Optimizer | AdamW, betas `(0.9, 0.95)`, weight decay `0.1` (not applied to biases and LayerNorm) |
| Learning rate | Peak `3e-4`, 1,200 warmup steps, cosine decay to `3e-5` |
| Max steps | 30,000 |
| Precision | Mixed precision (fp16 autocast + `GradScaler`) |
| Memory savers | Gradient checkpointing, gradient clipping at 1.0 |
| Multi-GPU | `DataParallel` when more than one GPU is available |
| Logging / saving | Train and val loss every 100 steps, checkpoint every 1,000 steps |

### 4. Fine-tune (`finetune.py`)

Continues training from a pretraining checkpoint on an instruction-style dataset (`Write a story about: ... With: ... ### Story:`).

| Setting | Value |
|---------|-------|
| Learning rate | Peak `3e-5`, 100 warmup steps, cosine decay to 10% |
| Max steps | 3,000 |
| Evaluation | Every 50 steps |
| Early stopping | Patience of 5 evaluations |
| Checkpoints | Best model saved as `tiny_stories_finetuned_best.pt`, plus one every 500 steps |
| Option | `FREEZE_FIRST_N_BLOCKS` freezes the embeddings and the first N blocks (default `0`, nothing frozen) |

The fine-tuning data (`finetune_train_encoded.bin`, `finetune_val_encoded.bin`) is expected in the same encoded `uint16` format as the pretraining data.

### 5. Generate (`generate.py`)

Rebuilds the model, loads a checkpoint, and generates text with temperature and top-k sampling (defaults: `temperature=0.8`, `top_k=50`).

```python
generate_story(
    prompt="Once upon a time, there was a little boy named Tom.",
    max_new_tokens=500,
    temperature=0.7,
    top_k=40,
)
```

Set `checkpoint_path` at the top of the file to choose which model to use, for example `tiny_stories_finetuned_best.pt` for the fine-tuned model.

## Setup

```bash
pip install torch numpy tokenizers tqdm
```

Run the scripts in this order: `vocab.py`, `encoder.py`, `model_trainloop.py`, then `finetune.py` and `generate.py`.

The scripts use Kaggle paths (`/kaggle/input/...` for the dataset and `/kaggle/working/...` for outputs). Change the path constants at the top of each file to run locally.

## Notes

- Generation does not stop at `<|endoftext|>`, so the model sometimes starts a new story after finishing the first (visible in the pretrained sample above). Trim the output at the first `<|endoftext|>` or add a stop check in `generate()`.
- Loss values are estimated from 10 random batches per evaluation, so they fluctuate a little between steps.