from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tqdm.auto import tqdm
import os

TRAIN_TXT = "/kaggle/input/datasets/ainsgown/tinystories-dataset/tiny_stories_data/TinyStories-train.txt"
VOCAB_OUT = "/kaggle/working/tinystories_vocab.json"
SEP = "<|endoftext|>"
CHUNK_CHARS = 1_000_000

#story count function
def story_count(txt_path, sep=SEP, chunk_chars=CHUNK_CHARS):
    """Cheap first pass: count separators without holding the file in memory."""
    count = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            count += chunk.count(sep)
    return count

def stream_stories(txt_path, sep=SEP, chunk_chars=CHUNK_CHARS):
    #Reads the file in small chunks and yields one story at a time.
    buffer = ""
    with open(txt_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            buffer += chunk

            # Split on separator; the LAST piece may be incomplete
            # (cut off mid-story by this chunk boundary)
            parts = buffer.split(sep)
            buffer = parts.pop()  # keep the incomplete tail for next loop

            for story in parts:
                story = story.strip()
                if story:
                    yield story

        # End of file - flush whatever's left in the buffer
        story = buffer.strip()
        if story:
            yield story

# pre-pass just to get a count for the progress bar
print("Counting stories (fast pass)...")
n_stories = story_count(TRAIN_TXT)
print(f"Found approximately {n_stories:,} stories")

# Wrap the generator in tqdm so training shows a live progress bar
def stories_with_progress(txt_path, total):
    for story in tqdm(stream_stories(txt_path), total=total, desc="Streaming stories"):
        yield story

# nitialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

vocab_size = 8000

trainer = BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=["<pad>", "<unk>", "<|endoftext|>"],
    min_frequency=2,
    show_progress=True,
)

# Train directly from the generator
print("Training BPE tokenizer...")
tokenizer.train_from_iterator(
    stories_with_progress(TRAIN_TXT, n_stories),
    trainer=trainer,
    length=n_stories,
)

# Save
tokenizer.save(VOCAB_OUT)
print("Saved to", VOCAB_OUT)
print("Vocab size:", tokenizer.get_vocab_size())
print("endoftext id:", tokenizer.token_to_id("<|endoftext|>"))
