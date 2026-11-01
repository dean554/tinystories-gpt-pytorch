import numpy as np
from tokenizers import Tokenizer
from tqdm.auto import tqdm

tokenizer = Tokenizer.from_file("/kaggle/working/tinystories_vocab.json")
eot_id = tokenizer.token_to_id("<|endoftext|>")

SEP = "<|endoftext|>"
CHUNK_CHARS = 1_000_000  # read ~1MB of text at a time

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
    """Reads the file in small chunks and yields one story at a time."""
    buffer = ""
    with open(txt_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            buffer += chunk
            parts = buffer.split(sep)
            buffer = parts.pop()  # hold back the possibly-incomplete tail
            for story in parts:
                story = story.strip()
                if story:
                    yield story
        story = buffer.strip()
        if story:
            yield story

def encode_txt_to_bin(txt_path, out_path, flush_every_tokens=500_000):
    n_stories = story_count(txt_path)
    total_tokens = 0

    with open(out_path, "wb") as out_f:
        buffer = []

        for story in tqdm(stream_stories(txt_path), total=n_stories,
                           desc=f"Encoding {txt_path.split('/')[-1]}"):
            ids = tokenizer.encode(story).ids
            buffer.extend(ids)
            buffer.append(eot_id)

            # Flush to disk once the buffer grows past the threshold
            if len(buffer) >= flush_every_tokens:
                arr = np.array(buffer, dtype=np.uint16)
                arr.tofile(out_f)
                total_tokens += len(arr)
                buffer = []

        # Flush whatever's left after the loop ends
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(out_f)
            total_tokens += len(arr)

    print(f"{txt_path}: {n_stories:,} stories -> {total_tokens:,} tokens -> saved to {out_path}")

encode_txt_to_bin(
    "/kaggle/input/datasets/ainsgown/tinystories-dataset/tiny_stories_data/TinyStories-train.txt",
    "/kaggle/working/train_encoded.bin",
)
encode_txt_to_bin(
    "/kaggle/input/datasets/ainsgown/tinystories-dataset/tiny_stories_data/TinyStories-valid.txt",
    "/kaggle/working/val_encoded.bin",
)
