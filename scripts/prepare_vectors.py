"""Cut the 5 GB GloVe file down to a small vocabulary of real, common English words.

GloVe files list words from most to least frequent, so the first N lines are the N most
common tokens. We keep a token only if it's a real word (it appears in the public-domain
ENABLE word list, which has no names, abbreviations or junk like "zxq"), then store the
vectors as float16 to keep the file small.

    uv run python scripts/prepare_vectors.py ~/projects/glove.42B.300d.txt
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORDLIST = ROOT / "data" / "raw" / "enable1.txt"
OUT = ROOT / "data" / "glove50k.npz"
KEEP = 50_000  # real words to keep
SCAN = 400_000  # how many of GloVe's most frequent tokens to look through


def main(glove_path: str) -> None:
    real = {w.strip() for w in WORDLIST.read_text().split()}
    words, vectors = [], []
    with open(glove_path, encoding="utf8") as f:
        for i, line in enumerate(f):
            if i >= SCAN or len(words) >= KEEP:
                break
            word, rest = line.split(" ", 1)
            if word in real and len(word) >= 2:
                words.append(word.upper())
                vectors.append(np.array(rest.split(), dtype=np.float32))
    m = np.stack(vectors)
    m /= np.linalg.norm(m, axis=1, keepdims=True)  # unit length, so a dot product is cosine similarity
    np.savez_compressed(OUT, words=np.array(words), vectors=m.astype(np.float16))
    print(f"kept {len(words)} words from the first {i} GloVe tokens -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
