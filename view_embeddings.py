import numpy as np
from pathlib import Path
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

cache_dir = Path(__file__).parent.parent.parent / "cache"
files = sorted(cache_dir.glob("*.npy"))

print(f"Total: {len(files)} embeddings\n")

page = 0
page_size = 50
while True:
    start = page * page_size
    end = min(start + page_size, len(files))
    if start >= len(files):
        break
    print(f"--- Page {page+1} ({start+1}-{end} of {len(files)}) ---")
    for i in range(start, end):
        f = files[i]
        data = np.load(f, allow_pickle=True).item()
        text = data["text"][:70].replace("\n", " ")
        print(f"  [{i+1}] {f.stem} | {text}")
    print()
    if end < len(files):
        inp = input("Press Enter for next page, q to quit: ")
        if inp.strip().lower() == "q":
            break
        page += 1
    else:
        break
