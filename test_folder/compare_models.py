# compare_models.py
from sentence_transformers import SentenceTransformer
import numpy as np

pairs = [
    ("Windows 95 operating system crashes",
     "Microsoft OS freezing and blue screen errors"),
    ("atheism and lack of religious belief",
     "why people reject the concept of God"),
    ("encryption and internet privacy",
     "keeping your online data secure and private"),
]

INSTRUCTION = "Represent this sentence for searching relevant passages: "

mini = SentenceTransformer("all-MiniLM-L6-v2")
bge  = SentenceTransformer("BAAI/bge-small-en-v1.5")

print(f"{'Pair':<45} {'MiniLM':>8} {'BGE':>8} {'Better':>8}")
print("-" * 75)

for a, b in pairs:
    # MiniLM — no instruction
    va_m = mini.encode(a, normalize_embeddings=True)
    vb_m = mini.encode(b, normalize_embeddings=True)
    sim_mini = float(np.dot(va_m, vb_m))

    # BGE — instruction on query side only
    va_b = bge.encode(INSTRUCTION + a, normalize_embeddings=True)
    vb_b = bge.encode(INSTRUCTION + b, normalize_embeddings=True)
    sim_bge = float(np.dot(va_b, vb_b))

    better = "BGE ✓" if sim_bge > sim_mini else "MiniLM ✓"
    print(f"{a[:44]:<45} {sim_mini:>8.4f} {sim_bge:>8.4f} {better:>8}")