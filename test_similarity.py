# test_similarity.py
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

pairs = [
    # ── EASY (expect 0.80+) ─────────────────────────────────────────────
    (
        "How do I install Python on Windows?",
        "Steps to set up Python on a Windows machine"
    ),
    (
        "What is the best graphics card for gaming?",
        "Which GPU should I buy for playing video games?"
    ),
    (
        "symptoms of a heart attack",
        "warning signs that someone is having a heart attack"
    ),
    (
        "how to fix a car engine",
        "troubleshooting problems with my car engine"
    ),
    (
        "Israel Palestine conflict history",
        "history of the Arab Israeli conflict"
    ),

    # ── MEDIUM (expect 0.65–0.79) ────────────────────────────────────────
    (
        "NASA space shuttle launch",
        "rocket launches and space missions"
    ),
    (
        "encryption and internet privacy",
        "keeping your online data secure and private"
    ),
    (
        "Christianity and the existence of God",
        "religious arguments for God belief"
    ),
    (
        "hockey game scores NHL",
        "professional ice hockey league results"
    ),
    (
        "Windows 95 operating system crashes",
        "Microsoft OS freezing and blue screen errors"
    ),

    # ── HARD (expect 0.50–0.64) ──────────────────────────────────────────
    (
        "US government gun control debate Congress",
        "Second Amendment firearms legislation in America"
    ),
    (
        "electric car battery technology",
        "Tesla lithium ion range and charging"
    ),
    (
        "atheism and lack of religious belief",
        "why people reject the concept of God"
    ),
    (
        "steroids in professional baseball",
        "drug use and doping scandals in MLB"
    ),
    (
        "SCSI hard drive setup",
        "configuring disk controllers and storage devices"
    ),

    # ── TRICK (expect below 0.55 — should NOT hit cache) ─────────────────
    (
        "NASA space shuttle launch mission",
        "gun control laws in Congress"           # totally different topic
    ),
    (
        "Christian prayer and Bible study",
        "ice hockey playoff results"             # totally different topic
    ),
    (
        "how to fix a car engine",
        "Middle East political conflict"         # totally different topic
    ),
    (
        "Python programming tutorial",
        "baseball World Series winner"           # totally different topic
    ),
    (
        "Windows operating system install",
        "God existence religious philosophy"     # totally different topic
    ),
]

print(f"{'Score':<8} {'Category':<10} Pair")
print("-" * 80)
for i, (a, b) in enumerate(pairs):
    va = model.encode(a, normalize_embeddings=True)
    vb = model.encode(b, normalize_embeddings=True)
    sim = float(np.dot(va, vb))

    if i < 5:
        category = "EASY"
    elif i < 10:
        category = "MEDIUM"
    elif i < 15:
        category = "HARD"
    else:
        category = "TRICK"

    print(f"{sim:.4f}   {category:<10} '{a[:40]}...'")
    print(f"{'':>8}   {'':10} vs '{b[:40]}...'")
    print()