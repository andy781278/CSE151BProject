import json, os
from datasets import load_dataset
from sentence_transformers import CrossEncoder

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SRC, OUT = "retrieved", "retrieved_rr"
FILES = [("ARC-Challenge", "train"), ("ARC-Challenge", "validation"),
         ("ARC-Challenge", "test"), ("ARC-Easy", "train")]

ce = CrossEncoder(MODEL, max_length=256, device="cuda")
os.makedirs(OUT, exist_ok=True)

for config, split in FILES:
    raw = load_dataset("allenai/ai2_arc", config)[split]
    qtext = {ex["id"]: (ex["question"], list(ex["choices"]["text"])) for ex in raw}
    ctx = json.load(open(f"{SRC}/{config}_{split}.json"))

    pairs, index = [], []
    for qid, slots in ctx.items():
        question, choices = qtext[qid]
        for c, cands in enumerate(slots):
            if not cands:
                continue
            choice = choices[c] if c < len(choices) else ""
            query = f"{question} {choice}".strip()
            for j, sent in enumerate(cands):
                pairs.append((query, sent))
                index.append((qid, c, j))

    print(f"{config}/{split}: {len(pairs)} pairs")
    scores = ce.predict(pairs, batch_size=256, show_progress_bar=True)

    buckets = {}
    for (qid, c, j), s in zip(index, scores):
        buckets.setdefault((qid, c), []).append((float(s), ctx[qid][c][j]))

    out = {}
    for (qid, c), scored in buckets.items():
        scored.sort(key=lambda x: -x[0])
        out.setdefault(qid, [None] * 5)[c] = [s for _, s in scored]

    with open(f"{OUT}/{config}_{split}.json", "w") as f:
        json.dump(out, f)
    print(f"  wrote {OUT}/{config}_{split}.json")