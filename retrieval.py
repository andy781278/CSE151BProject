import argparse
import json
import os

import bm25s
import Stemmer
from datasets import load_dataset

MAX_CHOICES = 5


def load_corpus(path, limit=None):
    print(f"reading {path}")
    with open(path, encoding="utf-8", errors="ignore") as f:
        docs = [line.strip() for line in f if len(line.strip()) > 20]
    if limit:
        docs = docs[:limit]
    print(f"{len(docs)} sentences")
    return docs


def build_index(docs, stemmer):
    print("tokenizing corpus")
    tokens = bm25s.tokenize(docs, stopwords="en", stemmer=stemmer, show_progress=True)
    print("indexing")
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=True)
    return retriever


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="ARC-V1-Feb2018-2/ARC_Corpus.txt")
    p.add_argument("--dataset", default="allenai/ai2_arc")
    p.add_argument("--configs", nargs="+",
                   default=["ARC-Challenge", "ARC-Easy"])
    p.add_argument("--top_k", type=int, default=20,
                   help="cache this many; sweep smaller k at train time")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N corpus sentences (for testing)")
    p.add_argument("--out", default="retrieved")
    args = p.parse_args()

    stemmer = Stemmer.Stemmer("english")
    docs = load_corpus(args.corpus, args.limit)
    retriever = build_index(docs, stemmer)

    os.makedirs(args.out, exist_ok=True)

    for config in args.configs:
        raw = load_dataset(args.dataset, config)
        for split in ["train", "validation", "test"]:
            queries, keys = [], []
            for ex in raw[split]:
                texts = list(ex["choices"]["text"])
                for c in range(MAX_CHOICES):
                    choice = texts[c] if c < len(texts) else ""
                    queries.append(f"{ex['question']} {choice}".strip())
                    keys.append((ex["id"], c))

            print(f"{config}/{split}: {len(queries)} queries")
            qtokens = bm25s.tokenize(queries, stopwords="en",
                                     stemmer=stemmer, show_progress=False)
            idx, _ = retriever.retrieve(qtokens, k=args.top_k, show_progress=True)

            out = {}
            for (qid, c), row in zip(keys, idx):
                out.setdefault(qid, [None] * MAX_CHOICES)
                out[qid][c] = [docs[i] for i in row]

            path = f"{args.out}/{config}_{split}.json"
            with open(path, "w") as f:
                json.dump(out, f)
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()