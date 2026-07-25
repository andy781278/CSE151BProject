import argparse


def get_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--dataset", default="allenai/ai2_arc")
    p.add_argument("--config", default="ARC-Challenge",
                   choices=["ARC-Challenge", "ARC-Easy"])
    p.add_argument("--max_len", type=int, default=128)

    # model
    p.add_argument("--model_name", default="microsoft/deberta-v3-large")

    # training
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=8e-6)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42069)

    # bookkeeping
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--run_name", default=None)

    return p.parse_args()