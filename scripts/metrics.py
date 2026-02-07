import os
import argparse
import pandas as pd
from nltk.metrics import edit_distance


def normalize_prediction(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("#", "").replace(".", "").replace(":", "").replace("+", "").replace(",", "")
    s = " ".join(s.strip().lower().split())
    return s


def compute_cer(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, gt) / len(gt)


def compute_wer_single_word(pred: str, gt: str) -> float:
    gt_tokens = gt.split()
    pred_tokens = pred.split()

    if len(gt_tokens) == 0:
        return 0.0 if len(pred_tokens) == 0 else 1.0

    # single-word benchmark: treat as correct/incorrect word match
    if len(gt_tokens) == 1 and len(pred_tokens) == 1:
        return 0.0 if pred_tokens[0] == gt_tokens[0] else 1.0

    # fallback if something produces multiple tokens
    return edit_distance(pred_tokens, gt_tokens) / len(gt_tokens)


def len_bin(n: int) -> str:
    if n <= 3:
        return "1-3"
    if n <= 6:
        return "4-6"
    if n <= 9:
        return "7-9"
    return "10+"


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def pick_col(df: pd.DataFrame, wanted: str) -> str:
    # exact match first
    if wanted in df.columns:
        return wanted
    # case-insensitive match
    low = {c.lower(): c for c in df.columns}
    if wanted.lower() in low:
        return low[wanted.lower()]
    raise ValueError(f"Could not find column '{wanted}' in: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="trocr_results.xlsx", help="XLSX/CSV with GT and prediction columns")
    args = ap.parse_args()

    df = read_table(args.infile)

    gt_col = pick_col(df, "Ground Truth")
    pred_col = pick_col(df, "Predicted Text")

    df["gt"] = df[gt_col].astype(str).map(normalize_prediction)
    df["pred"] = df[pred_col].astype(str).map(normalize_prediction)

    df["exact"] = (df["pred"] == df["gt"]).astype(int)
    df["cer"] = [compute_cer(p, g) for p, g in zip(df["pred"], df["gt"])]
    df["wer_single"] = [compute_wer_single_word(p, g) for p, g in zip(df["pred"], df["gt"])]

    n = len(df)
    word_acc = df["exact"].mean() if n else float("nan")
    cer_avg = df["cer"].mean() if n else float("nan")
    wer_avg = df["wer_single"].mean() if n else float("nan")

    print("\n=== OVERALL ===")
    print(f"N            : {n}")
    print(f"WordAcc      : {word_acc:.6f}")
    print(f"CER          : {cer_avg:.6f}")
    print(f"WER(single)  : {wer_avg:.6f}")

    df["len"] = df["gt"].map(len)
    df = df[df["len"] > 0].copy()
    df["LenBin"] = df["len"].map(len_bin)

    bybin = (
        df.groupby("LenBin", sort=False)
          .agg(
              N=("LenBin", "size"),
              WordAcc=("exact", "mean"),
              CER=("cer", "mean"),
              WER_single=("wer_single", "mean"),
          )
          .reindex(["1-3", "4-6", "7-9", "10+"])
          .reset_index()
    )

    print("\n=== LENGTH-BIN BREAKDOWN ===")
    print(bybin)


if __name__ == "__main__":
    main()
