# train_multitask.py
import os
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    TrainerCallback,
)
from evaluate import load as load_metric


# -------------------------
# utils
# -------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_log_file_two_cols_csv(path: str, encoding="utf-8"):
    """
    Reads a 2-column CSV-like log: filename,text
    Robust to commas inside text by splitting on the first comma only.
    Skips empty/bad lines.
    """
    rows = []
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," not in line:
                continue
            fn, txt = line.split(",", 1)
            fn = fn.strip()
            txt = txt.strip()
            if not fn or not txt:
                continue
            rows.append((fn, txt))
    return pd.DataFrame(rows, columns=["file_name", "text"])


def load_char2bits(csv_path: str):
    d = pd.read_csv(csv_path)

    if "alphabet" not in [c.lower() for c in d.columns]:
        d.columns = ["alphabet"] + list(d.columns[1:])

    bit_cols = [c for c in d.columns if c.lower() != "alphabet"]
    if not bit_cols:
        raise ValueError("Could not find bitstring column in alphabet CSV.")
    bit_col = bit_cols[0]

    first_bits = str(d.iloc[0][bit_col]).strip()
    char_bits = len(first_bits)

    char2bits = {}
    for _, row in d.iterrows():
        ch = str(row["alphabet"]).strip()  # keep case
        bits_str = str(row[bit_col]).strip()
        if len(bits_str) != char_bits:
            raise ValueError(f"Char {ch} has {len(bits_str)} bits, expected {char_bits}")
        vec = torch.tensor([int(b) for b in bits_str], dtype=torch.float32)
        char2bits[ch] = vec

    return char2bits, char_bits


# -------------------------
# dataset
# -------------------------
class IDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        df,
        processor,
        char2bits,
        char_bits,
        max_chars_per_token,
        max_target_length,
    ):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_target_length = max_target_length

        self.char2bits = char2bits
        self.char_bits = char_bits
        self.max_chars_per_token = max_chars_per_token
        self.binary_dim = char_bits * max_chars_per_token

        self._zero_char = torch.zeros(self.char_bits, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = str(self.df.loc[idx, "file_name"])
        text = str(self.df.loc[idx, "text"])

        img_path = os.path.join(self.root_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        tok = self.processor.tokenizer
        labels = tok(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = [t if t != tok.pad_token_id else -100 for t in labels]

        bit_labels = self._build_bit_labels(text, labels, tok)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
            "bit_labels": bit_labels,
        }

    def _build_bit_labels(self, text, labels, tokenizer):
        seq_len = len(labels)
        bit_labels = torch.zeros(seq_len, self.binary_dim, dtype=torch.float32)

        text_chars = [ch for ch in text if ch.isalpha()]
        char_idx = 0

        label_ids_for_tokens = [t if t != -100 else tokenizer.pad_token_id for t in labels]
        tokens = tokenizer.convert_ids_to_tokens(label_ids_for_tokens)

        special_tokens = {
            tokenizer.pad_token,
            tokenizer.eos_token,
            tokenizer.bos_token,
            tokenizer.cls_token,
            tokenizer.sep_token,
        }

        for t_idx, (token_id, token_str) in enumerate(zip(labels, tokens)):
            if token_id == -100:
                continue
            if token_str in special_tokens:
                continue

            tok_clean = token_str.replace("▁", "").replace("Ġ", "")
            token_chars = [c for c in tok_clean if c.isalpha()]
            if not token_chars:
                continue

            bits_list = []
            for _ in token_chars:
                if char_idx >= len(text_chars):
                    break
                ch_raw = text_chars[char_idx]  # keep case
                char_idx += 1
                bits_list.append(self.char2bits.get(ch_raw, self._zero_char))

            if not bits_list:
                continue

            token_vec = torch.cat(bits_list, dim=0)

            if token_vec.numel() > self.binary_dim:
                token_vec = token_vec[: self.binary_dim]
            elif token_vec.numel() < self.binary_dim:
                pad_len = self.binary_dim - token_vec.numel()
                token_vec = torch.cat([token_vec, torch.zeros(pad_len, dtype=torch.float32)], dim=0)

            bit_labels[t_idx] = token_vec

        return bit_labels


# -------------------------
# callbacks / trainer
# -------------------------
class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.processor.save_pretrained(ckpt_dir)


class MultiTaskSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, binary_loss_weight=0.05, struct_mode="multi", **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_loss_weight = float(binary_loss_weight)
        self.struct_mode = str(struct_mode).lower().strip()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        bit_labels = inputs.pop("bit_labels", None)

        outputs = model(
            pixel_values=inputs["pixel_values"],
            labels=inputs["labels"],
            output_hidden_states=True,
        )

        ce_loss = outputs.loss
        total_loss = ce_loss
        bce_loss_value = None

        if bit_labels is not None:
            labels = inputs["labels"]
            valid_mask = (labels != -100)

            if valid_mask.any():
                hs_all = outputs.decoder_hidden_states
                hs_layers = hs_all[1:]  # drop embeddings
                L = len(hs_layers)

                mode = self.struct_mode
                if mode == "last":
                    layer_indices = [L - 1]
                elif mode == "middle":
                    layer_indices = [L // 2]
                elif mode == "first":
                    layer_indices = [0]
                elif mode in ("first_middle_last", "fml"):
                    layer_indices = [0, L // 2, L - 1]
                elif mode in ("first_last", "last_first"):
                    layer_indices = [0, L - 1]
                elif mode in ("last_middle", "middle_last"):
                    layer_indices = [L // 2, L - 1]
                elif mode in ("middle_first", "first_middle"):
                    layer_indices = [0, L // 2]
                elif mode == "multi":
                    layer_indices = sorted(set([L // 3, (2 * L) // 3, L - 1]))
                elif mode == "all":
                    layer_indices = list(range(L))
                else:
                    layer_indices = [L - 1]

                layer_indices = sorted(set(layer_indices))

                bce_fn = nn.BCEWithLogitsLoss()
                active_bits = bit_labels[valid_mask]

                layer_losses = []
                for li in layer_indices:
                    dec_h = hs_layers[li]
                    logits = model.binary_head(dec_h)
                    active_logits = logits[valid_mask]
                    layer_losses.append(bce_fn(active_logits, active_bits))

                bce_loss_value = sum(layer_losses) / max(1, len(layer_losses))
                total_loss = ce_loss + self.binary_loss_weight * bce_loss_value

        self.log({"ce_loss": float(ce_loss.detach().cpu())})
        if bce_loss_value is not None:
            self.log({"bce_loss": float(bce_loss_value.detach().cpu())})

        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if "bit_labels" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "bit_labels"}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


def compute_metrics_factory(processor):
    cer_metric = load_metric("cer")
    wer_metric = load_metric("wer")
    vocab_size = processor.tokenizer.vocab_size
    pad_id = processor.tokenizer.pad_token_id

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        if isinstance(pred_ids, (tuple, list)):
            pred_ids = pred_ids[0]

        pred_arr = np.array(pred_ids)
        if pred_arr.ndim == 3:
            pred_arr = pred_arr.argmax(axis=-1)
        if np.issubdtype(pred_arr.dtype, np.floating):
            pred_arr = pred_arr.astype(np.int64)
        pred_arr = np.clip(pred_arr, 0, vocab_size - 1)

        labels_arr = np.array(labels_ids)
        labels_arr[labels_arr == -100] = pad_id

        pred_str = processor.batch_decode(pred_arr, skip_special_tokens=True)
        label_str = processor.batch_decode(labels_arr, skip_special_tokens=True)

        return {
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        }

    return compute_metrics


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Folder containing images (and usually log.txt)")
    ap.add_argument("--log_path", type=str, required=True, help="Path to log file: filename,text")
    ap.add_argument("--alphabet_csv", type=str, required=True, help="CSV mapping alphabet -> bitstring")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for checkpoints")

    ap.add_argument("--model_name", type=str, default="microsoft/trocr-base-handwritten")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--max_rows", type=int, default=None)

    ap.add_argument("--max_target_length", type=int, default=128)
    ap.add_argument("--max_chars_per_token", type=int, default=8)
    ap.add_argument("--lambda_bce", type=float, default=0.05)
    ap.add_argument("--struct_mode", type=str, default="multi")

    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--eval_steps", type=int, default=400)
    ap.add_argument("--save_steps", type=int, default=400)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_total_limit", type=int, default=5)
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="tr_OCR_multitask")
    ap.add_argument("--wandb_name", type=str, default=None)

    ap.add_argument("--freeze_encoder", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_all_seeds(args.seed)

    df = read_log_file_two_cols_csv(args.log_path)
    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("log file is empty after filtering. Check format: filename,text")

    # quick sanity check
    for fn in df["file_name"].head(25):
        p = os.path.join(args.data_root, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing image: {p}")

    train_df, val_df = train_test_split(df, test_size=args.test_size, random_state=args.seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    char2bits, char_bits = load_char2bits(args.alphabet_csv)
    binary_dim = char_bits * args.max_chars_per_token

    processor = TrOCRProcessor.from_pretrained(args.model_name)

    train_dataset = IDataset(
        root_dir=args.data_root,
        df=train_df,
        processor=processor,
        char2bits=char2bits,
        char_bits=char_bits,
        max_chars_per_token=args.max_chars_per_token,
        max_target_length=args.max_target_length,
    )
    eval_dataset = IDataset(
        root_dir=args.data_root,
        df=val_df,
        processor=processor,
        char2bits=char2bits,
        char_bits=char_bits,
        max_chars_per_token=args.max_chars_per_token,
        max_target_length=args.max_target_length,
    )

    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.binary_head = nn.Linear(model.decoder.config.hidden_size, binary_dim)

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_main = (local_rank in (-1, 0))

    report_to = []
    if args.use_wandb:
        if is_main:
            import wandb  # lazy import

            os.environ["WANDB_WATCH"] = "false"
            run_name = args.wandb_name or f"{os.path.basename(args.out_dir)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model": args.model_name,
                    "seed": args.seed,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "bs": args.bs,
                    "max_target_length": args.max_target_length,
                    "char_bits": char_bits,
                    "max_chars_per_token": args.max_chars_per_token,
                    "binary_dim": binary_dim,
                    "lambda_bce": args.lambda_bce,
                    "struct_mode": args.struct_mode,
                    "freeze_encoder": args.freeze_encoder,
                },
            )
            report_to = ["wandb"]
        else:
            os.environ["WANDB_MODE"] = "disabled"

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=False,
        run_name=os.path.basename(args.out_dir),
        predict_with_generate=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=bool(args.fp16),
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        remove_unused_columns=False,
        seed=args.seed,
        report_to=report_to,
    )

    trainer = MultiTaskSeq2SeqTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        args=training_args,
        compute_metrics=compute_metrics_factory(processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[SaveProcessorCallback(processor)],
        binary_loss_weight=args.lambda_bce,
        struct_mode=args.struct_mode,
    )

    print(f"Train examples: {len(train_dataset)} | Val examples: {len(eval_dataset)}")
    trainer.train()

    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)

    if args.use_wandb and is_main:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
