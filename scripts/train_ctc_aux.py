import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from transformers import (
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    default_data_collator,
    TrainerCallback,
)
from evaluate import load


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


OUT_DIR = "./trocr_small_ctc_aux"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_ROOT = "/storage/shrey/work/handwriting/"
LOG_PATH = os.path.join(DATA_ROOT, "log.txt")

MAX_TARGET_LENGTH = 128

CTC_LOSS_WEIGHT = 0.01
MAX_CTC_LEN = 32

CHARS = list("abcdefghijklmnopqrstuvwxyz")
BLANK_ID = 0
UNK_ID = len(CHARS) + 1
CHAR2ID = {c: i + 1 for i, c in enumerate(CHARS)}
CTC_NUM_LABELS = UNK_ID + 1


def text_to_ctc_ids(text: str):
    text = str(text).lower()
    ids = []
    for ch in text:
        if ch.isalpha():
            ids.append(CHAR2ID.get(ch, UNK_ID))
    ids = ids[:MAX_CTC_LEN]
    return ids


_SEED = 42
set_all_seeds(_SEED)


df = pd.read_csv(LOG_PATH, delimiter=",", on_bad_lines="skip")
if "file_name" not in df.columns or "text" not in df.columns:
    df = pd.read_csv(
        LOG_PATH,
        delimiter=",",
        header=None,
        names=["file_name", "text"],
        on_bad_lines="skip",
    )

df["file_name"] = df["file_name"].astype(str).str.strip()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["file_name"].str.lower() != "file_name"].reset_index(drop=True)
df = df.dropna()
df = df[(df["file_name"] != "") & (df["text"] != "")].reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=0.1, random_state=_SEED)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


class IDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tries = 0
        while tries < 5:
            row = self.df.iloc[idx]
            file_name = row["file_name"]
            text = str(row["text"])

            image_path = os.path.join(self.root_dir, file_name)
            try:
                image = Image.open(image_path).convert("RGB")
                break
            except Exception:
                tries += 1
                idx = (idx + 1) % len(self.df)

        if tries == 5:
            raise RuntimeError(f"Failed to load image after retries. Last path: {image_path}")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = [t if t != self.processor.tokenizer.pad_token_id else -100 for t in labels]

        ctc_ids = text_to_ctc_ids(text)
        ctc_len = len(ctc_ids)
        if ctc_len < MAX_CTC_LEN:
            ctc_ids = ctc_ids + [BLANK_ID] * (MAX_CTC_LEN - ctc_len)
        else:
            ctc_ids = ctc_ids[:MAX_CTC_LEN]
            ctc_len = MAX_CTC_LEN

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
            "ctc_labels": torch.tensor(ctc_ids, dtype=torch.long),
            "ctc_lengths": torch.tensor(ctc_len, dtype=torch.long),
        }


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")

train_dataset = IDataset(DATA_ROOT, train_df, processor, max_target_length=MAX_TARGET_LENGTH)
eval_dataset = IDataset(DATA_ROOT, test_df, processor, max_target_length=MAX_TARGET_LENGTH)

print("Train:", len(train_dataset), " Val:", len(eval_dataset))


model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size

model.generation_config.max_length = 64
model.generation_config.num_beams = 4

enc_hidden_dim = model.encoder.config.hidden_size
model.ctc_head = nn.Linear(enc_hidden_dim, CTC_NUM_LABELS)
model.config.ctc_num_labels = CTC_NUM_LABELS
model.config.ctc_blank_id = BLANK_ID


cer_metric = load("cer")
wer_metric = load("wer")
vocab_size = processor.tokenizer.vocab_size
pad_id = processor.tokenizer.pad_token_id


def compute_metrics(pred):
    pred_ids = pred.predictions
    if isinstance(pred_ids, (tuple, list)):
        pred_ids = pred_ids[0]
    pred_arr = np.array(pred_ids)

    if pred_arr.ndim == 3:
        pred_arr = pred_arr.argmax(axis=-1)
    if np.issubdtype(pred_arr.dtype, np.floating):
        pred_arr = pred_arr.astype(np.int64)
    pred_arr = np.clip(pred_arr, 0, vocab_size - 1)

    labels_arr = np.array(pred.label_ids)
    labels_arr[labels_arr == -100] = pad_id

    pred_str = processor.batch_decode(pred_arr, skip_special_tokens=True)
    label_str = processor.batch_decode(labels_arr, skip_special_tokens=True)

    return {
        "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
    }


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.processor.save_pretrained(ckpt_dir)


class CTCMultiTaskTrainer(Seq2SeqTrainer):
    def __init__(self, *args, ctc_weight=0.01, blank_id=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctc_weight = float(ctc_weight)
        self.blank_id = int(blank_id)
        self.ctc_loss_fn = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ctc_labels = inputs.pop("ctc_labels")
        ctc_lengths = inputs.pop("ctc_lengths")

        outputs = model(
            pixel_values=inputs["pixel_values"],
            labels=inputs["labels"],
            return_dict=True,
        )

        ce_loss = outputs.loss
        ctc_loss_val = torch.tensor(0.0, device=ce_loss.device)

        if self.ctc_weight > 0:
            enc_feats = outputs.encoder_last_hidden_state  # [B,S,D]

            if enc_feats.shape[1] == 577:
                enc_feats = enc_feats[:, 1:, :]

            B, S, D = enc_feats.shape
            H = int(S ** 0.5)
            W = H

            if H * W == S:
                enc_grid = enc_feats.view(B, H, W, D).permute(0, 3, 1, 2)  # [B,D,H,W]
                enc_1d = enc_grid.mean(dim=2, keepdim=True)                 # [B,D,1,W]
                enc_ups = F.interpolate(enc_1d, size=(1, 48), mode="bilinear", align_corners=False)
                enc_seq = enc_ups.squeeze(2).transpose(1, 2)                # [B,T,D], T=48
            else:
                enc_seq = enc_feats.transpose(1, 2).transpose(1, 2)         # no-op, keep [B,S,D]

            if model.ctc_head.weight.device != enc_seq.device:
                model.ctc_head.to(enc_seq.device)

            ctc_logits = model.ctc_head(enc_seq)                            # [B,T,C]
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)   # [T,B,C]

            T, B2, _ = log_probs.shape
            input_lengths = torch.full((B2,), T, dtype=torch.long, device=log_probs.device)

            ctc_loss_val = self.ctc_loss_fn(
                log_probs,
                ctc_labels.to(log_probs.device),
                input_lengths,
                ctc_lengths.to(log_probs.device),
            )

        total_loss = ce_loss + self.ctc_weight * ctc_loss_val
        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if "ctc_labels" in inputs:
            inputs = {k: v for k, v in inputs.items() if k not in ("ctc_labels", "ctc_lengths")}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    overwrite_output_dir=False,
    run_name=os.path.basename(OUT_DIR),
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=48,
    per_device_eval_batch_size=48,
    num_train_epochs=100,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    save_total_limit=4,
    save_safetensors=True,
    remove_unused_columns=False,
    seed=_SEED,
    data_seed=_SEED,
    report_to="wandb",
)


local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
is_main = (local_rank == -1) or (local_rank == 0)

os.environ["WANDB_WATCH"] = "false"
if is_main:
    wandb.init(
        project="tr_OCR_baseline_CTC",
        name=f"ctc_aux_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={"seed": _SEED, "ctc_weight": CTC_LOSS_WEIGHT, "max_ctc_len": MAX_CTC_LEN},
    )
else:
    os.environ["WANDB_MODE"] = "disabled"


trainer = CTCMultiTaskTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    callbacks=[SaveProcessorCallback(processor)],
    ctc_weight=CTC_LOSS_WEIGHT,
    blank_id=BLANK_ID,
)

trainer.train()

model.save_pretrained(OUT_DIR)
processor.save_pretrained(OUT_DIR)

if is_main:
    wandb.finish()
