import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
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


OUT_DIR = "./trocr_large_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_ROOT = "/storage/shrey/work/handwriting/"
LOG_PATH = os.path.join(DATA_ROOT, "log.txt")

_SEED = 999
set_all_seeds(_SEED)

df = pd.read_csv(LOG_PATH, delimiter=",", on_bad_lines="skip")
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)

df["file_name"] = df["file_name"].astype(str).str.strip()
df["text"] = df["text"].astype(str).str.strip()
df = df[(df["file_name"].str.len() > 0) & (df["text"].str.len() > 0)].reset_index(drop=True)

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
        file_name = self.df.loc[idx, "file_name"]
        text = str(self.df.loc[idx, "text"])

        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = [t if t != self.processor.tokenizer.pad_token_id else -100 for t in labels]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")

train_dataset = IDataset(DATA_ROOT, train_df, processor)
eval_dataset = IDataset(DATA_ROOT, test_df, processor)

print("Train:", len(train_dataset), " Val:", len(eval_dataset))


cer_metric = load("cer")
wer_metric = load("wer")
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


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.processor.save_pretrained(ckpt_dir)


model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size

model.generation_config.max_length = 64
model.generation_config.num_beams = 4


training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    overwrite_output_dir=False,
    run_name=os.path.basename(OUT_DIR),
    predict_with_generate=True,

    evaluation_strategy="steps",
    logging_steps=20,
    save_steps=400,
    eval_steps=400,

    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    num_train_epochs=100,
    learning_rate=5e-5,
    fp16=True,

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
        project="tr_OCR_fine_tune2",
        name=f"{os.path.basename(OUT_DIR)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={"seed": _SEED},
    )
else:
    os.environ["WANDB_MODE"] = "disabled"


trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    callbacks=[SaveProcessorCallback(processor)],
)

trainer.train()

model.save_pretrained(OUT_DIR)
processor.save_pretrained(OUT_DIR)

if is_main:
    wandb.finish()
