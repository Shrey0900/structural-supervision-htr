# Structural Auxiliary Supervision for OCR of Low-Legibility Handwritten Answers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"Structural Auxiliary Supervision for OCR of Low-Legibility Handwritten Answers"** by Shrey Chandola, Manikandan Ravikiran, and Rohit Saluja (IIT Mandi, India).

> **Note**: This paper is currently under review. The preprint will be made available soon.

## 📄 Abstract

We propose a fine-tuning strategy for transformer-based OCR models (TrOCR) that incorporates auxiliary structural supervision to improve recognition of low-legibility handwritten cloze-form answers. The decoder is trained to predict 32-bit stroke-direction codes alongside text during fine-tuning, achieving state-of-the-art performance on a challenging bad-handwriting benchmark.

## 🎯 Key Results

- **Best Performance**: TrOCR-Large reaches **1.69% CER** and **4.58% WER**
- **Significant improvements** over standard fine-tuning (26-41% relative CER reduction)
- Evaluated on 3,000 real student word images with heavily overlapped handwriting
- **Optimal hyperparameters**: λ=0.05, multi-layer loss placement (L/3, 2L/3, L)

---

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Standard Fine-tuning (Baseline)](#1-standard-fine-tuning-baseline)
  - [2. Auxiliary Supervision Fine-tuning](#2-auxiliary-supervision-fine-tuning)
  - [3. CTC Auxiliary Loss (Baseline)](#3-ctc-auxiliary-loss-baseline)
  - [4. Inference & Evaluation](#4-inference--evaluation)
- [Stroke Direction Codes](#stroke-direction-codes)
- [Expected Results](#expected-results)
- [Citation](#citation)
- [License](#license)

---

## 🚀 Installation

### Prerequisites

- Python 3.10
- CUDA-capable GPU (recommended for training)
- 16GB+ GPU RAM for TrOCR-Large (8GB for Base/Small)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Shrey0900/structural-supervision-htr.git
cd structural-supervision-htr
```

2. **Create conda environment (recommended):**
```bash
conda env create -f environment.yml
conda activate trocr-struct
```

**OR using pip:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies

Main packages:
- `transformers==4.45.2` - TrOCR and training
- `torch` (with CUDA support recommended)
- `datasets==3.5.0` - Data handling
- `evaluate==0.4.3` - Metrics (CER, WER)
- `wandb==0.22.3` - Experiment tracking (optional)
- `pandas`, `pillow`, `numpy`, `tqdm`

---

## 📊 Dataset

### Bad-Handwriting Benchmark

This project uses the **Bad-Handwriting Benchmark** from our companion paper:
> Chandola et al., "How far are we from automatic grading of handwritten cloze form questions?", AIED 2025.

**Dataset characteristics:**
- 3,000 single-word handwritten responses from Indian school students
- Cropped word images with ground-truth transcriptions
- Low-legibility writing with merged strokes, irregular baselines, and severe distortions

### Dataset Format

Your dataset should be organized as:

```
data/
├── images/
│   ├── word_001.png
│   ├── word_002.png
│   └── ...
└── log.txt  (or annotations.csv)
```

**log.txt / annotations.csv format:**
```csv
filename,text
word_001.png,Turbulence
word_002.png,Android
word_003.png,Logarithm
...
```

**Important notes:**
- Each line: `filename,text` (comma-separated)
- Text can contain commas (only first comma is used as delimiter)
- Images should be in RGB or grayscale (will be converted to RGB)
- No fixed image size required (will be resized to 384×384)

### Get the Dataset

**Repo Link**: https://github.com/Shrey0900/ClozeFormHandwriting

---

## 📁 Project Structure

```
structural-supervision-htr/
├── assets/
│   └── stroke_codes/
│       ├── alphabet_binary_32.csv    # 32-bit stroke codes for A-Z, a-z
│       └── README.md                 # Stroke code documentation
├── scripts/
│   ├── train_baseline.py             # Standard TrOCR fine-tuning (CE only)
│   ├── train_multitask.py            # Our method: CE + structural supervision
│   ├── train_ctc_aux.py              # Baseline: CE + CTC auxiliary loss
│   ├── predict_folder.py             # Batch inference on test images
│   └── metrics.py                    # Compute CER, WER from predictions
├── environment.yml                   # Conda environment
├── requirements.txt                  # pip dependencies
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## 🔧 Usage

### 1. Standard Fine-tuning (Baseline)

Train TrOCR with **only cross-entropy loss** (standard approach):

**Edit the script first**: Update paths in `scripts/train_baseline.py`:
```python
# Line 35-36
DATA_ROOT = "/path/to/your/data/"  # Folder containing images
LOG_PATH = os.path.join(DATA_ROOT, "log.txt")  # Annotations file
```

**Then run:**
```bash
python scripts/train_baseline.py
```

**Key hyperparameters in the script:**
- Model: `microsoft/trocr-large-handwritten`
- Batch size: 24 (per GPU)
- Learning rate: 5e-5
- Epochs: 100
- Evaluation strategy: every 400 steps
- Output directory: `./trocr_large_baseline`

**Outputs:**
- Checkpoints saved to `./trocr_large_baseline/checkpoint-{step}/`
- Best model (lowest CER) saved to `./trocr_large_baseline/`
- WandB logging (project: `tr_OCR_fine_tune2`)

---

### 2. Auxiliary Supervision Fine-tuning

Train TrOCR with **structural auxiliary supervision** (our method):

```bash
python scripts/train_multitask.py \
    --data_root /path/to/your/data \
    --log_path /path/to/your/data/log.txt \
    --alphabet_csv ./assets/stroke_codes/alphabet_binary_32.csv \
    --out_dir ./outputs/trocr_large_auxiliary \
    --model_name microsoft/trocr-large-handwritten \
    --bs 24 \
    --lr 5e-5 \
    --epochs 100 \
    --lambda_bce 0.05 \
    --struct_mode multi \
    --max_chars_per_token 8 \
    --fp16 \
    --use_wandb \
    --wandb_project tr_OCR_multitask
```

**Key Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_root` | Directory containing images | **Required** |
| `--log_path` | Path to annotations file (CSV format) | **Required** |
| `--alphabet_csv` | Path to stroke code CSV | **Required** |
| `--out_dir` | Output directory for checkpoints | **Required** |
| `--model_name` | TrOCR checkpoint | `microsoft/trocr-base-handwritten` |
| `--lambda_bce` | Auxiliary loss weight (λ) | `0.05` (best from paper) |
| `--struct_mode` | Layer placement for auxiliary loss | `multi` (best from paper) |
| `--max_chars_per_token` | Max characters per token (M) | `8` |
| `--bs` | Batch size per device | `48` |
| `--lr` | Learning rate | `5e-5` |
| `--epochs` | Number of training epochs | `100` |
| `--fp16` | Enable mixed precision training | Flag |
| `--freeze_encoder` | Freeze encoder weights | Flag |

**Layer placement options** (`--struct_mode`):
- `multi` ✅ **(Best, recommended)**: Apply loss at L/3, 2L/3, L layers
- `last`: Apply loss only at final decoder layer
- `fml`: Apply loss at first, middle, last layers
- `all`: Apply loss at all decoder layers

**Examples:**

**TrOCR-Small:**
```bash
python scripts/train_multitask.py \
    --data_root ./data \
    --log_path ./data/log.txt \
    --alphabet_csv ./assets/stroke_codes/alphabet_binary_32.csv \
    --out_dir ./outputs/trocr_small_auxiliary \
    --model_name microsoft/trocr-small-handwritten \
    --bs 48 \
    --lambda_bce 0.05 \
    --struct_mode multi \
    --fp16
```

**TrOCR-Base:**
```bash
python scripts/train_multitask.py \
    --data_root ./data \
    --log_path ./data/log.txt \
    --alphabet_csv ./assets/stroke_codes/alphabet_binary_32.csv \
    --out_dir ./outputs/trocr_base_auxiliary \
    --model_name microsoft/trocr-base-handwritten \
    --bs 48 \
    --lambda_bce 0.05 \
    --struct_mode multi \
    --fp16
```

**Ablation: Testing different λ values:**
```bash
for lambda in 0.005 0.01 0.05 0.1 0.5; do
    python scripts/train_multitask.py \
        --data_root ./data \
        --log_path ./data/log.txt \
        --alphabet_csv ./assets/stroke_codes/alphabet_binary_32.csv \
        --out_dir ./outputs/trocr_large_lambda_${lambda} \
        --model_name microsoft/trocr-large-handwritten \
        --lambda_bce $lambda \
        --struct_mode multi \
        --bs 24 \
        --fp16
done
```

**Ablation: Testing layer configurations:**
```bash
for mode in last fml multi all; do
    python scripts/train_multitask.py \
        --data_root ./data \
        --log_path ./data/log.txt \
        --alphabet_csv ./assets/stroke_codes/alphabet_binary_32.csv \
        --out_dir ./outputs/trocr_large_${mode} \
        --model_name microsoft/trocr-large-handwritten \
        --lambda_bce 0.05 \
        --struct_mode $mode \
        --bs 24 \
        --fp16
done
```

---

### 3. CTC Auxiliary Loss (Baseline)

Train TrOCR with **CTC auxiliary loss** (comparison baseline from paper):

**Edit the script first**: Update paths in `scripts/train_ctc_aux.py`:
```python
# Line 37-38
DATA_ROOT = "/path/to/your/data/"
LOG_PATH = os.path.join(DATA_ROOT, "log.txt")
```

**Then run:**
```bash
python scripts/train_ctc_aux.py
```

**Note**: The paper shows CTC auxiliary loss does **not** improve performance on this benchmark (actually degrades slightly). This script is provided for reproducibility.

---

### 4. Inference & Evaluation

#### A. Batch Inference on Test Images

Run inference on a folder of test images:

```bash
python scripts/predict_folder.py \
    --ckpt ./outputs/trocr_large_auxiliary \
    --data_dir ./data/test_images \
    --out_csv ./predictions.csv \
    --out_xlsx ./predictions.xlsx \
    --device cuda \
    --fp16 \
    --max_length 64 \
    --num_beams 4
```

**Arguments:**
- `--ckpt`: Path to trained model checkpoint directory
- `--data_dir`: Folder with test images
- `--out_csv`: Output CSV file with predictions
- `--out_xlsx`: Output Excel file (optional)
- `--device`: `cuda` or `cpu` (auto-detected if not specified)
- `--fp16`: Use mixed precision for faster inference
- `--max_length`: Max generation length
- `--num_beams`: Beam search width

**Ground truth parsing**: The script extracts ground truth from filenames:
```
word_123_Turbulence.png → Ground truth = "Turbulence"
```
Assumes format: `[prefix]_[groundtruth].png`

#### B. Compute Metrics

Calculate CER, WER, and accuracy from predictions:

```bash
python scripts/metrics.py --infile ./predictions.xlsx
```

**Output:**
```
=== OVERALL ===
N            : 600
WordAcc      : 0.954167
CER          : 0.016900
WER(single)  : 0.045833

=== LENGTH-BIN BREAKDOWN ===
  LenBin    N   WordAcc       CER  WER_single
0    1-3   45  0.977778  0.008889    0.022222
1    4-6  234  0.965812  0.012821    0.034188
2    7-9  198  0.949495  0.017172    0.050505
3    10+  123  0.934959  0.024797    0.065041
```

**Metrics:**
- **WordAcc**: Exact word match accuracy
- **CER**: Character Error Rate (normalized edit distance at character level)
- **WER(single)**: Word Error Rate (for single-word benchmark)
- **Length bins**: Performance breakdown by word length

---

## 🎨 Stroke Direction Codes

The auxiliary supervision uses **32-bit binary stroke-direction codes** for Latin characters (A-Z, a-z).

### Encoding

Each character is represented as a sequence of **up to 8 axis-aligned stroke moves**:
- **4 directions**: Up (↑), Down (↓), Left (←), Right (→)
- **4-way one-hot encoding**: 
  - Up = `0100`
  - Down = `0001`
  - Left = `0010`
  - Right = `1000`
- **8 steps × 4 bits = 32-bit signature**
- Unused steps are padded with `0000`

### Examples

| Character | Stroke Sequence | 32-bit Binary Code |
|-----------|----------------|-------------------|
| **a** | ← ↓ → ↑ ↓ → ∅ ∅ | `00100100000110000100000100000000` |
| **B** | ↑ ↑ → ↓ ← → ↓ ← | `10001000000101000010000101000010` |
| **f** | ↑ ← → → ← ↑ → ∅ | `10000001001010000001000000000000` |
| **W** | ↓ ↓ → ↑ ↓ → ↑ ↑ | `01000100000110000100000110001000` |

The full mapping is defined in `assets/stroke_codes/alphabet_binary_32.csv`.

### Multi-character tokens

For tokenizers that produce multi-character tokens:
- Concatenate stroke codes for all characters in the token
- Cap at `M=8` characters per token (configurable via `--max_chars_per_token`)
- Pad to fixed dimension: `32 × M = 256 bits`

**Example**: Token "and" → concatenate codes for 'a', 'n', 'd' → 96 bits, pad to 256 bits

---

## 📊 Expected Results

Based on the paper, you should expect the following performance on the **test set (600 images)**:

### Main Results

| Model | Training Method | CER (%) ↓ | WER (%) ↓ | WordAcc (%) ↑ |
|-------|----------------|-----------|-----------|---------------|
| TrOCR-Small | Baseline (CE only) | 4.31 | 8.50 | 91.50 |
| TrOCR-Small | + Auxiliary (λ=0.05) | **3.82** | **7.50** | **92.50** |
| TrOCR-Base | Baseline (CE only) | 3.70 | 8.08 | 91.92 |
| TrOCR-Base | + Auxiliary (λ=0.05) | **2.71** | **5.91** | **94.09** |
| TrOCR-Large | Baseline (CE only) | 2.89 | 7.00 | 93.00 |
| TrOCR-Large | + Auxiliary (λ=0.05) | **1.69** | **4.58** | **95.42** |

### Comparison with other methods

| Model | CER (%) | WER (%) |
|-------|---------|---------|
| Tesseract (off-the-shelf) | 77.50 | 100.00 |
| Google Cloud Vision | 23.45 | 58.66 |
| DeepSeek-VL2-Small (zero-shot) | 21.80 | 46.16 |
| Qwen2-VL-7B (zero-shot) | 24.17 | 45.08 |
| Qwen2-VL-7B (LoRA fine-tuned) | 13.16 | 24.75 |
| **TrOCR-Large + Auxiliary (Ours)** | **1.69** | **4.58** |

### Ablation: Auxiliary loss weight (λ)

| λ | TrOCR-Small CER | TrOCR-Base CER | TrOCR-Large CER |
|---|-----------------|----------------|-----------------|
| 0.005 | 4.25 | 2.80 | 2.85 |
| 0.01 | 3.89 | 3.47 | 2.41 |
| **0.05** ✅ | **3.82** | **2.71** | **1.69** |
| 0.1 | 3.91 | 2.89 | 2.08 |
| 0.5 | 3.97 | 3.48 | 2.78 |

### Ablation: Layer placement (TrOCR-Large, λ=0.05)

| Placement | Layers Supervised | CER (%) | WER (%) |
|-----------|------------------|---------|---------|
| Last | [12] | 2.19 | 5.75 |
| FML | [1, 6, 12] | 2.54 | 7.00 |
| **Multi** ✅ | **[4, 8, 12]** | **1.69** | **4.58** |
| All | [1-12] | 3.00 | 6.91 |

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```bash
# Reduce batch size
python scripts/train_multitask.py --bs 12 ...

# Or use gradient accumulation (not implemented, but can be added via training_args)
```

**2. Dataset loading errors**
```
FileNotFoundError: Missing image: /path/to/image.png
```
→ Check that `data_root` path is correct and images exist
→ Verify `log.txt` uses correct format: `filename,text`

**3. CUDA not available**
```
Using device: cpu
```
→ Training will be very slow on CPU
→ Consider using Google Colab or a cloud GPU
→ Remove `--fp16` flag when using CPU

**4. WandB login required**
```bash
# If using --use_wandb, login first:
wandb login

# Or disable wandb:
# Remove --use_wandb flag from command
```

**5. Character not in stroke codes**
```
KeyError: 'é'
```
→ The stroke codes currently only support A-Z, a-z
→ Non-Latin or accented characters will be treated as zero vectors
→ For other scripts, create custom stroke codes in `alphabet_binary_32.csv`

---

## 🔬 Implementation Details

### Training Details

- **Optimizer**: AdamW
- **Learning rate**: 5e-5 (constant)
- **Mixed precision**: FP16 (optional, recommended)
- **Gradient accumulation**: Not used (increase `--bs` if possible)
- **Max sequence length**: 128 tokens
- **Image preprocessing**: Resize to 384×384, normalize
- **Evaluation**: Every 400 training steps
- **Checkpointing**: Save top 5 checkpoints by CER

### Inference Details

- **Decoding**: Beam search with beam width 4
- **Max generation length**: 64 tokens
- **Post-processing**: Strip special tokens
- **The auxiliary head is NOT used during inference** - only the standard language modeling head

### Data Split

- **Training**: 90% of dataset (2,400 images in paper)
- **Validation**: 10% of training data (for hyperparameter selection)
- **Test**: Held-out 600 images (official benchmark split)

---

## 📚 Related Work

This repository is part of a two-paper series:

1. **Benchmark paper** (AIED 2025): "How far are we from automatic grading of handwritten cloze form questions?"
   - Introduces the bad-handwriting benchmark
   - Evaluates off-the-shelf OCR and VLM systems

2. **Method paper** (This work): "Structural Auxiliary Supervision for OCR of Low-Legibility Handwritten Answers"
   - Proposes auxiliary structural supervision for TrOCR
   - Sets new SOTA on the benchmark

---

## 📝 Citation

If you use this code or the bad-handwriting benchmark, please cite:

```bibtex
@inproceedings{chandola2025structural,
  title={Structural Auxiliary Supervision for OCR of Low-Legibility Handwritten Answers},
  author={Chandola, Shrey and Ravikiran, Manikandan and Saluja, Rohit},
  booktitle={[To be announced]},
  year={2025},
  organization={Indian Institute of Technology, Mandi}
}
```

Also cite the benchmark paper:

```bibtex
@inproceedings{chandola2025grading,
  title={How far are we from automatic grading of handwritten cloze form questions?},
  author={Chandola, Shrey and Ravikiran, Manikandan and Saluja, Rohit},
  booktitle={International Conference on Artificial Intelligence in Education},
  pages={336--343},
  year={2025},
  organization={Springer}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Contact

For questions, issues, or dataset access requests:

- **Shrey Chandola**: s24083@students.iitmandi.ac.in
- **Manikandan Ravikiran**: erpd2301@students.iitmandi.ac.in  
- **Rohit Saluja**: rohit@iitmandi.ac.in

**Affiliation**: Indian Institute of Technology, Mandi, Himachal Pradesh, India

---

## 🙏 Acknowledgments

- Microsoft Research for the TrOCR pre-trained models
- HuggingFace for the Transformers library
- The students and educators who contributed handwriting samples to the benchmark
- IIT Mandi for computational resources

---

## 🔖 Notes

### Tips for Best Results

1. **Start with TrOCR-Small** to verify your setup works
2. **Monitor validation CER** during training - best checkpoint is usually not the last
3. **Use early stopping** - models typically converge in 30-50 epochs
4. **λ=0.05 works well** across model sizes, but you may experiment
5. **Multi-layer placement** (`struct_mode=multi`) is consistently best

### Extending to Other Languages/Scripts

To adapt this method to non-Latin scripts:

1. **Create stroke codes** for your script's characters in a CSV file
2. **Update** `alphabet_binary_32.csv` with your codes
3. **Ensure** your tokenizer handles the script properly
4. **Retrain** with your script's handwriting data

### Known Limitations

- Currently only supports Latin alphabet (A-Z, a-z)
- Single-word benchmark (not tested on phrases/sentences)
- Stroke codes are hand-designed (not learned)
- Requires token-character alignment (works well with BPE tokenizers)

---

**Last Updated**: February 2026 | **Status**: Paper under review
