import os
import argparse
import pandas as pd
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from tqdm import tqdm


def parse_ground_truth(filename: str) -> str:
    parts = filename.split("_")
    last_part = parts[-1]
    return os.path.splitext(last_part)[0]


@torch.no_grad()
def ocr_image(model, processor, src_img, device, max_length=64, num_beams=4, use_amp=False):
    pixel_values = processor(images=[src_img], return_tensors="pt").pixel_values.to(device)

    if use_amp and device.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            generated_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
    else:
        generated_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)

    if hasattr(generated_ids, "sequences"):
        generated_ids = generated_ids.sequences

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def evaluate_test_dataset(model, processor, dataset_path: str, out_csv: str, out_xlsx: str | None,
                          device: str, max_length: int, num_beams: int, fp16: bool):
    image_files = [
        f for f in os.listdir(dataset_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_files.sort()

    results = []
    use_amp = fp16 and device.startswith("cuda")

    for file_name in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(dataset_path, file_name)
        ground_truth = parse_ground_truth(file_name)

        test_image = Image.open(image_path).convert("RGB")
        predicted_text = ocr_image(
            model=model,
            processor=processor,
            src_img=test_image,
            device=device,
            max_length=max_length,
            num_beams=num_beams,
            use_amp=use_amp,
        )

        results.append({
            "Image Name": file_name,
            "Ground Truth": ground_truth,
            "Predicted Text": predicted_text,
        })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    if out_xlsx:
        df.to_excel(out_xlsx, index=False)
        print(f"Saved: {out_xlsx}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HF model path or checkpoint directory")
    ap.add_argument("--data_dir", default="test", help="Folder with test images")
    ap.add_argument("--out_csv", default="trocr_results.csv")
    ap.add_argument("--out_xlsx", default="trocr_results.xlsx")
    ap.add_argument("--device", default=None, help="cuda/cpu (auto if not set)")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--num_beams", type=int, default=4)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VisionEncoderDecoderModel.from_pretrained(args.ckpt).to(device)
    processor = TrOCRProcessor.from_pretrained(args.ckpt)
    model.eval()

    evaluate_test_dataset(
        model=model,
        processor=processor,
        dataset_path=args.data_dir,
        out_csv=args.out_csv,
        out_xlsx=args.out_xlsx,
        device=device,
        max_length=args.max_length,
        num_beams=args.num_beams,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
