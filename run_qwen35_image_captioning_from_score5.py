#!/usr/bin/env python3
"""Caption only the images whose poem/image relevance score is exactly 5.

Workflow:
1. Read /home/jiasheng/LANGUAGES/qwen35_poem_painting_relevance.csv
2. Keep only rows whose score == TARGET_SCORE
3. Write a Web-Regular-style tab-separated file with columns:
       image_id    poem
4. Run Qwen3.5 captioning only on those selected image IDs

No argparse: edit the CONFIG section below and run:
    python run_qwen35_image_captioning_from_score5.py
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------------- CONFIG ----------------
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_NAME_FOR_OUTPUT = "qwen35_2b"

RELEVANCE_CSV = Path("/home/jiasheng/LANGUAGES/qwen35_poem_painting_relevance.csv")
IMAGE_ROOT = Path("/home/jiasheng/LANGUAGES/Web")

# This file is written in the same format as Web-Regular:
#   image_id<TAB>poem
FILTERED_WEBREGULAR_OUTPUT = Path("/home/jiasheng/LANGUAGES/Web-Regular-score5-only.csv")

# Caption output written after filtering.
OUTPUT_CSV = Path("/home/jiasheng/LANGUAGES/qwen35_image_captions_score5_only.csv")

# Only keep rows with this relevance score.
TARGET_SCORE = 5

# Replace this with your exact prompt.
CAPTION_PROMPT = (
    "Write a 15-25 word English description for this painting. "
    "Do not mention the medium, nationality, or art style. "
    "Describe what is happening in the image. Avoid flowery adjectives."
)

# Useful for quick tests after filtering. Leave as None to run all selected rows.
START_AT_SELECTED_ROW = 0
MAX_IMAGES = None  # e.g. 20 for a short test run

# If False, the script resumes and skips image_ids already present in OUTPUT_CSV.
RERUN_EXISTING = False

THINKING = False
MAX_NEW_TOKENS = 128 if not THINKING else 512

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
# ----------------------------------------

processor = None
model = None

OUTPUT_FIELDS = [
    "source_row_number",
    "image_id",
    "image_path",
    "poem",
    "score",
    "caption",
    "raw_response",
    "prompt",
    "model",
    "image_lookup_note",
    "error",
]


def load_model_if_needed() -> None:
    """Load Qwen3.5 the same way as the existing scripts."""
    global processor, model
    if processor is not None and model is not None:
        return

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()


def normalize_header(name: str) -> str:
    return name.strip().lstrip("\ufeff")


def parse_score_value(score_text: str, raw_response: str) -> Optional[int]:
    """Read the numeric score from the score column; fall back to raw_response JSON if needed."""
    score_text = str(score_text).strip()
    if score_text:
        try:
            return int(float(score_text))
        except Exception:
            pass

    raw_text = str(raw_response).strip()
    if raw_text:
        # Try plain JSON first.
        try:
            data = json.loads(raw_text)
            if "score" in data:
                return int(data["score"])
        except Exception:
            pass

        # Try to extract a JSON object from surrounding text.
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if "score" in data:
                    return int(data["score"])
            except Exception:
                pass

        # Last-resort regex.
        match = re.search(r'"score"\s*:\s*(10|[1-9])', raw_text)
        if match:
            return int(match.group(1))

    return None


def read_score5_rows(relevance_csv: Path) -> List[Dict[str, str]]:
    """Read the relevance CSV and keep only rows whose score == TARGET_SCORE.

    Deduplicate by image_id, preserving first-seen order.
    """
    selected_rows: List[Dict[str, str]] = []
    seen: Set[str] = set()

    with relevance_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {relevance_csv}")

        field_map = {normalize_header(name): name for name in reader.fieldnames}
        required = ["source_row_number", "image_id", "image_path", "poem", "score", "raw_response"]
        missing = [col for col in required if col not in field_map]
        if missing:
            raise ValueError(
                f"Expected columns {required!r}; missing {missing!r}; found {reader.fieldnames}"
            )

        for row in reader:
            image_id = str(row.get(field_map["image_id"], "")).strip()
            poem = str(row.get(field_map["poem"], "")).strip()
            if not image_id or not poem:
                continue

            parsed_score = parse_score_value(
                str(row.get(field_map["score"], "")),
                str(row.get(field_map["raw_response"], "")),
            )
            if parsed_score != TARGET_SCORE:
                continue

            image_key = image_id.casefold()
            if image_key in seen:
                continue
            seen.add(image_key)

            selected_rows.append(
                {
                    "source_row_number": str(row.get(field_map["source_row_number"], "")).strip(),
                    "image_id": image_id,
                    "image_path": str(row.get(field_map["image_path"], "")).strip(),
                    "poem": poem,
                    "score": str(parsed_score),
                }
            )

    if START_AT_SELECTED_ROW > 0:
        selected_rows = selected_rows[START_AT_SELECTED_ROW:]
    if MAX_IMAGES is not None:
        selected_rows = selected_rows[:MAX_IMAGES]

    return selected_rows


def write_filtered_webregular(rows: List[Dict[str, str]], output_path: Path) -> None:
    """Write a Web-Regular-style tab-separated file with columns image_id and poem."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["image_id", "poem"])
        for row in rows:
            writer.writerow([row["image_id"], row["poem"]])


def safe_relative_path(raw_path: str) -> Optional[Path]:
    p = Path(raw_path.strip())
    if p.is_absolute() or ".." in p.parts or not p.name:
        return None
    return p


def build_image_index(image_root: Path) -> Dict[str, List[Path]]:
    """Index image files by filename and stem so img_0 can match img_0.jpg."""
    index: Dict[str, List[Path]] = {}
    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        index.setdefault(path.name.casefold(), []).append(path)
        index.setdefault(path.stem.casefold(), []).append(path)
    return index


def find_image_path(
    image_id: str,
    image_root: Path,
    image_index: Dict[str, List[Path]],
    preferred_image_path: str = "",
) -> Tuple[Optional[Path], str]:
    """Return the best image path and a note about how it was found.

    First try the image_path already stored in the relevance CSV.
    Then fall back to IMAGE_ROOT lookup by image_id.
    """
    preferred = str(preferred_image_path).strip()
    if preferred:
        p = Path(preferred)
        if p.is_file():
            return p, "reused image_path from relevance CSV"

    rel = safe_relative_path(image_id)

    if rel is not None:
        direct = image_root / rel
        if direct.is_file():
            return direct, "direct"

        if rel.suffix:
            key = rel.name.casefold()
            matches = sorted(image_index.get(key, []))
            if matches:
                return matches[0], f"indexed filename match ({len(matches)} match(es))"
        else:
            for suffix in IMAGE_EXTENSIONS:
                candidate = image_root / rel.with_suffix(suffix)
                if candidate.is_file():
                    return candidate, f"direct with suffix {suffix}"

            key = rel.name.casefold()
            matches = sorted(image_index.get(key, []))
            if matches:
                return matches[0], f"indexed stem match ({len(matches)} match(es))"

    key = Path(image_id).name.casefold()
    matches = sorted(image_index.get(key, []))
    if matches:
        return matches[0], f"fallback indexed match ({len(matches)} match(es))"

    return None, "not found"


def query_qwen35(image_path: Path) -> str:
    load_model_if_needed()
    assert processor is not None and model is not None

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=THINKING,
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def clean_caption(response: str) -> str:
    """Keep a single clean caption string while preserving the raw response separately."""
    text = response.strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()

    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return text


def load_completed_image_ids(output_csv: Path) -> Set[str]:
    if RERUN_EXISTING or not output_csv.exists():
        return set()

    completed: Set[str] = set()
    with output_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            if image_id:
                completed.add(image_id.casefold())
    return completed


def ensure_output_header(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() and output_csv.stat().st_size > 0:
        return

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()


def append_result(output_csv: Path, result: Dict[str, str]) -> None:
    with output_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writerow(result)
        f.flush()


def main() -> None:
    print(f"Relevance CSV: {RELEVANCE_CSV}")
    print(f"Image root: {IMAGE_ROOT}")
    print(f"Filtered Web-Regular output: {FILTERED_WEBREGULAR_OUTPUT}")
    print(f"Caption output: {OUTPUT_CSV}")
    print(f"Target score: {TARGET_SCORE}")

    selected_rows = read_score5_rows(RELEVANCE_CSV)
    write_filtered_webregular(selected_rows, FILTERED_WEBREGULAR_OUTPUT)

    print(f"Selected rows with score == {TARGET_SCORE}: {len(selected_rows)}")
    print(f"Wrote filtered pair file: {FILTERED_WEBREGULAR_OUTPUT}")

    completed = load_completed_image_ids(OUTPUT_CSV)
    ensure_output_header(OUTPUT_CSV)

    print(f"Already completed in caption output: {len(completed)}")
    print("Indexing images...")
    image_index = build_image_index(IMAGE_ROOT)
    print(f"Indexed {sum(1 for paths in image_index.values() for _ in paths)} image index entries.")

    total_to_run = sum(1 for row in selected_rows if row["image_id"].casefold() not in completed)
    print(f"Images left to caption: {total_to_run}")

    for row in selected_rows:
        image_id = row["image_id"]
        poem = row["poem"]
        score = row["score"]

        if image_id.casefold() in completed:
            continue

        image_path, lookup_note = find_image_path(
            image_id=image_id,
            image_root=IMAGE_ROOT,
            image_index=image_index,
            preferred_image_path=row.get("image_path", ""),
        )

        print("\n" + "-" * 80)
        print(f"Source row: {row['source_row_number']}")
        print(f"Image ID: {image_id}")
        print(f"Score: {score}")
        print(f"Lookup: {lookup_note}")

        base_result = {
            "source_row_number": row["source_row_number"],
            "image_id": image_id,
            "image_path": str(image_path) if image_path is not None else "",
            "poem": poem,
            "score": score,
            "caption": "",
            "raw_response": "",
            "prompt": CAPTION_PROMPT,
            "model": MODEL_NAME_FOR_OUTPUT,
            "image_lookup_note": lookup_note,
            "error": "",
        }

        if image_path is None:
            base_result["error"] = f"Image not found under {IMAGE_ROOT}"
            append_result(OUTPUT_CSV, base_result)
            print(base_result["error"])
            continue

        try:
            print(f"Image path: {image_path}")
            response = query_qwen35(image_path)
            caption = clean_caption(response)

            base_result["raw_response"] = response
            base_result["caption"] = caption
            append_result(OUTPUT_CSV, base_result)

            print(f"Raw response: {response}")
            print(f"Caption: {caption}")

        except Exception as e:
            base_result["error"] = repr(e)
            append_result(OUTPUT_CSV, base_result)
            print(f"ERROR: {e!r}")

    print("\nDone.")
    print(f"Filtered pair file written to: {FILTERED_WEBREGULAR_OUTPUT}")
    print(f"Caption results written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
