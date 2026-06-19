#!/usr/bin/env python3
"""Score poem/painting relevance with Qwen3.5, one pair at a time.

No argparse: edit the CONFIG section below and run:
    python run_qwen35_poem_painting_relevance.py
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------------- CONFIG ----------------
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_NAME_FOR_OUTPUT = "qwen35_2b"

CSV_PATH = Path("/home/jiasheng/LANGUAGES/Web-Regular.csv")
IMAGE_ROOT = Path("/home/jiasheng/LANGUAGES/Web")
OUTPUT_CSV = Path("/home/jiasheng/LANGUAGES/qwen35_poem_painting_relevance.csv")

# Your uploaded CSV uses these columns.
IMAGE_ID_COLUMN = "image_id"
POEM_COLUMN = "poem"

# Useful for quick tests. Leave as None to run the full CSV.
START_AT_ROW = 0
MAX_ROWS = None  # e.g. 20 for a short test run

# If False, the script resumes and skips image_ids already present in OUTPUT_CSV.
RERUN_EXISTING = False

THINKING = False
MAX_NEW_TOKENS = 160 if not THINKING else 512

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
# ----------------------------------------

processor = None
model = None


def load_model_if_needed() -> None:
    """Load Qwen3.5 the same way as the existing clutter script."""
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


def detect_delimiter(csv_path: Path) -> str:
    """The Web-Regular.csv file is tab-separated, but this keeps it flexible."""
    first_line = csv_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
    if "\t" in first_line:
        return "\t"
    return ","


def normalize_header(name: str) -> str:
    return name.strip().lstrip("\ufeff")


def read_pairs(csv_path: Path) -> List[Dict[str, str]]:
    delimiter = detect_delimiter(csv_path)
    rows: List[Dict[str, str]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {csv_path}")

        field_map = {normalize_header(name): name for name in reader.fieldnames}
        if IMAGE_ID_COLUMN not in field_map or POEM_COLUMN not in field_map:
            raise ValueError(
                f"Expected columns {IMAGE_ID_COLUMN!r} and {POEM_COLUMN!r}; "
                f"found {reader.fieldnames}"
            )

        image_col = field_map[IMAGE_ID_COLUMN]
        poem_col = field_map[POEM_COLUMN]

        for row_number, row in enumerate(reader):
            if row_number < START_AT_ROW:
                continue
            if MAX_ROWS is not None and len(rows) >= MAX_ROWS:
                break

            image_id = str(row.get(image_col, "")).strip()
            poem = str(row.get(poem_col, "")).strip()
            if not image_id or not poem:
                continue

            rows.append(
                {
                    "source_row_number": str(row_number + 2),  # +2 because header is row 1
                    "image_id": image_id,
                    "poem": poem,
                }
            )

    return rows


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


def find_image_path(image_id: str, image_root: Path, image_index: Dict[str, List[Path]]) -> Tuple[Optional[Path], str]:
    """Return the best image path and a note about how it was found."""
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


def build_prompt(poem: str) -> str:
    return f"""You are evaluating how semantically relevant a poem is to a painting/image.

Look carefully at the image and read the poem. Score how relevant the poem is to the image on a 1-10 scale.

Scale:
1 = completely unrelated.
3 = weakly related.
5 = somewhat related in theme, mood, object, setting, or symbolism.

Return only valid JSON in this exact format:
{{"score": <integer from 1 to 10>, "reason": "<one concise sentence>"}}

Poem:
{poem}"""

# 7 = clearly related.
# 10 = highly relevant; the poem strongly matches the image content, scene, mood, and symbolism.


def query_qwen35(poem: str, image_path: Path) -> str:
    load_model_if_needed()
    assert processor is not None and model is not None

    prompt = build_prompt(poem)
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
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


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_score_and_reason(response: str) -> Tuple[Optional[int], str]:
    """Parse JSON first, then fall back to a conservative regex."""
    cleaned = strip_code_fences(response)

    json_candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        json_candidates.append(match.group(0))

    for candidate in json_candidates:
        try:
            data = json.loads(candidate)
            score = int(data.get("score"))
            reason = str(data.get("reason", "")).strip()
            if 1 <= score <= 10:
                return score, reason
        except Exception:
            pass

    score_match = re.search(r"(?i)\bscore\b\D*(10|[1-9])\b", cleaned)
    if score_match:
        return int(score_match.group(1)), ""

    return None, ""


def load_completed_image_ids(output_csv: Path) -> Set[str]:
    if RERUN_EXISTING or not output_csv.exists():
        return set()

    completed: Set[str] = set()
    with output_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            if image_id:
                completed.add(image_id)
    return completed


def ensure_output_header(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() and output_csv.stat().st_size > 0:
        return

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()


OUTPUT_FIELDS = [
    "source_row_number",
    "image_id",
    "image_path",
    "poem",
    "score",
    "reason",
    "raw_response",
    "model",
    "image_lookup_note",
    "error",
]


def append_result(output_csv: Path, result: Dict[str, str]) -> None:
    with output_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writerow(result)
        f.flush()


def main() -> None:
    print(f"CSV: {CSV_PATH}")
    print(f"Image root: {IMAGE_ROOT}")
    print(f"Output: {OUTPUT_CSV}")

    pairs = read_pairs(CSV_PATH)
    completed = load_completed_image_ids(OUTPUT_CSV)
    ensure_output_header(OUTPUT_CSV)

    print(f"Loaded {len(pairs)} poem/image rows from CSV.")
    print(f"Already completed in output: {len(completed)}")

    print("Indexing images...")
    image_index = build_image_index(IMAGE_ROOT)
    print(f"Indexed {sum(1 for paths in image_index.values() for _ in paths)} image index entries.")

    total_to_run = sum(1 for pair in pairs if pair["image_id"] not in completed)
    print(f"Rows left to score: {total_to_run}")

    for current_number, pair in enumerate(pairs, start=1):
        image_id = pair["image_id"]
        poem = pair["poem"]

        if image_id in completed:
            continue

        image_path, lookup_note = find_image_path(image_id, IMAGE_ROOT, image_index)
        print("\n" + "-" * 80)
        print(f"CSV row: {pair['source_row_number']}")
        print(f"Image ID: {image_id}")
        print(f"Lookup: {lookup_note}")

        base_result = {
            "source_row_number": pair["source_row_number"],
            "image_id": image_id,
            "image_path": str(image_path) if image_path is not None else "",
            "poem": poem,
            "score": "",
            "reason": "",
            "raw_response": "",
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
            print(f"Poem: {poem}")
            response = query_qwen35(poem, image_path)
            score, reason = parse_score_and_reason(response)

            base_result["raw_response"] = response
            base_result["score"] = "" if score is None else str(score)
            base_result["reason"] = reason
            if score is None:
                base_result["error"] = "Could not parse a 1-10 score from model response"

            append_result(OUTPUT_CSV, base_result)
            print(f"Raw response: {response}")
            print(f"Parsed score: {base_result['score']}")
            if reason:
                print(f"Reason: {reason}")

        except Exception as e:
            base_result["error"] = repr(e)
            append_result(OUTPUT_CSV, base_result)
            print(f"ERROR: {e!r}")

    print("\nDone.")
    print(f"Results written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
