#!/usr/bin/env python3
# train_pretrained_clip_chinese_poem.py
#
# Fine-tune an already pretrained CLIP-style model on the same Chinese poem/image pairs.
# Default is Chinese-CLIP Base because your text column is Chinese poetry.
# For the English OpenCLIP checkpoints from Gandelsman et al.'s TextSpan repo, see CLIP_MODEL_NAME alternatives below.

import os
import math
import json
import time
import shutil
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

# ============================================================
# Paths / data columns: kept from your original script
# ============================================================

CSV_PATH = "/home/jiasheng/LANGUAGES/qwen35_image_captions_score5_only.csv"
WEB_DIR = "/home/jiasheng/LANGUAGES/Web"
OUTPUT_DIR = "/home/jiasheng/LANGUAGES/chinese_clip_pretrained_finetune_59k"

IMAGE_PATH_COLUMN = "image_path"
IMAGE_ID_COLUMN = "image_id"
TEXT_COLUMN = "poem"

TOTAL_PAIRS_TO_USE = 59_000
VAL_SIZE = 1_000
TEST_SIZE = 1_000

# ============================================================
# Model choice
# ============================================================

# Recommended for this dataset because poems are Chinese:
CLIP_MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"   # ~188M params, ViT-B/16 + Chinese RoBERTa

# Larger Chinese option if you have enough GPU memory and want to try more capacity:
# CLIP_MODEL_NAME = "OFA-Sys/chinese-clip-vit-large-patch14" # ~406M params

# English OpenCLIP-style options matching the TextSpan paper/repo family.
# These are less appropriate for Chinese poems unless you translate poems to English first.
# CLIP_MODEL_NAME = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K" # ~0.15B params
# CLIP_MODEL_NAME = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K" # ~0.43B params
# CLIP_MODEL_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" # ~0.99B params

# Use full fine-tuning by default. For very small GPUs or overfitting, set one/both True.
FREEZE_VISION_ENCODER = False
FREEZE_TEXT_ENCODER = False

# For 22k pairs, keep LR small: this is pretrained CLIP, not new encoders + new heads.
BACKBONE_LR = 2e-6
PROJECTION_LR = 1e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0

INITIAL_TEMPERATURE = 0.07

EPOCHS = 10
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1

MAX_TEXT_LENGTH = 64        # Chinese-CLIP can handle this; OpenAI/OpenCLIP may effectively cap at 77.
NUM_WORKERS = 4
SEED = 42

USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True
LOG_EVERY_OPT_STEPS = 25

# ============================================================
# Utilities
# ============================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_path(row: pd.Series) -> str | None:
    candidates: list[str] = []

    raw_path = row.get(IMAGE_PATH_COLUMN, "")
    if isinstance(raw_path, str) and raw_path.strip():
        raw_path = raw_path.strip()
        candidates.append(raw_path)
        candidates.append(os.path.join(WEB_DIR, os.path.basename(raw_path)))

    image_id = row.get(IMAGE_ID_COLUMN, "")
    if isinstance(image_id, str) and image_id.strip():
        image_id = image_id.strip()
        for ext in [".jpeg", ".jpg", ".png", ".webp"]:
            candidates.append(os.path.join(WEB_DIR, image_id + ext))

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path and os.path.isfile(path):
            return path

    return None


def prepare_dataframe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    if len(df) < TOTAL_PAIRS_TO_USE:
        raise RuntimeError(
            f"CSV only has {len(df)} rows, but TOTAL_PAIRS_TO_USE={TOTAL_PAIRS_TO_USE}."
        )

    df = df.iloc[:TOTAL_PAIRS_TO_USE].copy()

    if TEXT_COLUMN not in df.columns:
        raise RuntimeError(f"Missing text column: {TEXT_COLUMN}")

    print("Resolving image paths...")
    df["resolved_image_path"] = df.apply(resolve_image_path, axis=1)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()

    before = len(df)
    df = df[(df["resolved_image_path"].notna()) & (df[TEXT_COLUMN] != "")].copy()
    after = len(df)

    if after != before:
        raise RuntimeError(
            f"After filtering missing images/text, only {after}/{before} rows remain. "
            f"Since you asked to use the first {TOTAL_PAIRS_TO_USE} pairs exactly, fix those rows or "
            f"lower TOTAL_PAIRS_TO_USE."
        )

    train_size = TOTAL_PAIRS_TO_USE - VAL_SIZE - TEST_SIZE
    if train_size <= 0:
        raise RuntimeError("VAL_SIZE + TEST_SIZE must be smaller than TOTAL_PAIRS_TO_USE.")

    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size : train_size + VAL_SIZE].reset_index(drop=True)
    test_df = df.iloc[train_size + VAL_SIZE : train_size + VAL_SIZE + TEST_SIZE].reset_index(drop=True)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    test_df.to_csv(out_dir / "test_split.csv", index=False)

    print(f"Train pairs: {len(train_df)}")
    print(f"Val pairs:   {len(val_df)}")
    print(f"Test pairs:  {len(test_df)}")
    return train_df, val_df, test_df


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def maybe_enable_gradient_checkpointing(module: nn.Module) -> None:
    if not USE_GRADIENT_CHECKPOINTING:
        return
    if hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable()
    if hasattr(module, "config") and hasattr(module.config, "use_cache"):
        module.config.use_cache = False


def get_amp_dtype(device: torch.device):
    if device.type != "cuda":
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ============================================================
# Dataset and collator
# ============================================================

class PairedImageTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.image_paths = df["resolved_image_path"].tolist()
        self.texts = df[TEXT_COLUMN].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        return {
            "image_path": self.image_paths[idx],
            "text": self.texts[idx],
        }


class CLIPCollator:
    def __init__(self, processor, max_text_length: int):
        self.processor = processor
        self.max_text_length = max_text_length

    def __call__(self, batch):
        image_paths = [x["image_path"] for x in batch]
        texts = [x["text"] for x in batch]

        images = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
            except Exception as exc:
                raise RuntimeError(f"Failed to open image: {path}") from exc

        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        # Debug metadata; these are small lists of strings for the current batch only.
        inputs["image_paths"] = image_paths
        inputs["texts"] = texts
        return inputs

# ============================================================
# Model wrapper
# ============================================================

class PretrainedCLIPFineTuner(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip = clip_model

    @staticmethod
    def _pool_model_output(outputs: Any) -> torch.Tensor:
        """
        Some Chinese-CLIP checkpoints expose get_*_features(), but in some
        transformers versions those methods may return BaseModelOutputWithPooling
        instead of the final projected tensor. This converts either case to a tensor.
        """
        if torch.is_tensor(outputs):
            return outputs

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state[:, 0]

        if isinstance(outputs, (tuple, list)):
            if len(outputs) > 1 and torch.is_tensor(outputs[1]):
                return outputs[1]
            if len(outputs) > 0 and torch.is_tensor(outputs[0]):
                first = outputs[0]
                if first.ndim == 3:
                    return first[:, 0]
                return first

        raise TypeError(f"Could not convert model output of type {type(outputs)} to a tensor")

    @staticmethod
    def _maybe_project(features: torch.Tensor, projection: nn.Module | None) -> torch.Tensor:
        if projection is None:
            return features

        # Apply projection only when the feature dimension matches the projection input.
        # If get_*_features() already returned projected features, do not project again.
        in_features = getattr(projection, "in_features", None)
        if in_features is None or features.shape[-1] == in_features:
            return projection(features)
        return features

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Prefer the public helper, but be robust to checkpoints/transformers versions
        # where it returns a BaseModelOutputWithPooling instead of a tensor.
        outputs = self.clip.get_image_features(pixel_values=pixel_values)
        image_features = self._pool_model_output(outputs)

        image_projection = getattr(self.clip, "visual_projection", None)
        if image_projection is None:
            image_projection = getattr(self.clip, "vision_projection", None)
        image_features = self._maybe_project(image_features, image_projection)

        return F.normalize(image_features, p=2, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            text_kwargs["token_type_ids"] = token_type_ids

        outputs = self.clip.get_text_features(**text_kwargs)
        text_features = self._pool_model_output(outputs)

        text_projection = getattr(self.clip, "text_projection", None)
        text_features = self._maybe_project(text_features, text_projection)

        return F.normalize(text_features, p=2, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **_: Any,
    ):
        image_embeds = self.encode_image(pixel_values=pixel_values)
        text_embeds = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return image_embeds, text_embeds

    @property
    def logit_scale(self) -> torch.nn.Parameter:
        return self.clip.logit_scale


def freeze_named_module_params(model: nn.Module, module_name_fragments: list[str]) -> None:
    for name, param in model.named_parameters():
        if any(fragment in name for fragment in module_name_fragments):
            param.requires_grad = False


def apply_freezing(model: PretrainedCLIPFineTuner) -> None:
    if FREEZE_VISION_ENCODER:
        freeze_named_module_params(model, ["vision_model"])
        print("Frozen vision encoder parameters.")
    if FREEZE_TEXT_ENCODER:
        freeze_named_module_params(model, ["text_model"])
        print("Frozen text encoder parameters.")


def contrastive_loss(model: PretrainedCLIPFineTuner, image_embeds: torch.Tensor, text_embeds: torch.Tensor):
    # Same CLIP symmetric image<->text contrastive objective, using pretrained logit_scale.
    logit_scale = model.logit_scale.exp().clamp(max=100.0)
    logits_per_image = logit_scale * image_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()

    labels = torch.arange(image_embeds.size(0), device=image_embeds.device)

    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    loss = 0.5 * (loss_i2t + loss_t2i)

    return loss, logits_per_image

# ============================================================
# Optimizer
# ============================================================


def build_optimizer(model: PretrainedCLIPFineTuner):
    decay_params = []
    no_decay_params = []
    proj_decay_params = []
    proj_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lname = name.lower()
        is_no_decay = (
            lname.endswith(".bias")
            or "layernorm" in lname
            or "layer_norm" in lname
            or ".norm" in lname
            or "logit_scale" in lname
        )
        is_projection_or_scale = (
            "visual_projection" in lname
            or "text_projection" in lname
            or "logit_scale" in lname
        )

        if is_projection_or_scale:
            if is_no_decay:
                proj_no_decay_params.append(param)
            else:
                proj_decay_params.append(param)
        else:
            if is_no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    param_groups = [
        {"params": decay_params, "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "lr": BACKBONE_LR, "weight_decay": 0.0},
        {"params": proj_decay_params, "lr": PROJECTION_LR, "weight_decay": WEIGHT_DECAY},
        {"params": proj_no_decay_params, "lr": PROJECTION_LR, "weight_decay": 0.0},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)

# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def collect_embeddings(model: PretrainedCLIPFineTuner, loader: DataLoader, device: torch.device):
    model.eval()

    all_image_embeds = []
    all_text_embeds = []

    for batch in tqdm(loader, desc="Encoding", leave=False):
        batch.pop("image_paths", None)
        batch.pop("texts", None)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}

        image_embeds, text_embeds = model(**batch)

        all_image_embeds.append(image_embeds.float().cpu())
        all_text_embeds.append(text_embeds.float().cpu())

    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)

    return image_embeds, text_embeds


def retrieval_metrics_from_logits(logits: torch.Tensor, prefix: str):
    n = logits.size(0)
    labels = torch.arange(n)

    sorted_indices = torch.argsort(logits, dim=1, descending=True)
    ranks = (sorted_indices == labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    ranks = ranks.float()

    metrics = {
        f"{prefix}_R@1": (ranks <= 1).float().mean().item(),
        f"{prefix}_R@10": (ranks <= 10).float().mean().item(),
        f"{prefix}_R@50": (ranks <= 50).float().mean().item(),
        f"{prefix}_R@100": (ranks <= 100).float().mean().item(),
        f"{prefix}_R@250": (ranks <= 250).float().mean().item(),
        f"{prefix}_R@500": (ranks <= 500).float().mean().item(),
        f"{prefix}_median_rank": ranks.median().item(),
        f"{prefix}_mean_rank": ranks.mean().item(),
    }

    return metrics


@torch.no_grad()
def evaluate(model: PretrainedCLIPFineTuner, loader: DataLoader, device: torch.device, split_name: str):
    image_embeds, text_embeds = collect_embeddings(model, loader, device)

    image_embeds = image_embeds.to(device)
    text_embeds = text_embeds.to(device)

    logit_scale = model.logit_scale.exp().clamp(max=100.0).detach()
    logits = logit_scale * image_embeds @ text_embeds.t()

    labels = torch.arange(logits.size(0), device=device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_i2t + loss_t2i)

    logits_cpu = logits.cpu()

    metrics = {"split": split_name, "loss": loss.item()}
    metrics.update(retrieval_metrics_from_logits(logits_cpu, "i2t"))
    metrics.update(retrieval_metrics_from_logits(logits_cpu.t(), "t2i"))

    metrics["mean_R@1"] = 0.5 * (metrics["i2t_R@1"] + metrics["t2i_R@1"])
    metrics["mean_R@10"] = 0.5 * (metrics["i2t_R@10"] + metrics["t2i_R@10"])
    metrics["mean_R@50"] = 0.5 * (metrics["i2t_R@50"] + metrics["t2i_R@50"])
    metrics["mean_R@100"] = 0.5 * (metrics["i2t_R@100"] + metrics["t2i_R@100"])
    metrics["mean_R@250"] = 0.5 * (metrics["i2t_R@250"] + metrics["t2i_R@250"])
    metrics["mean_R@500"] = 0.5 * (metrics["i2t_R@500"] + metrics["t2i_R@500"])

    return metrics


def print_metrics(metrics: dict):
    print(
        f"[{metrics['split']}] "
        f"loss={metrics['loss']:.4f} | "
        f"I2T R@1={metrics['i2t_R@1']:.4f} "
        f"R@10={metrics['i2t_R@10']:.4f} "
        f"R@50={metrics['i2t_R@50']:.4f} "
        f"R@100={metrics['i2t_R@100']:.4f} "
        f"R@250={metrics['i2t_R@250']:.4f} "
        f"R@500={metrics['i2t_R@500']:.4f} | "
        f"T2I R@1={metrics['t2i_R@1']:.4f} "
        f"R@10={metrics['t2i_R@10']:.4f} "
        f"R@50={metrics['t2i_R@50']:.4f} "
        f"R@100={metrics['t2i_R@100']:.4f} "
        f"R@250={metrics['t2i_R@250']:.4f} "
        f"R@500={metrics['t2i_R@500']:.4f} | "
        f"mean_R@1={metrics['mean_R@1']:.4f} "
        f"mean_R@10={metrics['mean_R@10']:.4f} "
        f"mean_R@50={metrics['mean_R@50']:.4f} "
        f"mean_R@100={metrics['mean_R@100']:.4f} "
        f"mean_R@250={metrics['mean_R@250']:.4f} "
        f"mean_R@500={metrics['mean_R@500']:.4f}"
    )


@torch.no_grad()
def inspect_bad_predictions_both(
    model,
    loader,
    df,
    device,
    split_name="test",
    top_k=10,
    max_examples=20,
):
    image_embeds, text_embeds = collect_embeddings(model, loader, device)

    image_embeds = image_embeds.to(device)
    text_embeds = text_embeds.to(device)

    logit_scale = model.logit_scale.exp().clamp(max=100.0).detach()
    logits = logit_scale * image_embeds @ text_embeds.t()

    return {
        "i2t": inspect_direction_from_logits(
            logits=logits,
            df=df,
            direction="i2t",
            split_name=split_name,
            top_k=top_k,
            max_examples=max_examples,
        ),
        "t2i": inspect_direction_from_logits(
            logits=logits.t(),
            df=df,
            direction="t2i",
            split_name=split_name,
            top_k=top_k,
            max_examples=max_examples,
        ),
    }


def inspect_direction_from_logits(logits, df, direction, split_name, top_k, max_examples):
    n = logits.size(0)
    labels = torch.arange(n, device=logits.device)

    sorted_indices = torch.argsort(logits, dim=1, descending=True)
    ranks = (sorted_indices == labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    bad = torch.where(ranks > 1)[0]

    results = []

    for idx in bad[:max_examples]:
        i = idx.item()
        gt_rank = ranks[i].item()
        gt_score = logits[i, i].item()
        top_indices = sorted_indices[i, :top_k].detach().cpu().tolist()

        if direction == "i2t":
            item = {
                "split": split_name,
                "direction": "i2t",
                "query_image_index": i,
                "query_image_path": df.iloc[i]["resolved_image_path"],
                "groundtruth_text": df.iloc[i][TEXT_COLUMN],
                "groundtruth_rank": int(gt_rank),
                "groundtruth_score": float(gt_score),
                "top_text_predictions": [],
            }

            for rank, text_idx in enumerate(top_indices, start=1):
                item["top_text_predictions"].append(
                    {
                        "rank": rank,
                        "text_index": int(text_idx),
                        "score": float(logits[i, text_idx].item()),
                        "is_groundtruth": text_idx == i,
                        "text": df.iloc[text_idx][TEXT_COLUMN],
                        "matched_image_path_for_that_text": df.iloc[text_idx]["resolved_image_path"],
                    }
                )

        elif direction == "t2i":
            item = {
                "split": split_name,
                "direction": "t2i",
                "query_text_index": i,
                "query_text": df.iloc[i][TEXT_COLUMN],
                "groundtruth_image_path": df.iloc[i]["resolved_image_path"],
                "groundtruth_rank": int(gt_rank),
                "groundtruth_score": float(gt_score),
                "top_image_predictions": [],
            }

            for rank, image_idx in enumerate(top_indices, start=1):
                item["top_image_predictions"].append(
                    {
                        "rank": rank,
                        "image_index": int(image_idx),
                        "score": float(logits[i, image_idx].item()),
                        "is_groundtruth": image_idx == i,
                        "image_path": df.iloc[image_idx]["resolved_image_path"],
                        "matched_text_for_that_image": df.iloc[image_idx][TEXT_COLUMN],
                    }
                )
        else:
            raise ValueError(f"Unknown direction: {direction}")

        results.append(item)

    return results

# ============================================================
# Checkpointing
# ============================================================


def save_checkpoint(
    model: PretrainedCLIPFineTuner,
    processor,
    path: Path,
    epoch: int,
    global_step: int,
    metrics: dict,
):
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")

    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    tmp_path.mkdir(parents=True, exist_ok=True)

    # Save as a normal Hugging Face model for later inference/fine-tuning.
    model.clip.save_pretrained(str(tmp_path / "hf_model"))
    processor.save_pretrained(str(tmp_path / "processor"))

    # Also save training metadata and a wrapper-compatible state_dict.
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "clip_model_name": CLIP_MODEL_NAME,
            "metrics": metrics,
            "config": {
                "csv_path": CSV_PATH,
                "web_dir": WEB_DIR,
                "total_pairs_to_use": TOTAL_PAIRS_TO_USE,
                "val_size": VAL_SIZE,
                "test_size": TEST_SIZE,
                "text_column": TEXT_COLUMN,
                "max_text_length": MAX_TEXT_LENGTH,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "backbone_lr": BACKBONE_LR,
                "projection_lr": PROJECTION_LR,
                "freeze_vision_encoder": FREEZE_VISION_ENCODER,
                "freeze_text_encoder": FREEZE_TEXT_ENCODER,
            },
        },
        tmp_path / "training_state.pt",
    )

    with open(tmp_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if path.exists():
        shutil.rmtree(path)

    shutil.move(str(tmp_path), str(path))

# ============================================================
# Train
# ============================================================


def debug_bad_retrieval(
    logits: torch.Tensor,
    query_items: list[str],
    target_items: list[str],
    direction_name: str,
    query_label: str,
    target_label: str,
    epoch: int,
    step: int,
    top_k: int = 10,
):
    labels = torch.arange(logits.size(0), device=logits.device)
    sorted_indices = torch.argsort(logits, dim=1, descending=True)
    ranks = (sorted_indices == labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1

    bad = torch.where(ranks > 1)[0]
    if len(bad) == 0:
        return

    worst_pos = torch.argmax(ranks[bad])
    i = bad[worst_pos].item()

    gt_rank = ranks[i].item()
    top = sorted_indices[i, :top_k].detach().cpu().tolist()

    print("\n" + "=" * 80)
    print(f"Bad {direction_name} train example, epoch={epoch}, step={step}")
    print(f"{query_label} row in batch: {i}")
    print(f"Ground-truth rank within batch: {gt_rank}")

    print(f"\nQuery {query_label}:")
    print(query_items[i])

    print(f"\nGround-truth {target_label}:")
    print(target_items[i])

    print(f"\nTop {target_label} predictions in this batch:")
    for rank, j in enumerate(top, start=1):
        score = logits[i, j].item()
        marker = " <-- ground truth" if j == i else ""
        print(f"\nRank {rank} | batch {target_label} row {j} | score={score:.4f}{marker}")
        print(target_items[j])

    print("=" * 80 + "\n")


def train_one_epoch(
    model: PretrainedCLIPFineTuner,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    amp_dtype,
    epoch: int,
    global_step: int,
):
    model.train()

    running_loss = 0.0
    running_count = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)

    for step, batch in enumerate(pbar):
        metadata = {
            "image_paths": batch.pop("image_paths"),
            "texts": batch.pop("texts"),
        }
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda"), dtype=amp_dtype):
            image_embeds, text_embeds = model(**batch)
            loss, logits_per_image = contrastive_loss(model, image_embeds, text_embeds)
            loss_for_backward = loss / GRAD_ACCUM_STEPS

        # Debug bad within-batch predictions occasionally.
        if step % 100 == 0:
            with torch.no_grad():
                debug_bad_retrieval(
                    logits=logits_per_image,
                    query_items=metadata["image_paths"],
                    target_items=metadata["texts"],
                    direction_name="I2T image-to-text",
                    query_label="image",
                    target_label="text",
                    epoch=epoch,
                    step=step,
                    top_k=10,
                )
                debug_bad_retrieval(
                    logits=logits_per_image.t(),
                    query_items=metadata["texts"],
                    target_items=metadata["image_paths"],
                    direction_name="T2I text-to-image",
                    query_label="text",
                    target_label="image",
                    epoch=epoch,
                    step=step,
                    top_k=10,
                )

        scaler.scale(loss_for_backward).backward()

        batch_loss = loss.item()
        running_loss += batch_loss
        running_count += 1

        is_update_step = ((step + 1) % GRAD_ACCUM_STEPS == 0) or ((step + 1) == len(loader))

        if is_update_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100.0))

            global_step += 1

            if global_step % LOG_EVERY_OPT_STEPS == 0:
                avg_loss = running_loss / max(1, running_count)
                temp = 1.0 / model.logit_scale.exp().item()
                lr0 = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "temp": f"{temp:.4f}", "lr": f"{lr0:.2e}"})

    avg_epoch_loss = running_loss / max(1, running_count)
    return global_step, avg_epoch_loss


def main():
    seed_everything(SEED)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(device)

    print(f"Device: {device}")
    print(f"AMP dtype: {amp_dtype if device.type == 'cuda' else 'disabled'}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = prepare_dataframe()

    print(f"Loading processor: {CLIP_MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME, use_fast=True)

    print(f"Loading pretrained CLIP model: {CLIP_MODEL_NAME}")
    clip_model = AutoModel.from_pretrained(CLIP_MODEL_NAME)
    maybe_enable_gradient_checkpointing(clip_model)

    model = PretrainedCLIPFineTuner(clip_model)
    apply_freezing(model)
    model.to(device)

    print(f"Total parameters:     {count_all_params(model):,}")
    print(f"Trainable parameters: {count_trainable_params(model):,}")

    collator = CLIPCollator(processor=processor, max_text_length=MAX_TEXT_LENGTH)

    train_loader = DataLoader(
        PairedImageTextDataset(train_df),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collator,
        drop_last=False,
    )

    val_loader = DataLoader(
        PairedImageTextDataset(val_df),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collator,
        drop_last=False,
    )

    test_loader = DataLoader(
        PairedImageTextDataset(test_df),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collator,
        drop_last=False,
    )

    optimizer = build_optimizer(model)

    updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_training_steps = updates_per_epoch * EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    scaler_enabled = USE_AMP and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    print(f"Optimizer update steps per epoch: {updates_per_epoch}")
    print(f"Total optimizer update steps:     {total_training_steps}")
    print(f"Warmup steps:                     {warmup_steps}")
    print(f"Grad accumulation steps:          {GRAD_ACCUM_STEPS}")
    print(f"Effective train batch size:       {TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS}")

    best_val_score = -1.0
    global_step = 0

    log_path = out_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        global_step, train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_dtype=amp_dtype,
            epoch=epoch,
            global_step=global_step,
        )

        print(f"Epoch {epoch} train loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader, device, split_name="val")
        print_metrics(val_metrics)

        log_obj = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val": val_metrics,
        }
        append_jsonl(log_path, log_obj)

        save_checkpoint(
            model=model,
            processor=processor,
            path=out_dir / f"epoch_{epoch:03d}",
            epoch=epoch,
            global_step=global_step,
            metrics=val_metrics,
        )

        val_score = val_metrics["mean_R@1"]
        if val_score > best_val_score:
            best_val_score = val_score
            print(f"New best val mean_R@1: {best_val_score:.4f}. Saving best checkpoint.")
            save_checkpoint(
                model=model,
                processor=processor,
                path=out_dir / "best",
                epoch=epoch,
                global_step=global_step,
                metrics=val_metrics,
            )

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed / 3600:.2f} hours.")

    print("Loading best checkpoint for test evaluation...")
    best_ckpt_path = out_dir / "best" / "training_state.pt"
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    test_metrics = evaluate(model, test_loader, device, split_name="test")
    print_metrics(test_metrics)

    bad_examples = inspect_bad_predictions_both(
        model=model,
        loader=test_loader,
        df=test_df,
        device=device,
        split_name="test",
        top_k=10,
        max_examples=50,
    )

    with open(out_dir / "test_bad_predictions.json", "w", encoding="utf-8") as f:
        json.dump(bad_examples, f, ensure_ascii=False, indent=2)

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    append_jsonl(
        log_path,
        {
            "final_test": test_metrics,
            "best_checkpoint": str(best_ckpt_path),
        },
    )

    print(f"Best checkpoint: {out_dir / 'best'}")
    print(f"Bad predictions: {out_dir / 'test_bad_predictions.json'}")
    print(f"Test metrics:    {out_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
