import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# TF 只负责读取 RLDS，不使用 H800
try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

import tensorflow_datasets as tfds
import numpy as np
import torch

from PIL import Image
from torchvision import transforms as T
from laq_model import LatentActionQuantization


DATASET_DIR = (
    "/mnt/data/RLDS/EgoVerse_full/"
    "aria_bimanual_front_1/"
    "ego_verse_infidata/1.0.0"
)

CHECKPOINT = (
    "/mnt/data/laixianxin/LAPA/"
    "lapa_checkpoints/vae.100000.pt"
)

OFFSET = 30
DEVICE = "cuda:0"


def scalar(x):
    """Tensor / ndarray / bytes -> 普通 Python 值"""
    if hasattr(x, "numpy"):
        x = x.numpy()

    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()

    if isinstance(x, np.bytes_):
        x = bytes(x)

    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")

    return x


print("=" * 70)
print("1. Load RLDS")
print("dataset:", DATASET_DIR)

builder = tfds.builder_from_directory(DATASET_DIR)

print("name:", builder.info.name)
print("splits:", list(builder.info.splits.keys()))

if "train" in builder.info.splits:
    print("train episodes:", builder.info.splits["train"].num_examples)

ds = builder.as_dataset(
    split="train",
    shuffle_files=False,
)

episode = next(iter(ds.take(1)))

print("episode load: OK")


print("\n" + "=" * 70)
print("2. Episode metadata")

meta = episode["episode_metadata"]

for key in [
    "source_episode_id",
    "episode_index",
    "fps",
    "num_frames",
    "task",
    "task_name",
]:
    if key in meta:
        print(f"{key}: {scalar(meta[key])}")


print("\n" + "=" * 70)
print("3. Find t and t+30")

image_t = None
image_future = None

start_idx = None
target_idx = None
prompt = None

seen_indices = []

for step in episode["steps"]:
    idx = int(scalar(step["frame_index"]))
    seen_indices.append(idx)

    if start_idx is None:
        start_idx = idx
        target_idx = idx + OFFSET

        image_t = step["observation"]["images"]["front_1"].numpy()

        if "prompt" in step:
            prompt = scalar(step["prompt"])

    if idx == target_idx:
        image_future = step["observation"]["images"]["front_1"].numpy()
        break

    if target_idx is not None and idx > target_idx:
        break


print("first frame:", start_idx)
print("target frame:", target_idx)
print("prompt:", prompt)
print("first frame indices:", seen_indices[:40])

if image_t is None:
    raise RuntimeError("Cannot read first frame.")

if image_future is None:
    raise RuntimeError(
        f"Cannot find exact frame pair {start_idx} -> {target_idx}. "
        f"Observed indices start with {seen_indices[:40]}"
    )

print("frame pair: OK")
print("image_t:", image_t.shape, image_t.dtype)
print("image_future:", image_future.shape, image_future.dtype)


print("\n" + "=" * 70)
print("4. Preprocess")

# 和官方 inference_sthv2.py 对齐
transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.Resize((256, 256)),
    T.ToTensor(),
])

img1 = transform(Image.fromarray(image_t))
img2 = transform(Image.fromarray(image_future))

# [C,H,W] -> [C,2,H,W] -> [1,C,2,H,W]
video = torch.cat(
    [
        img1.unsqueeze(1),
        img2.unsqueeze(1),
    ],
    dim=1,
).unsqueeze(0)

print("LAQ input:", tuple(video.shape))
print("range:", float(video.min()), float(video.max()))

assert video.shape == (1, 3, 2, 256, 256)


print("\n" + "=" * 70)
print("5. Load LAQ")

laq = LatentActionQuantization(
    dim=1024,
    quant_dim=32,
    codebook_size=8,
    image_size=256,
    patch_size=32,
    spatial_depth=8,
    temporal_depth=8,
    dim_head=64,
    heads=16,
    code_seq_len=4,
).to(DEVICE)

laq.load(CHECKPOINT)
laq.eval()

print("checkpoint:", CHECKPOINT)
print("checkpoint load: OK")


print("\n" + "=" * 70)
print("6. LAQ inference")

video = video.to(DEVICE)

with torch.inference_mode():
    indices = laq(
        video,
        return_only_codebook_ids=True,
    )

indices = indices.detach().cpu()

print("output shape:", tuple(indices.shape))
print("raw output:")
print(indices)

tokens = indices[0].tolist()

print("latent_action:", tokens)

assert len(tokens) == 4
assert all(0 <= int(x) < 8 for x in tokens)

print("\n" + "=" * 70)
print("SMOKE TEST PASS")
print(
    f"frame={start_idx}->{target_idx}, "
    f"latent_action={tokens}"
)
