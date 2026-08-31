import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import time
from collections import deque

import numpy as np
import tensorflow as tf

# TensorFlow 只读 RLDS，不占 H800
try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

import tensorflow_datasets as tfds
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


def scalar(x):
    if hasattr(x, "numpy"):
        x = x.numpy()

    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()

    if isinstance(x, np.bytes_):
        x = bytes(x)

    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")

    return x


def build_laq(device):
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
    ).to(device)

    laq.load(CHECKPOINT)
    laq.eval()

    return laq


def print_counts(all_tokens):
    arr = np.asarray(all_tokens, dtype=np.int64)

    print("\n" + "=" * 70)
    print("CODEBOOK USAGE")

    total = np.bincount(
        arr.reshape(-1),
        minlength=8
    )

    print("\nOverall counts:")
    for code, count in enumerate(total):
        pct = count / total.sum() * 100
        print(
            f"code {code}: "
            f"{count:6d} "
            f"({pct:6.2f}%)"
        )

    print(
        "\nOverall unique codes:",
        int(np.count_nonzero(total))
    )

    print("\nPer-position counts:")

    for pos in range(4):
        counts = np.bincount(
            arr[:, pos],
            minlength=8
        )

        print(f"\nposition {pos}:")
        print("  ", counts.tolist())

        used = np.count_nonzero(counts)
        print(f"   unique={used}/8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="第几个 train episode，默认0",
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    device = "cuda:0"

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see CUDA.")

    print("=" * 70)
    print("GPU")
    print("device:", torch.cuda.get_device_name(0))
    print("batch size:", args.batch_size)
    print("offset:", args.offset)

    # ------------------------------------------------------------
    # Load RLDS episode
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOAD RLDS")

    t0 = time.perf_counter()

    builder = tfds.builder_from_directory(
        DATASET_DIR
    )

    ds = builder.as_dataset(
        split="train",
        shuffle_files=False,
    )

    episode = next(
        iter(
            ds.skip(args.episode_index).take(1)
        )
    )

    metadata = episode["episode_metadata"]

    episode_id = scalar(
        metadata["source_episode_id"]
    )

    fps = int(
        scalar(metadata["fps"])
    )

    num_frames = int(
        scalar(metadata["num_frames"])
    )

    task = scalar(
        metadata["task"]
    )

    print("episode_index:", args.episode_index)
    print("source_episode_id:", episode_id)
    print("fps:", fps)
    print("num_frames:", num_frames)
    print("task:", task)

    print(
        "episode open time:",
        f"{time.perf_counter() - t0:.3f}s"
    )

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOAD LAQ")

    torch.cuda.empty_cache()

    t_model = time.perf_counter()

    laq = build_laq(device)

    torch.cuda.synchronize()

    model_load_sec = (
        time.perf_counter() - t_model
    )

    print(
        "checkpoint load time:",
        f"{model_load_sec:.3f}s"
    )

    mem_gb = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    print(
        "model GPU memory:",
        f"{mem_gb:.2f} GiB"
    )

    # ------------------------------------------------------------
    # Image preprocess
    # ------------------------------------------------------------

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    def preprocess(image_array):
        image = Image.fromarray(
            image_array
        )

        if image.mode != "RGB":
            image = image.convert("RGB")

        return transform(image)

    # 只保存最近 offset+1 帧
    #
    # deque 中元素：
    # (frame_index, processed_image)
    #
    frame_buffer = deque(
        maxlen=args.offset + 1
    )

    batch_videos = []
    batch_pair_indices = []

    all_tokens = []
    all_pair_indices = []

    preprocess_sec = 0.0
    inference_sec = 0.0

    pairs_processed = 0
    skipped_noncontinuous = 0
    batch_number = 0

    def run_batch():
        nonlocal inference_sec
        nonlocal pairs_processed
        nonlocal batch_number

        if not batch_videos:
            return

        video = torch.stack(
            batch_videos,
            dim=0
        )

        # video:
        # [B, C, 2, H, W]

        video = video.to(
            device,
            non_blocking=True,
        )

        # 首批开始前确保之前操作结束
        torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            indices = laq(
                video,
                return_only_codebook_ids=True,
            )

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter() - start
        )

        inference_sec += elapsed

        tokens = (
            indices
            .detach()
            .cpu()
            .numpy()
        )

        assert tokens.shape[1] == 4

        all_tokens.extend(
            tokens.tolist()
        )

        all_pair_indices.extend(
            batch_pair_indices
        )

        pairs_processed += len(tokens)
        batch_number += 1

        if (
            batch_number <= 3
            or batch_number % 10 == 0
        ):
            print(
                f"batch={batch_number:3d} "
                f"pairs={pairs_processed:5d} "
                f"last_batch={len(tokens):3d} "
                f"gpu_time={elapsed:.3f}s"
            )

        batch_videos.clear()
        batch_pair_indices.clear()

    # ------------------------------------------------------------
    # Iterate episode
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROCESS FULL EPISODE")

    wall_start = time.perf_counter()

    for step in episode["steps"]:

        frame_idx = int(
            scalar(step["frame_index"])
        )

        raw_image = (
            step["observation"]
                ["images"]
                ["front_1"]
                .numpy()
        )

        t_pre = time.perf_counter()

        image = preprocess(
            raw_image
        )

        preprocess_sec += (
            time.perf_counter() - t_pre
        )

        frame_buffer.append(
            (
                frame_idx,
                image,
            )
        )

        # buffer 还不够 offset+1
        if len(frame_buffer) < args.offset + 1:
            continue

        old_idx, old_img = frame_buffer[0]
        new_idx, new_img = frame_buffer[-1]

        # 必须确认是真正 frame t -> t+30
        if new_idx - old_idx != args.offset:
            skipped_noncontinuous += 1
            continue

        # [C,H,W] + [C,H,W]
        # -> [C,2,H,W]
        pair = torch.stack(
            [
                old_img,
                new_img,
            ],
            dim=1,
        )

        batch_videos.append(pair)

        batch_pair_indices.append(
            (
                old_idx,
                new_idx,
            )
        )

        if len(batch_videos) >= args.batch_size:
            run_batch()

    # 最后不足一个 batch 的也跑
    run_batch()

    wall_sec = (
        time.perf_counter() - wall_start
    )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BENCHMARK RESULT")

    print(
        "episode:",
        episode_id
    )

    print(
        "frames:",
        num_frames
    )

    print(
        "valid pairs:",
        pairs_processed
    )

    print(
        "skipped non-continuous:",
        skipped_noncontinuous
    )

    print(
        "batch size:",
        args.batch_size
    )

    print(
        "number of batches:",
        batch_number
    )

    print(
        "preprocess time:",
        f"{preprocess_sec:.3f}s"
    )

    print(
        "pure GPU inference time:",
        f"{inference_sec:.3f}s"
    )

    print(
        "episode wall time:",
        f"{wall_sec:.3f}s"
    )

    if inference_sec > 0:
        print(
            "GPU inference throughput:",
            f"{pairs_processed / inference_sec:.2f} pairs/s"
        )

    if wall_sec > 0:
        print(
            "end-to-end throughput:",
            f"{pairs_processed / wall_sec:.2f} pairs/s"
        )

    print(
        "peak GPU memory:",
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
    )

    # ------------------------------------------------------------
    # First few examples
    # ------------------------------------------------------------

    print("\nFirst 10 latent labels:")

    for pair_idx, tokens in list(
        zip(
            all_pair_indices,
            all_tokens,
        )
    )[:10]:
        print(
            f"{pair_idx[0]:4d}"
            f" -> "
            f"{pair_idx[1]:4d}"
            f" : "
            f"{tokens}"
        )

    # ------------------------------------------------------------
    # Token distribution
    # ------------------------------------------------------------

    if all_tokens:
        print_counts(
            all_tokens
        )

    print("\n" + "=" * 70)
    print("FULL EPISODE BENCHMARK PASS")


if __name__ == "__main__":
    main()
