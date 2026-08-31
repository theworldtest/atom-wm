import os
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path


DATASET_DIR = (
    "/mnt/data/RLDS/EgoVerse_full/"
    "aria_bimanual_front_1/"
    "ego_verse_infidata/1.0.0"
)

CHECKPOINT = (
    "/mnt/data/laixianxin/LAPA/"
    "lapa_checkpoints/vae.100000.pt"
)

OUTPUT_DIR = (
    "/mnt/data/laixianxin/LAPA/laq/"
    "benchmark_8gpu_out"
)


def worker(
    rank,
    start_episode,
    end_episode,
    batch_size,
    offset,
    output_dir,
):
    # ============================================================
    # 每个 worker 只看自己的一张 GPU
    # ============================================================
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # 必须在设置 CUDA_VISIBLE_DEVICES 后再 import
    import numpy as np
    import tensorflow as tf

    # TensorFlow 只负责读 TFRecord，不碰 GPU
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass

    import tensorflow_datasets as tfds
    import torch

    from collections import deque
    from PIL import Image
    from torchvision import transforms as T
    from laq_model import LatentActionQuantization

    device = "cuda:0"

    torch.cuda.set_device(0)

    print(
        f"[GPU {rank}] "
        f"visible device={torch.cuda.get_device_name(0)} "
        f"episodes={start_episode}:{end_episode}",
        flush=True,
    )

    # ============================================================
    # Helpers
    # ============================================================

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

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    def preprocess(array):
        image = Image.fromarray(array)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return transform(image)

    # ============================================================
    # Load LAQ once
    # ============================================================

    model_t0 = time.perf_counter()

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

    torch.cuda.synchronize()

    model_load_time = time.perf_counter() - model_t0

    print(
        f"[GPU {rank}] model loaded "
        f"in {model_load_time:.2f}s",
        flush=True,
    )

    # ============================================================
    # Load only this worker's episode range
    #
    # train[0:10]
    # train[10:20]
    # ...
    # ============================================================

    builder = tfds.builder_from_directory(DATASET_DIR)

    split = f"train[{start_episode}:{end_episode}]"

    ds = builder.as_dataset(
        split=split,
        shuffle_files=False,
    )

    # ============================================================
    # Statistics
    # ============================================================

    overall_counts = np.zeros(8, dtype=np.int64)
    position_counts = np.zeros((4, 8), dtype=np.int64)

    total_pairs = 0
    total_frames = 0
    total_episodes = 0
    skipped_noncontinuous = 0

    inference_time = 0.0
    preprocess_time = 0.0

    sample_labels = []

    wall_t0 = time.perf_counter()

    # ============================================================
    # Episode loop
    # ============================================================

    for local_ep_idx, episode in enumerate(ds):

        global_ep_idx = start_episode + local_ep_idx

        meta = episode["episode_metadata"]

        episode_id = scalar(
            meta["source_episode_id"]
        )

        num_frames = int(
            scalar(meta["num_frames"])
        )

        fps = int(
            scalar(meta["fps"])
        )

        task = scalar(
            meta["task"]
        )

        total_episodes += 1
        total_frames += num_frames

        frame_buffer = deque(
            maxlen=offset + 1
        )

        batch_videos = []
        batch_metadata = []

        episode_pairs = 0

        def run_batch():
            nonlocal inference_time
            nonlocal total_pairs
            nonlocal episode_pairs

            if not batch_videos:
                return

            video = torch.stack(
                batch_videos,
                dim=0,
            ).to(
                device,
                non_blocking=True,
            )

            torch.cuda.synchronize()
            infer_t0 = time.perf_counter()

            with torch.inference_mode():
                indices = laq(
                    video,
                    return_only_codebook_ids=True,
                )

            torch.cuda.synchronize()

            inference_time += (
                time.perf_counter() - infer_t0
            )

            tokens = (
                indices
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )

            # ----------------------------------------
            # statistics
            # ----------------------------------------

            flat_counts = np.bincount(
                tokens.reshape(-1),
                minlength=8,
            )

            overall_counts[:] += flat_counts

            for pos in range(4):
                position_counts[pos] += np.bincount(
                    tokens[:, pos],
                    minlength=8,
                )

            total_pairs += len(tokens)
            episode_pairs += len(tokens)

            # 保存少量例子用于 sanity check
            if len(sample_labels) < 20:
                for pair_info, token in zip(
                    batch_metadata,
                    tokens.tolist(),
                ):
                    if len(sample_labels) >= 20:
                        break

                    sample_labels.append({
                        "source_episode_id": episode_id,
                        "frame_index": pair_info[0],
                        "future_frame_index": pair_info[1],
                        "task": task,
                        "latent_action": token,
                    })

            batch_videos.clear()
            batch_metadata.clear()

        # ========================================================
        # Step loop
        # ========================================================

        for step in episode["steps"]:

            frame_idx = int(
                scalar(step["frame_index"])
            )

            raw = (
                step["observation"]
                ["images"]
                ["front_1"]
                .numpy()
            )

            pre_t0 = time.perf_counter()

            image = preprocess(raw)

            preprocess_time += (
                time.perf_counter() - pre_t0
            )

            frame_buffer.append(
                (
                    frame_idx,
                    image,
                )
            )

            if len(frame_buffer) < offset + 1:
                continue

            old_idx, old_img = frame_buffer[0]
            new_idx, new_img = frame_buffer[-1]

            if new_idx - old_idx != offset:
                skipped_noncontinuous += 1
                continue

            pair = torch.stack(
                [
                    old_img,
                    new_img,
                ],
                dim=1,
            )

            batch_videos.append(pair)

            batch_metadata.append(
                (
                    old_idx,
                    new_idx,
                )
            )

            if len(batch_videos) >= batch_size:
                run_batch()

        # last incomplete batch
        run_batch()

        print(
            f"[GPU {rank}] "
            f"episode {global_ep_idx} "
            f"id={episode_id} "
            f"fps={fps} "
            f"frames={num_frames} "
            f"pairs={episode_pairs}",
            flush=True,
        )

    # ============================================================
    # Final statistics
    # ============================================================

    wall_time = time.perf_counter() - wall_t0

    peak_memory_gib = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    result = {
        "rank": rank,
        "device": torch.cuda.get_device_name(0),
        "episode_start": start_episode,
        "episode_end": end_episode,
        "episodes_processed": total_episodes,
        "frames": total_frames,
        "valid_pairs": total_pairs,
        "skipped_noncontinuous": skipped_noncontinuous,
        "batch_size": batch_size,
        "offset": offset,
        "model_load_sec": model_load_time,
        "preprocess_sec": preprocess_time,
        "gpu_inference_sec": inference_time,
        "wall_sec": wall_time,
        "gpu_pairs_per_sec": (
            total_pairs / inference_time
            if inference_time > 0 else 0
        ),
        "end_to_end_pairs_per_sec": (
            total_pairs / wall_time
            if wall_time > 0 else 0
        ),
        "peak_gpu_memory_gib": peak_memory_gib,
        "overall_counts": overall_counts.tolist(),
        "position_counts": position_counts.tolist(),
        "samples": sample_labels,
    }

    out_file = (
        Path(output_dir)
        / f"rank_{rank}.json"
    )

    with open(out_file, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[GPU {rank}] DONE "
        f"episodes={total_episodes} "
        f"pairs={total_pairs} "
        f"wall={wall_time:.2f}s "
        f"e2e={result['end_to_end_pairs_per_sec']:.2f} pairs/s "
        f"gpu={result['gpu_pairs_per_sec']:.2f} pairs/s "
        f"peak_mem={peak_memory_gib:.2f} GiB",
        flush=True,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=80,
        help="总共测试多少个train episode",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
    )

    args = parser.parse_args()

    if args.num_episodes % args.num_gpus != 0:
        raise ValueError(
            "--num-episodes 目前要求能被 --num-gpus 整除"
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    episodes_per_gpu = (
        args.num_episodes
        // args.num_gpus
    )

    print("=" * 80)
    print("8-GPU LAQ BENCHMARK")
    print("num_gpus:", args.num_gpus)
    print("num_episodes:", args.num_episodes)
    print("episodes_per_gpu:", episodes_per_gpu)
    print("batch_size:", args.batch_size)
    print("offset:", args.offset)
    print("output:", args.output_dir)
    print("=" * 80)

    ctx = mp.get_context("spawn")

    processes = []

    global_t0 = time.perf_counter()

    for rank in range(args.num_gpus):

        start = rank * episodes_per_gpu
        end = start + episodes_per_gpu

        p = ctx.Process(
            target=worker,
            args=(
                rank,
                start,
                end,
                args.batch_size,
                args.offset,
                args.output_dir,
            ),
        )

        p.start()
        processes.append(p)

    failed = False

    for p in processes:
        p.join()

        if p.exitcode != 0:
            failed = True

    if failed:
        raise RuntimeError(
            "At least one GPU worker failed. "
            "Check logs above."
        )

    global_wall = (
        time.perf_counter() - global_t0
    )

    # ============================================================
    # Merge per-rank summaries
    # ============================================================

    import numpy as np

    results = []

    for rank in range(args.num_gpus):

        f = (
            Path(args.output_dir)
            / f"rank_{rank}.json"
        )

        with open(f) as fp:
            results.append(
                json.load(fp)
            )

    total_pairs = sum(
        x["valid_pairs"]
        for x in results
    )

    total_frames = sum(
        x["frames"]
        for x in results
    )

    total_episodes = sum(
        x["episodes_processed"]
        for x in results
    )

    total_skipped = sum(
        x["skipped_noncontinuous"]
        for x in results
    )

    overall_counts = np.zeros(
        8,
        dtype=np.int64,
    )

    position_counts = np.zeros(
        (4, 8),
        dtype=np.int64,
    )

    for x in results:
        overall_counts += np.array(
            x["overall_counts"],
            dtype=np.int64,
        )

        position_counts += np.array(
            x["position_counts"],
            dtype=np.int64,
        )

    aggregate = {
        "num_gpus": args.num_gpus,
        "episodes": total_episodes,
        "frames": total_frames,
        "valid_pairs": total_pairs,
        "skipped_noncontinuous": total_skipped,
        "batch_size": args.batch_size,
        "offset": args.offset,
        "global_wall_sec": global_wall,
        "aggregate_pairs_per_sec": (
            total_pairs / global_wall
        ),
        "overall_counts": overall_counts.tolist(),
        "position_counts": position_counts.tolist(),
        "per_rank": [
            {
                "rank": x["rank"],
                "episodes": x["episodes_processed"],
                "pairs": x["valid_pairs"],
                "wall_sec": x["wall_sec"],
                "end_to_end_pairs_per_sec":
                    x["end_to_end_pairs_per_sec"],
                "gpu_pairs_per_sec":
                    x["gpu_pairs_per_sec"],
                "peak_gpu_memory_gib":
                    x["peak_gpu_memory_gib"],
            }
            for x in results
        ],
    }

    aggregate_file = (
        Path(args.output_dir)
        / "aggregate.json"
    )

    with open(
        aggregate_file,
        "w",
    ) as f:
        json.dump(
            aggregate,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ============================================================
    # Print summary
    # ============================================================

    print("\n" + "=" * 80)
    print("8-GPU BENCHMARK RESULT")

    print("episodes:", total_episodes)
    print("frames:", total_frames)
    print("valid pairs:", total_pairs)
    print(
        "skipped non-continuous:",
        total_skipped,
    )

    print(
        "global wall:",
        f"{global_wall:.2f}s",
    )

    print(
        "aggregate throughput:",
        f"{total_pairs / global_wall:.2f} pairs/s",
    )

    print("\nPER GPU:")

    for x in results:
        print(
            f"GPU{x['rank']}: "
            f"pairs={x['valid_pairs']} "
            f"wall={x['wall_sec']:.2f}s "
            f"e2e={x['end_to_end_pairs_per_sec']:.2f} "
            f"gpu={x['gpu_pairs_per_sec']:.2f} "
            f"mem={x['peak_gpu_memory_gib']:.2f}GiB"
        )

    print("\nOVERALL CODEBOOK COUNTS")

    total_tokens = overall_counts.sum()

    for code in range(8):
        pct = (
            overall_counts[code]
            / total_tokens
            * 100
        )

        print(
            f"code {code}: "
            f"{overall_counts[code]} "
            f"({pct:.2f}%)"
        )

    print(
        "overall unique:",
        int(
            np.count_nonzero(
                overall_counts
            )
        ),
        "/8",
    )

    print("\nPER POSITION")

    for pos in range(4):
        print(
            f"position {pos}: "
            f"{position_counts[pos].tolist()} "
            f"unique="
            f"{np.count_nonzero(position_counts[pos])}/8"
        )

    print("\naggregate file:")
    print(aggregate_file)

    print("=" * 80)
    print("8-GPU BENCHMARK PASS")


if __name__ == "__main__":
    main()
