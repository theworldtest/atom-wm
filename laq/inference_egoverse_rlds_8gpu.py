import os
import re
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path


# ============================================================
# Utils used before importing torch / tensorflow
# ============================================================

def safe_name(s):
    s = str(s)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def discover_datasets(data_root):
    """
    Expected structure, for example:

    EgoVerse_full/
      aria_bimanual_front_1/
        ego_verse_infidata/
          1.0.0/
            dataset_info.json
            features.json
            *.tfrecord
    """
    root = Path(data_root)

    found = []

    # category / dataset_name / version / dataset_info.json
    for info in sorted(root.glob("*/*/*/dataset_info.json")):
        dataset_dir = info.parent

        if not (dataset_dir / "features.json").exists():
            continue

        rel = dataset_dir.relative_to(root).as_posix()

        found.append({
            "dataset_dir": str(dataset_dir),
            "dataset_key": rel,
        })

    return found


def get_split_num_examples(dataset_dir, split):
    """
    Read TFDS dataset_info.json using stdlib only.

    We intentionally do NOT import tensorflow / tensorflow_datasets
    in the parent process, because multiprocessing "spawn" copies the
    parent's sys.path into child interpreters.

    Supports common TFDS dataset_info.json layouts:
      splits = [
        {
          "name": "train",
          "shardLengths": ["100", "200", ...]
        }
      ]

    and dict-like variants if present.
    """
    info_file = Path(dataset_dir) / "dataset_info.json"

    with open(info_file, "r") as f:
        info = json.load(f)

    splits = info.get("splits", [])

    # --------------------------------------------------------
    # Standard TFDS form: list of split objects
    # --------------------------------------------------------
    if isinstance(splits, list):
        for item in splits:
            if item.get("name") != split:
                continue

            # Most TFDS dataset_info files store shardLengths.
            shard_lengths = item.get("shardLengths")

            if shard_lengths is not None:
                return sum(
                    int(x)
                    for x in shard_lengths
                )

            # Be tolerant of alternate metadata forms.
            for key in (
                "numExamples",
                "num_examples",
                "numexamples",
            ):
                if key in item:
                    return int(item[key])

            raise RuntimeError(
                f"Found split={split} in {info_file}, "
                f"but cannot determine number of examples. "
                f"Keys={sorted(item.keys())}"
            )

        return None

    # --------------------------------------------------------
    # Tolerate dict-like split metadata
    # --------------------------------------------------------
    if isinstance(splits, dict):
        if split not in splits:
            return None

        item = splits[split]

        if isinstance(item, int):
            return int(item)

        if isinstance(item, str) and item.isdigit():
            return int(item)

        if isinstance(item, dict):
            shard_lengths = (
                item.get("shardLengths")
                or item.get("shard_lengths")
            )

            if shard_lengths is not None:
                return sum(
                    int(x)
                    for x in shard_lengths
                )

            for key in (
                "numExamples",
                "num_examples",
                "numexamples",
            ):
                if key in item:
                    return int(item[key])

    raise RuntimeError(
        f"Unsupported splits format in {info_file}: "
        f"{type(splits)}"
    )



# ============================================================
# GPU worker
# ============================================================

def worker(
    rank,
    jobs,
    checkpoint,
    output_dir,
    batch_size,
    offset,
    split,
):
    # --------------------------------------------------------
    # One process -> one physical GPU
    # --------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import numpy as np
    import tensorflow as tf

    # TensorFlow only reads RLDS. Do not let it use GPU.
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass

    import tensorflow_datasets as tfds
    import torch

    from collections import deque
    from PIL import Image
    from torchvision import transforms as T

    from laq_model.latent_action_quantization import LatentActionQuantization

    device = "cuda:0"
    torch.cuda.set_device(0)

    print(
        f"[GPU {rank}] "
        f"device={torch.cuda.get_device_name(0)} "
        f"jobs={len(jobs)}",
        flush=True,
    )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

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

    # Same transform as official inference_sthv2.py
    transform = T.Compose([
        T.Lambda(
            lambda img:
            img.convert("RGB")
            if img.mode != "RGB"
            else img
        ),
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    def preprocess(array):
        return transform(
            Image.fromarray(array)
        )

    # --------------------------------------------------------
    # Load LAQ exactly once per GPU
    # --------------------------------------------------------

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

    laq.load(checkpoint)
    laq.eval()

    torch.cuda.synchronize()

    print(
        f"[GPU {rank}] LAQ loaded "
        f"in {time.perf_counter() - model_t0:.2f}s",
        flush=True,
    )

    # --------------------------------------------------------
    # Rank statistics
    # --------------------------------------------------------

    rank_wall_t0 = time.perf_counter()

    rank_episodes_new = 0
    rank_episodes_skipped = 0
    rank_frames = 0
    rank_labels = 0
    rank_gpu_sec = 0.0

    rank_counts = np.zeros(
        8,
        dtype=np.int64,
    )

    rank_position_counts = np.zeros(
        (4, 8),
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Job = a contiguous range from one RLDS dataset
    # --------------------------------------------------------

    for job in jobs:

        dataset_dir = job["dataset_dir"]
        dataset_key = job["dataset_key"]

        start = job["start"]
        end = job["end"]

        if end <= start:
            continue

        print(
            f"[GPU {rank}] OPEN "
            f"{dataset_key} "
            f"{split}[{start}:{end}]",
            flush=True,
        )

        builder = tfds.builder_from_directory(
            dataset_dir
        )

        ds = builder.as_dataset(
            split=f"{split}[{start}:{end}]",
            shuffle_files=False,
        )

        for local_i, episode in enumerate(ds):

            split_episode_index = (
                start + local_i
            )

            meta = episode["episode_metadata"]

            def meta_value(key, default=None):
                if key not in meta:
                    return default
                return scalar(meta[key])

            source_episode_id = meta_value(
                "source_episode_id",
                f"split_{split_episode_index}",
            )

            task = meta_value(
                "task",
                "",
            )

            task_name = meta_value(
                "task_name",
                "",
            )

            fps = meta_value(
                "fps",
                None,
            )

            metadata_num_frames = meta_value(
                "num_frames",
                None,
            )

            # ------------------------------------------------
            # Each episode gets its own atomic output file.
            #
            # Advantages:
            # - crash-safe
            # - rerun automatically resumes
            # - no 8-process write lock
            # ------------------------------------------------

            episode_dir = (
                Path(output_dir)
                / "episodes"
                / Path(dataset_key)
            )

            episode_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            filename = (
                f"{split_episode_index:06d}_"
                f"{safe_name(source_episode_id)}.jsonl"
            )

            final_file = (
                episode_dir / filename
            )

            # Already finished -> resume skip
            if final_file.exists():
                rank_episodes_skipped += 1

                print(
                    f"[GPU {rank}] SKIP existing "
                    f"{dataset_key} "
                    f"episode={split_episode_index}",
                    flush=True,
                )

                continue

            tmp_file = Path(
                str(final_file)
                + f".tmp.{os.getpid()}"
            )

            episode_t0 = time.perf_counter()

            # =================================================
            # Streaming frame buffer
            #
            # Official semantics:
            #
            # future = min(current + offset, last_frame)
            #
            # During normal streaming:
            #   t -> t+30
            #
            # After end:
            #   last 30 frames -> last frame
            # =================================================

            frame_buffer = deque()

            batch_videos = []
            batch_records = []

            episode_frames = 0
            episode_labels = 0
            episode_discontinuous = 0

            last_item = None

            with open(
                tmp_file,
                "w",
                buffering=1024 * 1024,
            ) as fp:

                def run_batch():
                    nonlocal rank_gpu_sec
                    nonlocal rank_labels
                    nonlocal episode_labels

                    if not batch_videos:
                        return

                    videos = torch.stack(
                        batch_videos,
                        dim=0,
                    )

                    videos = videos.to(
                        device,
                        non_blocking=True,
                    )

                    torch.cuda.synchronize()

                    infer_t0 = time.perf_counter()

                    # Official inference uses no_grad
                    with torch.no_grad():
                        indices = laq(
                            videos,
                            return_only_codebook_ids=True,
                        )

                    torch.cuda.synchronize()

                    rank_gpu_sec += (
                        time.perf_counter()
                        - infer_t0
                    )

                    tokens = (
                        indices
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64)
                    )

                    if tokens.ndim != 2:
                        raise RuntimeError(
                            f"Unexpected LAQ output "
                            f"shape={tokens.shape}"
                        )

                    if tokens.shape[1] != 4:
                        raise RuntimeError(
                            f"Expected 4 latent tokens, "
                            f"got shape={tokens.shape}"
                        )

                    # Statistics
                    rank_counts[:] += np.bincount(
                        tokens.reshape(-1),
                        minlength=8,
                    )

                    for pos in range(4):
                        rank_position_counts[pos] += (
                            np.bincount(
                                tokens[:, pos],
                                minlength=8,
                            )
                        )

                    # Write labels
                    for record, token in zip(
                        batch_records,
                        tokens.tolist(),
                    ):
                        # Official inference_sthv2.py stores
                        # delta tokens as strings.
                        record["delta"] = [
                            str(int(x))
                            for x in token
                        ]

                        fp.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    n = len(tokens)

                    rank_labels += n
                    episode_labels += n

                    batch_videos.clear()
                    batch_records.clear()

                def add_pair(
                    current_item,
                    future_item,
                ):
                    current_img = (
                        current_item["image"]
                    )

                    future_img = (
                        future_item["image"]
                    )

                    # [C,H,W] x 2
                    # -> [C,2,H,W]
                    pair = torch.stack(
                        [
                            current_img,
                            future_img,
                        ],
                        dim=1,
                    )

                    batch_videos.append(
                        pair
                    )

                    instruction = (
                        current_item["prompt"]
                        or task
                        or task_name
                        or ""
                    )

                    # Intermediate record.
                    #
                    # Later VQGAN stage will add:
                    #   vision
                    # and then:
                    #   fields="[instruction],[vision],delta"
                    #
                    # before 7B latent pretraining.
                    record = {
                        "id": (
                            f"{dataset_key}|"
                            f"{source_episode_id}|"
                            f"{current_item['frame_index']}"
                        ),
                        "dataset": dataset_key,
                        "split": split,
                        "split_episode_index":
                            split_episode_index,
                        "source_episode_id":
                            source_episode_id,
                        "frame_index":
                            current_item["frame_index"],
                        "future_frame_index":
                            future_item["frame_index"],
                        "instruction":
                            instruction,
                    }

                    batch_records.append(
                        record
                    )

                    if len(batch_videos) >= batch_size:
                        run_batch()

                # ---------------------------------------------
                # Iterate RLDS steps
                # ---------------------------------------------

                previous_frame_idx = None

                for step in episode["steps"]:

                    frame_idx = int(
                        scalar(
                            step["frame_index"]
                        )
                    )

                    if (
                        previous_frame_idx
                        is not None
                        and frame_idx
                        != previous_frame_idx + 1
                    ):
                        episode_discontinuous += 1

                    previous_frame_idx = frame_idx

                    raw_image = (
                        step["observation"]
                        ["images"]
                        ["front_1"]
                        .numpy()
                    )

                    image = preprocess(
                        raw_image
                    )

                    prompt = ""

                    if "prompt" in step:
                        prompt = scalar(
                            step["prompt"]
                        )

                    item = {
                        "frame_index": frame_idx,
                        "image": image,
                        "prompt": prompt,
                    }

                    frame_buffer.append(
                        item
                    )

                    last_item = item

                    episode_frames += 1

                    # Once we have offset+1 frames:
                    #
                    # oldest -> newest
                    #
                    # This is positional t -> t+offset,
                    # matching official window semantics.
                    if len(frame_buffer) > offset:

                        current_item = (
                            frame_buffer.popleft()
                        )

                        future_item = item

                        add_pair(
                            current_item,
                            future_item,
                        )

                # ---------------------------------------------
                # Official tail clamp:
                #
                # future=min(t+offset,last)
                #
                # The frames still in frame_buffer are the
                # final <=offset frames.
                # ---------------------------------------------

                if last_item is not None:

                    while frame_buffer:

                        current_item = (
                            frame_buffer.popleft()
                        )

                        add_pair(
                            current_item,
                            last_item,
                        )

                run_batch()

            # Every input step should have exactly one delta label
            if episode_labels != episode_frames:

                raise RuntimeError(
                    f"Episode label mismatch: "
                    f"id={source_episode_id}, "
                    f"frames={episode_frames}, "
                    f"labels={episode_labels}"
                )

            # Atomic completion
            os.replace(
                tmp_file,
                final_file,
            )

            rank_episodes_new += 1
            rank_frames += episode_frames

            elapsed = (
                time.perf_counter()
                - episode_t0
            )

            print(
                f"[GPU {rank}] DONE "
                f"{dataset_key} "
                f"episode={split_episode_index} "
                f"id={source_episode_id} "
                f"fps={fps} "
                f"meta_frames={metadata_num_frames} "
                f"read_frames={episode_frames} "
                f"labels={episode_labels} "
                f"frame_jumps={episode_discontinuous} "
                f"time={elapsed:.2f}s",
                flush=True,
            )

    # --------------------------------------------------------
    # Save worker summary
    # --------------------------------------------------------

    rank_wall = (
        time.perf_counter()
        - rank_wall_t0
    )

    summary = {
        "rank": rank,
        "device":
            torch.cuda.get_device_name(0),
        "episodes_new":
            rank_episodes_new,
        "episodes_skipped":
            rank_episodes_skipped,
        "frames_new":
            rank_frames,
        "labels_new":
            rank_labels,
        "wall_sec":
            rank_wall,
        "gpu_inference_sec":
            rank_gpu_sec,
        "new_labels_per_sec":
            (
                rank_labels / rank_wall
                if rank_wall > 0
                else 0
            ),
        "peak_gpu_memory_gib":
            (
                torch.cuda.max_memory_allocated()
                / 1024**3
            ),
        "overall_counts_new":
            rank_counts.tolist(),
        "position_counts_new":
            rank_position_counts.tolist(),
    }

    summary_file = (
        Path(output_dir)
        / f"rank_{rank}_summary.json"
    )

    with open(
        summary_file,
        "w",
    ) as fp:
        json.dump(
            summary,
            fp,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[GPU {rank}] ALL DONE "
        f"new_episodes={rank_episodes_new} "
        f"skipped={rank_episodes_skipped} "
        f"labels={rank_labels} "
        f"wall={rank_wall:.1f}s",
        flush=True,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="/mnt/data/RLDS/EgoVerse_full",
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "/mnt/data/laixianxin/LAPA/"
            "lapa_checkpoints/vae.100000.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "/mnt/data/laixianxin/LAPA/"
            "data/egoverse_laq_labels"
        ),
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
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
        "--split",
        default="train",
    )

    parser.add_argument(
        "--max-episodes-per-dataset",
        type=int,
        default=None,
        help=(
            "Only process the first N episodes of each RLDS dataset. "
            "Default=None means full dataset."
        ),
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Discover every EgoVerse RLDS dataset
    # --------------------------------------------------------

    datasets = discover_datasets(
        args.data_root
    )

    if not datasets:
        raise RuntimeError(
            f"No RLDS datasets found under "
            f"{args.data_root}"
        )

    # ========================================================
    # IMPORTANT:
    #
    # Parent process must NOT import TensorFlow / TFDS.
    #
    # multiprocessing spawn inherits the parent's sys.path.
    # Keep the parent process stdlib-only before spawning GPUs.
    #
    # We obtain split sizes directly from dataset_info.json.
    # ========================================================

    total_episodes = 0

    print("=" * 80)
    print("DISCOVERED RLDS DATASETS")

    for d in datasets:

        original_n = get_split_num_examples(
            d["dataset_dir"],
            args.split,
        )

        if original_n is None:
            d["original_num_examples"] = 0
            d["num_examples"] = 0

            print(
                f"SKIP(no {args.split}): "
                f"{d['dataset_key']}"
            )

            continue

        original_n = int(original_n)

        n = original_n

        if args.max_episodes_per_dataset is not None:
            n = min(
                n,
                args.max_episodes_per_dataset,
            )

        d["original_num_examples"] = original_n
        d["num_examples"] = int(n)

        total_episodes += int(n)

        if n != original_n:
            print(
                f"{d['dataset_key']}: "
                f"{args.split}={original_n} "
                f"-> smoke={n}"
            )
        else:
            print(
                f"{d['dataset_key']}: "
                f"{args.split}={n}"
            )

    print("-" * 80)

    print(
        "TOTAL TRAIN EPISODES:",
        total_episodes,
    )

    print(
        "GPUs:",
        args.num_gpus,
    )

    print(
        "batch_size:",
        args.batch_size,
    )

    print(
        "offset:",
        args.offset,
    )

    print(
        "checkpoint:",
        args.checkpoint,
    )

    print(
        "output:",
        args.output_dir,
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Split EVERY dataset approximately evenly across 8 GPUs.
    #
    # For a dataset with N episodes:
    #
    # GPU0: [0, N/8)
    # GPU1: [N/8, 2N/8)
    # ...
    #
    # Each GPU processes part of every embodiment.
    # --------------------------------------------------------

    jobs_by_rank = [
        []
        for _ in range(args.num_gpus)
    ]

    for d in datasets:

        n = d["num_examples"]

        if n <= 0:
            continue

        for rank in range(args.num_gpus):

            start = (
                n * rank
                // args.num_gpus
            )

            end = (
                n * (rank + 1)
                // args.num_gpus
            )

            if end <= start:
                continue

            jobs_by_rank[rank].append({
                "dataset_dir":
                    d["dataset_dir"],
                "dataset_key":
                    d["dataset_key"],
                "start":
                    start,
                "end":
                    end,
            })

    for rank, jobs in enumerate(
        jobs_by_rank
    ):
        n = sum(
            j["end"] - j["start"]
            for j in jobs
        )

        print(
            f"GPU{rank}: "
            f"{n} episodes "
            f"across {len(jobs)} datasets"
        )

    # --------------------------------------------------------
    # Launch 8 independent inference workers
    # --------------------------------------------------------

    ctx = mp.get_context("spawn")

    processes = []

    global_t0 = time.perf_counter()

    for rank in range(args.num_gpus):

        p = ctx.Process(
            target=worker,
            args=(
                rank,
                jobs_by_rank[rank],
                args.checkpoint,
                args.output_dir,
                args.batch_size,
                args.offset,
                args.split,
            ),
        )

        p.start()
        processes.append(p)

    failed = []

    for rank, p in enumerate(
        processes
    ):
        p.join()

        if p.exitcode != 0:
            failed.append(
                (rank, p.exitcode)
            )

    if failed:
        raise RuntimeError(
            f"Workers failed: {failed}. "
            f"Completed episode files are kept. "
            f"Re-run the same command to resume."
        )

    global_worker_wall = (
        time.perf_counter()
        - global_t0
    )

    # --------------------------------------------------------
    # Merge atomic per-episode labels into one JSONL.
    #
    # At the same time recompute GLOBAL codebook stats from
    # every completed episode file. This also works correctly
    # after a resumed run.
    # --------------------------------------------------------

    import numpy as np

    episode_root = (
        Path(args.output_dir)
        / "episodes"
    )

    episode_files = sorted(
        episode_root.rglob("*.jsonl")
    )

    final_jsonl = (
        Path(args.output_dir)
        / f"latent_action_egoverse_{args.split}.jsonl"
    )

    final_tmp = Path(
        str(final_jsonl)
        + ".tmp"
    )

    global_counts = np.zeros(
        8,
        dtype=np.int64,
    )

    position_counts = np.zeros(
        (4, 8),
        dtype=np.int64,
    )

    total_labels = 0

    merge_t0 = time.perf_counter()

    with open(
        final_tmp,
        "w",
        buffering=1024 * 1024,
    ) as out:

        for ep_file in episode_files:

            with open(
                ep_file,
                "r",
            ) as inp:

                for line in inp:

                    if not line.strip():
                        continue

                    obj = json.loads(
                        line
                    )

                    delta = np.asarray(
                        [
                            int(x)
                            for x in obj[
                                "delta"
                            ]
                        ],
                        dtype=np.int64,
                    )

                    if delta.shape != (4,):
                        raise RuntimeError(
                            f"Bad delta in "
                            f"{ep_file}: "
                            f"{delta}"
                        )

                    global_counts += (
                        np.bincount(
                            delta,
                            minlength=8,
                        )
                    )

                    for pos in range(4):
                        position_counts[
                            pos,
                            delta[pos],
                        ] += 1

                    out.write(line)

                    total_labels += 1

    os.replace(
        final_tmp,
        final_jsonl,
    )

    merge_sec = (
        time.perf_counter()
        - merge_t0
    )

    # --------------------------------------------------------
    # Final manifest
    # --------------------------------------------------------

    manifest = {
        "data_root":
            args.data_root,
        "split":
            args.split,
        "num_gpus":
            args.num_gpus,
        "batch_size":
            args.batch_size,
        "offset":
            args.offset,
        "codebook_size":
            8,
        "code_seq_len":
            4,
        "checkpoint":
            args.checkpoint,
        "datasets": datasets,
        "completed_episode_files":
            len(episode_files),
        "total_labels":
            total_labels,
        "worker_wall_sec":
            global_worker_wall,
        "merge_sec":
            merge_sec,
        "overall_code_counts":
            global_counts.tolist(),
        "position_code_counts":
            position_counts.tolist(),
        "overall_unique_codes":
            int(
                np.count_nonzero(
                    global_counts
                )
            ),
        "final_jsonl":
            str(final_jsonl),
    }

    manifest_file = (
        Path(args.output_dir)
        / "manifest.json"
    )

    with open(
        manifest_file,
        "w",
    ) as fp:
        json.dump(
            manifest,
            fp,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("EgoVerse LAQ LABELING COMPLETE")

    print(
        "completed episodes:",
        len(episode_files),
    )

    print(
        "total latent labels:",
        total_labels,
    )

    print(
        "8-GPU worker wall:",
        f"{global_worker_wall:.1f}s",
    )

    print(
        "merge time:",
        f"{merge_sec:.1f}s",
    )

    if global_worker_wall > 0:
        print(
            "aggregate throughput:",
            f"{total_labels / global_worker_wall:.2f} labels/s",
        )

    print("\nOVERALL CODEBOOK")

    total_tokens = (
        global_counts.sum()
    )

    for i in range(8):

        pct = (
            100
            * global_counts[i]
            / total_tokens
            if total_tokens > 0
            else 0
        )

        print(
            f"code {i}: "
            f"{global_counts[i]} "
            f"({pct:.2f}%)"
        )

    print(
        "unique:",
        np.count_nonzero(
            global_counts
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

    print("\nFINAL JSONL:")
    print(final_jsonl)

    print("\nMANIFEST:")
    print(manifest_file)

    print("=" * 80)


if __name__ == "__main__":
    main()
