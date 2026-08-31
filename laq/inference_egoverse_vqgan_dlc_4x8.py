#!/usr/bin/env python3
"""
EgoVerse RLDS -> LAPA VQGAN vision-token extraction for Alibaba PAI-DLC
multi-node jobs (e.g. 4 nodes x 8 GPUs = 32 GPUs total).

Key points
----------
- DLC injects RANK and WORLD_SIZE per node.
- One Python parent process runs on each node.
- Each node spawns one independent JAX worker per local GPU.
- global_gpu_rank = node_rank * local_gpu_count + local_rank
- Dataset sharding is by global_gpu_rank / total_gpu_count, so nodes never
  intentionally process the same episode.
- No DDP/NCCL/JAX multi-host collectives are used.
- TensorFlow is CPU-only and reads RLDS.
- JAX owns exactly one local GPU per worker.
- Atomic per-episode JSONL writes + filename-based resume.
- Shared-NAS filesystem barrier lets node 0 write a complete manifest after
  every node finishes.
"""

import os
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path


def discover_dataset_dirs(data_root, dataset_filter=None):
    root = Path(data_root)
    out = []
    for p in sorted(root.glob("*/*/*/dataset_info.json")):
        dataset_dir = p.parent
        rel = dataset_dir.relative_to(root)
        group = rel.parts[0]
        key = rel.as_posix()
        if dataset_filter and dataset_filter not in group and dataset_filter not in key:
            continue
        out.append((group, str(dataset_dir)))
    if not out:
        raise RuntimeError(
            f"No TFDS datasets found under {data_root}"
            + (f" matching {dataset_filter!r}" if dataset_filter else "")
        )
    return out


def parse_visible_devices(local_gpus):
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        toks = [x.strip() for x in raw.split(",") if x.strip()]
        if len(toks) < local_gpus:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES has {len(toks)} entries but "
                f"--local-gpus={local_gpus}: {raw}"
            )
        return toks[:local_gpus]
    return [str(i) for i in range(local_gpus)]


def contiguous_pending_ranges(start, end, done):
    ranges = []
    cur = None
    for i in range(start, end):
        if i in done:
            if cur is not None:
                ranges.append((cur, i))
                cur = None
        elif cur is None:
            cur = i
    if cur is not None:
        ranges.append((cur, end))
    return ranges


def as_py_scalar(x):
    try:
        x = x.numpy()
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, np.ndarray) and x.shape == ():
            x = x.item()
        elif isinstance(x, np.generic):
            x = x.item()
    except Exception:
        pass
    return x


def as_str(x):
    x = as_py_scalar(x)
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def worker_main(
    node_rank,
    local_rank,
    global_rank,
    total_gpus,
    device_token,
    args_dict,
    datasets,
):
    # Must be set before TF/JAX imports.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_token)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[k] = str(args_dict["cpu_threads_per_worker"])

    # We intentionally use the pip CUDA12/cuDNN8.9 stack validated in lapa.
    os.environ.pop("LD_LIBRARY_PATH", None)

    import numpy as np
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(args_dict["tf_intra_threads"])
        tf.config.threading.set_inter_op_parallelism_threads(args_dict["tf_inter_threads"])
    except Exception:
        pass

    import tensorflow_datasets as tfds
    import albumentations as A
    import cv2

    try:
        cv2.setNumThreads(args_dict["opencv_threads"])
    except Exception:
        pass

    import jax
    from latent_pretraining.vqgan import VQGAN

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(
            f"[global {global_rank}] expected exactly one JAX GPU, got {devices}"
        )

    print(
        f"[node {node_rank} local {local_rank} global {global_rank:02d}] "
        f"device_token={device_token} jax_devices={devices}",
        flush=True,
    )

    preprocessor = A.Compose([
        A.LongestMaxSize(max_size=256),
        A.Resize(256, 256),
    ])

    t0 = time.perf_counter()
    vqgan = VQGAN(args_dict["vqgan_checkpoint"], replicate=False)
    print(
        f"[global {global_rank:02d}] VQGAN loaded in "
        f"{time.perf_counter()-t0:.2f}s",
        flush=True,
    )

    # Compile once at the fixed batch size.
    warm = np.zeros((args_dict["batch_size"], 256, 256, 3), dtype=np.float32)
    warm_ids = jax.device_get(vqgan.encode(warm))[1]
    if warm_ids.ndim != 3 or tuple(warm_ids.shape[1:]) != (16, 16):
        raise RuntimeError(
            f"[global {global_rank}] unexpected warmup VQ shape {warm_ids.shape}"
        )
    del warm, warm_ids

    out_root = Path(args_dict["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_episodes = 0
    total_skipped = 0
    processed_for_rank = 0
    rank_start = time.perf_counter()
    max_eps_rank = args_dict["max_episodes_per_rank"]

    for group, dataset_dir in datasets:
        if max_eps_rank is not None and processed_for_rank >= max_eps_rank:
            break

        builder = tfds.builder_from_directory(dataset_dir)
        n_episodes = int(builder.info.splits["train"].num_examples)

        # CRITICAL: shard by GLOBAL GPU rank, not local rank.
        start = global_rank * n_episodes // total_gpus
        end = (global_rank + 1) * n_episodes // total_gpus

        group_dir = out_root / group / "episodes"
        group_dir.mkdir(parents=True, exist_ok=True)

        done = set()
        for p in group_dir.glob("ep_*.jsonl"):
            try:
                done.add(int(p.stem.split("_", 1)[1]))
            except Exception:
                pass

        assigned_done = {i for i in done if start <= i < end}
        total_skipped += len(assigned_done)
        pending_ranges = contiguous_pending_ranges(start, end, assigned_done)

        print(
            f"[global {global_rank:02d}] {group}: total={n_episodes} "
            f"assigned=[{start},{end}) existing={len(assigned_done)} "
            f"pending_ranges={len(pending_ranges)}",
            flush=True,
        )

        stop_dataset = False
        for a, b in pending_ranges:
            if stop_dataset:
                break

            ds = builder.as_dataset(
                split=f"train[{a}:{b}]",
                shuffle_files=False,
                read_config=tfds.ReadConfig(
                    skip_prefetch=True,
                    try_autocache=False,
                ),
            )

            for local_i, episode in enumerate(ds):
                if max_eps_rank is not None and processed_for_rank >= max_eps_rank:
                    stop_dataset = True
                    break

                fallback_ep_idx = a + local_i
                metadata = episode.get("episode_metadata", {})

                if isinstance(metadata, dict) and "episode_index" in metadata:
                    ep_idx = int(as_py_scalar(metadata["episode_index"]))
                else:
                    ep_idx = fallback_ep_idx

                # TFDS slice position is authoritative for filenames/resume.
                if ep_idx < a or ep_idx >= b:
                    ep_idx = fallback_ep_idx

                final_path = group_dir / f"ep_{ep_idx:06d}.jsonl"
                if final_path.exists():
                    total_skipped += 1
                    continue

                if isinstance(metadata, dict) and "source_episode_id" in metadata:
                    source_episode_id = as_str(metadata["source_episode_id"])
                else:
                    source_episode_id = f"{group}:{ep_idx}"

                tmp_path = group_dir / (
                    f".ep_{ep_idx:06d}.g{global_rank:02d}.pid{os.getpid()}.tmp"
                )

                ep_start = time.perf_counter()
                ep_frames = 0
                batch_images = []
                batch_indices = []

                def flush_batch(fout):
                    nonlocal ep_frames, batch_images, batch_indices
                    if not batch_images:
                        return

                    n_real = len(batch_images)
                    imgs = np.asarray([
                        preprocessor(image=img.astype(np.uint8))["image"]
                        for img in batch_images
                    ])
                    imgs = (imgs / 127.5 - 1.0).astype(np.float32)

                    # Keep a fixed JAX shape for the final partial batch.
                    if n_real < args_dict["batch_size"]:
                        pad = args_dict["batch_size"] - n_real
                        imgs = np.concatenate(
                            [imgs, np.repeat(imgs[-1:], pad, axis=0)],
                            axis=0,
                        )

                    ids = jax.device_get(vqgan.encode(imgs))[1]
                    ids = ids.astype(np.int32)[:n_real]

                    if ids.ndim != 3 or tuple(ids.shape[1:]) != (16, 16):
                        raise RuntimeError(
                            f"{group} ep={ep_idx}: unexpected VQ shape {ids.shape}"
                        )
                    if int(ids.min()) < 0 or int(ids.max()) >= 8192:
                        raise RuntimeError(
                            f"{group} ep={ep_idx}: VQ ids out of range "
                            f"{int(ids.min())}..{int(ids.max())}"
                        )

                    for frame_index, code in zip(batch_indices, ids):
                        rec = {
                            "dataset": group,
                            "split": "train",
                            "split_episode_index": ep_idx,
                            "source_episode_id": source_episode_id,
                            "frame_index": int(frame_index),
                            "vision": code.reshape(-1).tolist(),
                        }
                        fout.write(json.dumps(rec, separators=(",", ":")) + "\n")
                        ep_frames += 1

                    batch_images = []
                    batch_indices = []

                try:
                    with open(tmp_path, "w", buffering=1024 * 1024) as fout:
                        for default_frame_idx, step in enumerate(episode["steps"]):
                            if "frame_index" in step:
                                frame_idx = int(as_py_scalar(step["frame_index"]))
                            else:
                                frame_idx = default_frame_idx

                            img = step["observation"]["images"]["front_1"].numpy()
                            batch_images.append(img)
                            batch_indices.append(frame_idx)

                            if len(batch_images) == args_dict["batch_size"]:
                                flush_batch(fout)

                        flush_batch(fout)

                    os.replace(tmp_path, final_path)

                except Exception:
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass
                    raise

                total_frames += ep_frames
                total_episodes += 1
                processed_for_rank += 1

                elapsed = time.perf_counter() - ep_start
                print(
                    f"[global {global_rank:02d}] DONE {group} ep={ep_idx} "
                    f"frames={ep_frames} sec={elapsed:.2f} "
                    f"fps={(ep_frames / elapsed if elapsed else 0):.1f}",
                    flush=True,
                )

    elapsed = time.perf_counter() - rank_start
    stats = {
        "node_rank": node_rank,
        "local_rank": local_rank,
        "global_rank": global_rank,
        "device_token": str(device_token),
        "episodes_written": total_episodes,
        "episodes_skipped": total_skipped,
        "frames_written": total_frames,
        "elapsed_sec": elapsed,
        "frames_per_sec": total_frames / elapsed if elapsed else 0.0,
    }

    stats_path = out_root / f"global_rank_{global_rank:02d}.stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(
        f"[global {global_rank:02d}] COMPLETE {json.dumps(stats)}",
        flush=True,
    )


def build_manifest(out_root, total_gpus, args, datasets):
    stats = []
    missing = []
    for global_rank in range(total_gpus):
        p = out_root / f"global_rank_{global_rank:02d}.stats.json"
        if not p.exists():
            missing.append(str(p))
            continue
        with open(p) as f:
            stats.append(json.load(f))

    if missing:
        raise RuntimeError(
            "Cannot build complete manifest; missing rank stats:\n"
            + "\n".join(missing)
        )

    summary = {
        "world_size_nodes": args.world_size,
        "local_gpus_per_node": args.local_gpus,
        "total_gpus": total_gpus,
        "batch_size": args.batch_size,
        "dataset_filter": args.dataset_filter,
        "max_episodes_per_rank": args.max_episodes_per_rank,
        "datasets": [x[0] for x in datasets],
        "episodes_written_this_run": sum(x["episodes_written"] for x in stats),
        "episodes_skipped_this_run": sum(x["episodes_skipped"] for x in stats),
        "frames_written_this_run": sum(x["frames_written"] for x in stats),
        "sum_worker_elapsed_sec": sum(x["elapsed_sec"] for x in stats),
        "vision_tokens_per_frame": 256,
        "vq_vocab_size": 8192,
        "output_root": str(out_root),
    }

    tmp = out_root / ".manifest_run.json.tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, out_root / "manifest_run.json")

    print("=" * 80)
    print("EgoVerse VQGAN DLC MULTINODE EXTRACTION COMPLETE")
    print(json.dumps(summary, indent=2))
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/mnt/data/RLDS/EgoVerse_full")
    ap.add_argument(
        "--output-root",
        default="/mnt/data/laixianxin/LAPA/data/egoverse_vqgan_tokens",
    )
    ap.add_argument(
        "--vqgan-checkpoint",
        default="/mnt/data/laixianxin/LAPA/lapa_checkpoints/vqgan",
    )
    ap.add_argument(
        "--node-rank",
        type=int,
        default=int(os.environ.get("RANK", "0")),
    )
    ap.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", "1")),
    )
    ap.add_argument(
        "--local-gpus",
        type=int,
        default=int(os.environ.get("NPROC_PER_NODE", "8")),
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dataset-filter", default=None)
    ap.add_argument("--max-episodes-per-rank", type=int, default=None)
    ap.add_argument("--cpu-threads-per-worker", type=int, default=1)
    ap.add_argument("--tf-intra-threads", type=int, default=2)
    ap.add_argument("--tf-inter-threads", type=int, default=1)
    ap.add_argument("--opencv-threads", type=int, default=1)
    ap.add_argument(
        "--barrier-timeout-sec",
        type=int,
        default=3600,
        help="Filesystem barrier timeout for node 0 after local workers finish.",
    )
    args = ap.parse_args()

    if args.node_rank < 0 or args.node_rank >= args.world_size:
        raise ValueError(
            f"node_rank={args.node_rank} invalid for world_size={args.world_size}"
        )
    if not Path(args.vqgan_checkpoint).exists():
        raise FileNotFoundError(args.vqgan_checkpoint)

    total_gpus = args.world_size * args.local_gpus
    datasets = discover_dataset_dirs(args.data_root, args.dataset_filter)

    print("=" * 80)
    print(
        f"DLC topology: node_rank={args.node_rank}/{args.world_size}, "
        f"local_gpus={args.local_gpus}, total_gpus={total_gpus}"
    )
    print(f"hostname={os.uname().nodename}")
    print("Datasets:")
    for group, d in datasets:
        print(f"  {group}: {d}")
    print("=" * 80)

    visible = parse_visible_devices(args.local_gpus)
    print(f"Local visible GPU tokens: {visible}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Remove this node's old done marker so a rerun cannot falsely satisfy barrier.
    node_done = out_root / f"node_{args.node_rank:02d}.done.json"
    if node_done.exists():
        node_done.unlink()

    args_dict = vars(args).copy()

    ctx = mp.get_context("spawn")
    procs = []
    for local_rank in range(args.local_gpus):
        global_rank = args.node_rank * args.local_gpus + local_rank
        p = ctx.Process(
            target=worker_main,
            args=(
                args.node_rank,
                local_rank,
                global_rank,
                total_gpus,
                visible[local_rank],
                args_dict,
                datasets,
            ),
        )
        p.start()
        procs.append((local_rank, global_rank, p))

    failed = []
    for local_rank, global_rank, p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append(
                {
                    "local_rank": local_rank,
                    "global_rank": global_rank,
                    "exitcode": p.exitcode,
                }
            )

    if failed:
        raise RuntimeError(
            f"Node {args.node_rank} workers failed: {failed}"
        )

    # Atomic shared-filesystem node completion marker.
    done_payload = {
        "node_rank": args.node_rank,
        "world_size": args.world_size,
        "local_gpus": args.local_gpus,
        "total_gpus": total_gpus,
        "hostname": os.uname().nodename,
        "finished_at_unix": time.time(),
    }
    node_tmp = out_root / f".node_{args.node_rank:02d}.done.tmp"
    with open(node_tmp, "w") as f:
        json.dump(done_payload, f, indent=2)
    os.replace(node_tmp, node_done)

    print(
        f"[node {args.node_rank}] local 8-GPU work COMPLETE; "
        f"wrote {node_done}",
        flush=True,
    )

    # Only node 0 waits for the other nodes and creates the final manifest.
    if args.node_rank == 0:
        deadline = time.time() + args.barrier_timeout_sec
        expected = [
            out_root / f"node_{i:02d}.done.json"
            for i in range(args.world_size)
        ]

        while True:
            missing = [p for p in expected if not p.exists()]
            if not missing:
                break
            if time.time() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for DLC nodes: "
                    + ", ".join(str(p) for p in missing)
                )
            print(
                "[node 0] waiting for other nodes: "
                + ", ".join(p.name for p in missing),
                flush=True,
            )
            time.sleep(10)

        build_manifest(out_root, total_gpus, args, datasets)


if __name__ == "__main__":
    mp.freeze_support()
    main()
