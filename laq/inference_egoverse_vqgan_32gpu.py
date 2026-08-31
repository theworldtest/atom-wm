#!/usr/bin/env python3
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


def parse_visible_devices(num_gpus):
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        toks = [x.strip() for x in raw.split(",") if x.strip()]
        if len(toks) < num_gpus:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES has {len(toks)} entries, "
                f"but --num-gpus={num_gpus}: {raw}"
            )
        return toks[:num_gpus]
    return [str(i) for i in range(num_gpus)]


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


def worker_main(rank, device_token, args_dict, datasets):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_token)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(args_dict["cpu_threads_per_worker"])

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

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError(f"[rank {rank}] expected one GPU, got {jax.devices()}")

    print(f"[rank {rank:02d}] device={device_token} jax_devices={jax.devices()}", flush=True)

    preprocessor = A.Compose([
        A.LongestMaxSize(max_size=256),
        A.Resize(256, 256),
    ])

    t0 = time.perf_counter()
    vqgan = VQGAN(args_dict["vqgan_checkpoint"], replicate=False)
    print(f"[rank {rank:02d}] VQGAN loaded in {time.perf_counter()-t0:.2f}s", flush=True)

    warm = np.zeros((args_dict["batch_size"], 256, 256, 3), dtype=np.float32)
    warm_ids = jax.device_get(vqgan.encode(warm))[1]
    if warm_ids.ndim != 3 or tuple(warm_ids.shape[1:]) != (16, 16):
        raise RuntimeError(f"[rank {rank}] unexpected warmup shape {warm_ids.shape}")
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

        start = rank * n_episodes // args_dict["num_gpus"]
        end = (rank + 1) * n_episodes // args_dict["num_gpus"]

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
            f"[rank {rank:02d}] {group}: total={n_episodes} "
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
                read_config=tfds.ReadConfig(skip_prefetch=True, try_autocache=False),
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

                tmp_path = group_dir / f".ep_{ep_idx:06d}.rank{rank:02d}.pid{os.getpid()}.tmp"

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

                    if n_real < args_dict["batch_size"]:
                        pad = args_dict["batch_size"] - n_real
                        imgs = np.concatenate(
                            [imgs, np.repeat(imgs[-1:], pad, axis=0)], axis=0
                        )

                    ids = jax.device_get(vqgan.encode(imgs))[1].astype(np.int32)[:n_real]

                    if ids.ndim != 3 or tuple(ids.shape[1:]) != (16, 16):
                        raise RuntimeError(f"{group} ep={ep_idx}: unexpected shape {ids.shape}")
                    if int(ids.min()) < 0 or int(ids.max()) >= 8192:
                        raise RuntimeError(
                            f"{group} ep={ep_idx}: ids out of range "
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
                    f"[rank {rank:02d}] DONE {group} ep={ep_idx} "
                    f"frames={ep_frames} sec={elapsed:.2f} "
                    f"fps={(ep_frames/elapsed if elapsed else 0):.1f}",
                    flush=True,
                )

    elapsed = time.perf_counter() - rank_start
    stats = {
        "rank": rank,
        "device_token": str(device_token),
        "episodes_written": total_episodes,
        "episodes_skipped": total_skipped,
        "frames_written": total_frames,
        "elapsed_sec": elapsed,
        "frames_per_sec": total_frames / elapsed if elapsed else 0.0,
    }
    with open(out_root / f"rank_{rank:02d}.stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[rank {rank:02d}] COMPLETE {json.dumps(stats)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/mnt/data/RLDS/EgoVerse_full")
    ap.add_argument("--output-root", default="/mnt/data/laixianxin/LAPA/data/egoverse_vqgan_tokens")
    ap.add_argument("--vqgan-checkpoint", default="/mnt/data/laixianxin/LAPA/lapa_checkpoints/vqgan")
    ap.add_argument("--num-gpus", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dataset-filter", default=None)
    ap.add_argument("--max-episodes-per-rank", type=int, default=None)
    ap.add_argument("--cpu-threads-per-worker", type=int, default=1)
    ap.add_argument("--tf-intra-threads", type=int, default=2)
    ap.add_argument("--tf-inter-threads", type=int, default=1)
    ap.add_argument("--opencv-threads", type=int, default=1)
    args = ap.parse_args()

    if not Path(args.vqgan_checkpoint).exists():
        raise FileNotFoundError(args.vqgan_checkpoint)

    datasets = discover_dataset_dirs(args.data_root, args.dataset_filter)
    print("Discovered datasets:")
    for group, d in datasets:
        print(f"  {group}: {d}")

    visible = parse_visible_devices(args.num_gpus)
    print(f"Launching {args.num_gpus} independent VQGAN workers")
    print(f"Visible device tokens: {visible}")
    print(f"Output: {args.output_root}")

    args_dict = vars(args).copy()
    ctx = mp.get_context("spawn")
    procs = []

    for rank in range(args.num_gpus):
        p = ctx.Process(target=worker_main, args=(rank, visible[rank], args_dict, datasets))
        p.start()
        procs.append(p)

    failed = []
    for rank, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            failed.append((rank, p.exitcode))

    if failed:
        raise RuntimeError(f"Workers failed: {failed}")

    out_root = Path(args.output_root)
    stats = []
    for rank in range(args.num_gpus):
        p = out_root / f"rank_{rank:02d}.stats.json"
        if p.exists():
            with open(p) as f:
                stats.append(json.load(f))

    summary = {
        "num_gpus": args.num_gpus,
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
        "output_root": args.output_root,
    }
    with open(out_root / "manifest_run.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("EgoVerse VQGAN EXTRACTION COMPLETE")
    print(json.dumps(summary, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    mp.freeze_support()
    main()
