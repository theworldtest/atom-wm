#!/usr/bin/env python3

import os
import json
import time
import argparse
from pathlib import Path


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


def discover_dataset_dirs(data_root):
    root = Path(data_root)

    out = {}

    for p in sorted(root.glob("*/*/*/dataset_info.json")):
        dataset_dir = p.parent
        rel = dataset_dir.relative_to(root)
        group = rel.parts[0]

        if group in out:
            raise RuntimeError(
                f"More than one TFDS dataset found for {group}: "
                f"{out[group]} and {dataset_dir}"
            )

        out[group] = str(dataset_dir)

    if not out:
        raise RuntimeError(
            f"No TFDS datasets found under {data_root}"
        )

    return out


def load_missing_manifest(path):
    groups = {}

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            group = str(x["dataset"])
            idx = int(x["split_episode_index"])
            sid = str(x["source_episode_id"])
            expected_frames = int(x["expected_frames"])

            groups.setdefault(group, []).append(
                {
                    "dataset": group,
                    "split_episode_index": idx,
                    "source_episode_id": sid,
                    "expected_frames": expected_frames,
                }
            )

    for group in groups:
        groups[group].sort(
            key=lambda x: x["split_episode_index"]
        )

        ordinals = [
            x["split_episode_index"]
            for x in groups[group]
        ]

        if len(ordinals) != len(set(ordinals)):
            raise RuntimeError(
                f"Duplicate ordinal in missing manifest: {group}"
            )

        sids = [
            x["source_episode_id"]
            for x in groups[group]
        ]

        if len(sids) != len(set(sids)):
            raise RuntimeError(
                f"Duplicate source ID in missing manifest: {group}"
            )

    return groups


def compress_ranges(records):
    """
    Input records sorted by split_episode_index.

    Returns:
      [(start, end, [records...]), ...]
    where end is exclusive.
    """
    if not records:
        return []

    out = []

    start = records[0]["split_episode_index"]
    prev = start
    buf = [records[0]]

    for rec in records[1:]:
        idx = rec["split_episode_index"]

        if idx == prev + 1:
            buf.append(rec)
            prev = idx
            continue

        out.append(
            (start, prev + 1, buf)
        )

        start = idx
        prev = idx
        buf = [rec]

    out.append(
        (start, prev + 1, buf)
    )

    return out


def build_split_chunks(records, max_records_per_chunk):
    """
    Build TFDS union split expressions such as:

      train[82:83]+train[110:111]+train[149:151]

    Each chunk contains at most max_records_per_chunk episodes.
    """
    ranges = compress_ranges(records)

    chunks = []
    current_records = []
    current_parts = []

    for start, end, recs in ranges:
        # If adding this entire contiguous range would exceed the
        # chunk limit, flush the current chunk first.
        if (
            current_records
            and len(current_records) + len(recs)
            > max_records_per_chunk
        ):
            chunks.append(
                (
                    "+".join(current_parts),
                    current_records,
                )
            )
            current_records = []
            current_parts = []

        # Extremely long contiguous ranges are split into pieces.
        pos = 0

        while pos < len(recs):
            remaining = (
                max_records_per_chunk
                - len(current_records)
            )

            take = min(
                remaining,
                len(recs) - pos,
            )

            piece = recs[pos:pos + take]

            pstart = piece[0]["split_episode_index"]
            pend = piece[-1]["split_episode_index"] + 1

            current_parts.append(
                f"train[{pstart}:{pend}]"
            )
            current_records.extend(piece)

            pos += take

            if len(current_records) == max_records_per_chunk:
                chunks.append(
                    (
                        "+".join(current_parts),
                        current_records,
                    )
                )
                current_records = []
                current_parts = []

    if current_records:
        chunks.append(
            (
                "+".join(current_parts),
                current_records,
            )
        )

    return chunks


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-root",
        default="/mnt/data/RLDS/EgoVerse_full",
    )

    ap.add_argument(
        "--missing-manifest",
        default=(
            "/mnt/data/laixianxin/LAPA/data/"
            "egoverse_vqgan_tokens/"
            "repair_inventory/"
            "missing_vqgan_episodes.jsonl"
        ),
    )

    ap.add_argument(
        "--output-root",
        default=(
            "/mnt/data/laixianxin/LAPA/data/"
            "egoverse_vqgan_tokens_repair_missing"
        ),
    )

    ap.add_argument(
        "--vqgan-checkpoint",
        default=(
            "/mnt/data/laixianxin/LAPA/"
            "lapa_checkpoints/vqgan"
        ),
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--episodes-per-read-chunk",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--tf-intra-threads",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--tf-inter-threads",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
    )

    args = ap.parse_args()

    # Must happen before TF/JAX imports.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Use validated pip CUDA/cuDNN stack.
    os.environ.pop("LD_LIBRARY_PATH", None)

    import numpy as np
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(
            args.tf_intra_threads
        )
        tf.config.threading.set_inter_op_parallelism_threads(
            args.tf_inter_threads
        )
    except Exception:
        pass

    import tensorflow_datasets as tfds
    import albumentations as A
    import cv2

    try:
        cv2.setNumThreads(args.opencv_threads)
    except Exception:
        pass

    import jax
    from latent_pretraining.vqgan import VQGAN

    devices = jax.devices()

    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(
            f"Expected exactly one JAX GPU, got {devices}"
        )

    print("=" * 100)
    print("EgoVerse VQGAN missing-episode repair")
    print("JAX devices =", devices)
    print("=" * 100)

    manifest_path = Path(args.missing_manifest)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    if not Path(args.vqgan_checkpoint).exists():
        raise FileNotFoundError(args.vqgan_checkpoint)

    missing_by_group = load_missing_manifest(
        manifest_path
    )

    dataset_dirs = discover_dataset_dirs(
        args.data_root
    )

    total_target_eps = sum(
        len(v)
        for v in missing_by_group.values()
    )

    total_target_frames = sum(
        x["expected_frames"]
        for v in missing_by_group.values()
        for x in v
    )

    print(
        "target episodes =",
        f"{total_target_eps:,}",
    )
    print(
        "target frames   =",
        f"{total_target_frames:,}",
    )

    for group, records in missing_by_group.items():
        print(
            f"{group:55s} "
            f"episodes={len(records):5d} "
            f"frames="
            f"{sum(x['expected_frames'] for x in records):,}"
        )

    if total_target_eps != 3303:
        print(
            "WARNING: target episode count is not 3303. "
            "Continuing because manifest is authoritative."
        )

    preprocessor = A.Compose([
        A.LongestMaxSize(max_size=256),
        A.Resize(256, 256),
    ])

    print()
    print("Loading VQGAN ...")

    t0 = time.perf_counter()

    vqgan = VQGAN(
        args.vqgan_checkpoint,
        replicate=False,
    )

    print(
        f"VQGAN loaded in "
        f"{time.perf_counter()-t0:.2f}s"
    )

    # Compile fixed shape once.
    warm = np.zeros(
        (
            args.batch_size,
            256,
            256,
            3,
        ),
        dtype=np.float32,
    )

    warm_ids = jax.device_get(
        vqgan.encode(warm)
    )[1]

    if (
        warm_ids.ndim != 3
        or tuple(warm_ids.shape[1:]) != (16, 16)
    ):
        raise RuntimeError(
            f"Unexpected warmup shape: {warm_ids.shape}"
        )

    del warm, warm_ids

    output_root = Path(args.output_root)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    written_eps = 0
    skipped_eps = 0
    written_frames = 0

    run_start = time.perf_counter()

    for group, records in missing_by_group.items():

        if group not in dataset_dirs:
            raise RuntimeError(
                f"No TFDS dataset found for {group}"
            )

        dataset_dir = dataset_dirs[group]

        builder = tfds.builder_from_directory(
            dataset_dir
        )

        n_episodes = int(
            builder.info.splits["train"].num_examples
        )

        for rec in records:
            idx = rec["split_episode_index"]

            if idx < 0 or idx >= n_episodes:
                raise RuntimeError(
                    f"{group}: target ordinal {idx} "
                    f"outside [0,{n_episodes})"
                )

        group_dir = (
            output_root
            / group
            / "episodes"
        )

        group_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pending = []

        for rec in records:
            idx = rec["split_episode_index"]

            final_path = (
                group_dir
                / f"ep_{idx:06d}.jsonl"
            )

            if final_path.exists():
                with open(final_path) as f:
                    line = f.readline()

                if not line:
                    raise RuntimeError(
                        f"Existing repair file is empty: "
                        f"{final_path}"
                    )

                old = json.loads(line)

                if (
                    str(old["source_episode_id"])
                    != rec["source_episode_id"]
                ):
                    raise RuntimeError(
                        "Existing repair output identity "
                        "mismatch:\n"
                        f"path={final_path}\n"
                        f"expected source="
                        f"{rec['source_episode_id']}\n"
                        f"stored source="
                        f"{old['source_episode_id']}"
                    )

                skipped_eps += 1
            else:
                pending.append(rec)

        print()
        print("=" * 100)
        print(
            f"{group}: targets={len(records):,} "
            f"already_repaired={len(records)-len(pending):,} "
            f"pending={len(pending):,}"
        )
        print("=" * 100)

        if not pending:
            continue

        chunks = build_split_chunks(
            pending,
            args.episodes_per_read_chunk,
        )

        print(
            f"{group}: TFDS read chunks={len(chunks)}"
        )

        for chunk_i, (split_expr, chunk_records) in enumerate(
            chunks,
            1,
        ):
            print(
                f"[{group}] chunk "
                f"{chunk_i}/{len(chunks)} "
                f"episodes={len(chunk_records)}",
                flush=True,
            )

            ds = builder.as_dataset(
                split=split_expr,
                shuffle_files=False,
                read_config=tfds.ReadConfig(
                    skip_prefetch=True,
                    try_autocache=False,
                ),
            )

            yielded = 0

            for j, episode in enumerate(ds):

                if j >= len(chunk_records):
                    raise RuntimeError(
                        f"{group}: TFDS yielded more "
                        f"episodes than expected for "
                        f"{split_expr}"
                    )

                target = chunk_records[j]

                canonical_idx = int(
                    target["split_episode_index"]
                )

                expected_sid = str(
                    target["source_episode_id"]
                )

                expected_frames = int(
                    target["expected_frames"]
                )

                metadata = episode.get(
                    "episode_metadata",
                    {},
                )

                if (
                    not isinstance(metadata, dict)
                    or "source_episode_id" not in metadata
                ):
                    raise RuntimeError(
                        f"{group} ordinal={canonical_idx}: "
                        f"missing source_episode_id "
                        f"in TFDS metadata"
                    )

                actual_sid = as_str(
                    metadata["source_episode_id"]
                )

                # THIS is the key repair safety check.
                if actual_sid != expected_sid:
                    raise RuntimeError(
                        "\nSOURCE ID MISMATCH - ABORTING\n"
                        f"dataset={group}\n"
                        f"canonical ordinal="
                        f"{canonical_idx}\n"
                        f"expected source="
                        f"{expected_sid}\n"
                        f"TFDS source="
                        f"{actual_sid}\n"
                        f"split={split_expr}\n"
                    )

                final_path = (
                    group_dir
                    / f"ep_{canonical_idx:06d}.jsonl"
                )

                tmp_path = (
                    group_dir
                    / (
                        f".ep_{canonical_idx:06d}."
                        f"pid{os.getpid()}.tmp"
                    )
                )

                ep_start = time.perf_counter()

                batch_images = []
                batch_indices = []
                ep_frames = 0

                def flush_batch(fout):
                    nonlocal batch_images
                    nonlocal batch_indices
                    nonlocal ep_frames

                    if not batch_images:
                        return

                    n_real = len(batch_images)

                    imgs = np.asarray([
                        preprocessor(
                            image=img.astype(np.uint8)
                        )["image"]
                        for img in batch_images
                    ])

                    imgs = (
                        imgs / 127.5 - 1.0
                    ).astype(np.float32)

                    if n_real < args.batch_size:
                        pad = (
                            args.batch_size - n_real
                        )

                        imgs = np.concatenate(
                            [
                                imgs,
                                np.repeat(
                                    imgs[-1:],
                                    pad,
                                    axis=0,
                                ),
                            ],
                            axis=0,
                        )

                    ids = jax.device_get(
                        vqgan.encode(imgs)
                    )[1]

                    ids = (
                        ids.astype(np.int32)[:n_real]
                    )

                    if (
                        ids.ndim != 3
                        or tuple(ids.shape[1:])
                        != (16, 16)
                    ):
                        raise RuntimeError(
                            f"{group} "
                            f"ordinal={canonical_idx}: "
                            f"bad VQ shape {ids.shape}"
                        )

                    if (
                        int(ids.min()) < 0
                        or int(ids.max()) >= 8192
                    ):
                        raise RuntimeError(
                            f"{group} "
                            f"ordinal={canonical_idx}: "
                            f"VQ range "
                            f"{int(ids.min())}.."
                            f"{int(ids.max())}"
                        )

                    for frame_index, code in zip(
                        batch_indices,
                        ids,
                    ):
                        out = {
                            "dataset": group,
                            "split": "train",

                            # ALWAYS canonical TFDS ordinal.
                            "split_episode_index":
                                canonical_idx,

                            "source_episode_id":
                                expected_sid,

                            "frame_index":
                                int(frame_index),

                            "vision":
                                code.reshape(-1).tolist(),
                        }

                        fout.write(
                            json.dumps(
                                out,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )

                        ep_frames += 1

                    batch_images = []
                    batch_indices = []

                try:
                    with open(
                        tmp_path,
                        "w",
                        buffering=1024 * 1024,
                    ) as fout:

                        for default_frame_idx, step in enumerate(
                            episode["steps"]
                        ):
                            if "frame_index" in step:
                                frame_idx = int(
                                    as_py_scalar(
                                        step["frame_index"]
                                    )
                                )
                            else:
                                frame_idx = (
                                    default_frame_idx
                                )

                            img = (
                                step["observation"]
                                ["images"]
                                ["front_1"]
                                .numpy()
                            )

                            batch_images.append(img)
                            batch_indices.append(
                                frame_idx
                            )

                            if (
                                len(batch_images)
                                == args.batch_size
                            ):
                                flush_batch(fout)

                        flush_batch(fout)

                    if ep_frames != expected_frames:
                        raise RuntimeError(
                            f"{group} "
                            f"ordinal={canonical_idx}: "
                            f"frame count mismatch: "
                            f"VQ={ep_frames} "
                            f"LAQ={expected_frames}"
                        )

                    os.replace(
                        tmp_path,
                        final_path,
                    )

                except Exception:
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass
                    raise

                written_eps += 1
                written_frames += ep_frames
                yielded += 1

                elapsed = (
                    time.perf_counter()
                    - ep_start
                )

                total_elapsed = (
                    time.perf_counter()
                    - run_start
                )

                processed = (
                    written_eps + skipped_eps
                )

                fps = (
                    written_frames / total_elapsed
                    if total_elapsed > 0
                    else 0
                )

                remain_frames = max(
                    0,
                    total_target_frames
                    - written_frames,
                )

                eta_h = (
                    remain_frames / fps / 3600
                    if fps > 0
                    else 0
                )

                print(
                    f"DONE {group} "
                    f"ordinal={canonical_idx} "
                    f"frames={ep_frames} "
                    f"sec={elapsed:.2f} "
                    f"fps="
                    f"{ep_frames/elapsed:.1f} "
                    f"| repair "
                    f"{processed}/{total_target_eps} "
                    f"| ETA~{eta_h:.2f}h",
                    flush=True,
                )

            if yielded != len(chunk_records):
                raise RuntimeError(
                    f"{group}: split {split_expr} "
                    f"yielded {yielded}, "
                    f"expected {len(chunk_records)}"
                )

    # Final existence + identity check.
    missing_after = []

    for group, records in missing_by_group.items():
        for rec in records:

            idx = rec["split_episode_index"]

            p = (
                output_root
                / group
                / "episodes"
                / f"ep_{idx:06d}.jsonl"
            )

            if not p.exists():
                missing_after.append(
                    str(p)
                )
                continue

            with open(p) as f:
                first = json.loads(
                    f.readline()
                )

            if (
                str(first["source_episode_id"])
                != rec["source_episode_id"]
            ):
                raise RuntimeError(
                    f"Final identity mismatch: {p}"
                )

    if missing_after:
        raise RuntimeError(
            "Repair incomplete. Missing outputs:\n"
            + "\n".join(
                missing_after[:100]
            )
        )

    elapsed = (
        time.perf_counter()
        - run_start
    )

    summary = {
        "status": "PASS",
        "target_episodes":
            total_target_eps,
        "target_frames":
            total_target_frames,
        "episodes_written_this_run":
            written_eps,
        "episodes_skipped_existing_repair":
            skipped_eps,
        "frames_written_this_run":
            written_frames,
        "elapsed_sec":
            elapsed,
        "output_root":
            str(output_root),
        "missing_manifest":
            str(manifest_path),
        "vision_tokens_per_frame":
            256,
        "vq_vocab_size":
            8192,
    }

    summary_path = (
        output_root
        / "repair_run_summary.json"
    )

    tmp_summary = (
        output_root
        / ".repair_run_summary.json.tmp"
    )

    with open(tmp_summary, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    os.replace(
        tmp_summary,
        summary_path,
    )

    print()
    print("=" * 100)
    print("MISSING VQGAN REPAIR COMPLETE")
    print("=" * 100)
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
