#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


EXPECTED_TOTAL_EPISODES = 60_246
EXPECTED_TOTAL_FRAMES = 116_584_930

DEFAULT_MANIFEST = (
    "/mnt/data/laixianxin/LAPA/data/egoverse_vqgan_tokens/"
    "canonical_vqgan_manifest.jsonl"
)

DEFAULT_LAQ = (
    "/mnt/data/laixianxin/LAPA/data/egoverse_laq_labels/"
    "latent_action_egoverse_train.jsonl"
)

FIELDS = "[instruction],[vision],delta"


def make_prompt(raw_instruction):
    raw_instruction = str(raw_instruction).strip()

    return (
        "<s> You are a helpful assistant. "
        "USER: What action should the robot take to "
        f"`{raw_instruction}` "
        "ASSISTANT:"
    )


def load_manifest(path):
    items = []

    with open(path, "r") as f:
        for line in f:
            if line.strip():
                x = json.loads(line)

                items.append({
                    "dataset": x["dataset"],
                    "split": x["split"],
                    "split_episode_index": int(x["split_episode_index"]),
                    "source_episode_id": x["source_episode_id"],
                    "expected_frames": int(x["expected_frames"]),
                    "vqgan_path": x["vqgan_path"],
                    "source": x.get("source", ""),
                })

    return items


def load_one_vq_episode(m):
    """
    IMPORTANT:
    We DO NOT trust row["split_episode_index"].

    Canonical episode identity comes from canonical_vqgan_manifest.jsonl.

    VQ file's source_episode_id + frame_index are checked against
    the canonical manifest.
    """

    dataset = m["dataset"]
    canonical_ep = int(m["split_episode_index"])
    source_id = m["source_episode_id"]
    expected_frames = int(m["expected_frames"])
    path = m["vqgan_path"]

    frame_map = {}

    n = 0

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            row_source = x["source_episode_id"]
            fi = int(x["frame_index"])
            vision = x["vision"]

            if row_source != source_id:
                raise RuntimeError(
                    f"SOURCE ID MISMATCH\n"
                    f"path={path}\n"
                    f"canonical={source_id}\n"
                    f"stored={row_source}"
                )

            if fi != n:
                raise RuntimeError(
                    f"FRAME ORDER ERROR path={path} "
                    f"expected={n} got={fi}"
                )

            if len(vision) != 256:
                raise RuntimeError(
                    f"VISION LENGTH ERROR path={path} "
                    f"frame={fi} len={len(vision)}"
                )

            # Critical fix:
            # use canonical manifest episode index, NOT stored VQ index.
            key = (dataset, canonical_ep, fi)

            frame_map[key] = vision

            n += 1

    if n != expected_frames:
        raise RuntimeError(
            f"FRAME COUNT ERROR path={path} "
            f"expected={expected_frames} got={n}"
        )

    return frame_map, n


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
    )

    p.add_argument(
        "--laq",
        default=DEFAULT_LAQ,
    )

    p.add_argument(
        "--output",
        required=True,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=32,
    )

    p.add_argument(
        "--episode-batch",
        type=int,
        default=500,
    )

    p.add_argument(
        "--max-vq-episodes",
        type=int,
        default=0,
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
    )

    args = p.parse_args()

    print("=" * 100)
    print("EgoVerse -> LAPA CANONICAL JSONL BUILDER")
    print("workers       =", args.workers)
    print("episode batch =", args.episode_batch)
    print("manifest      =", args.manifest)
    print("laq           =", args.laq)
    print("output        =", args.output)
    print("=" * 100)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    manifest = load_manifest(args.manifest)

    full_run = args.max_vq_episodes <= 0

    if full_run:
        if len(manifest) != EXPECTED_TOTAL_EPISODES:
            raise RuntimeError(
                f"manifest episode count mismatch: "
                f"{len(manifest)} != {EXPECTED_TOTAL_EPISODES}"
            )
    else:
        manifest = manifest[:args.max_vq_episodes]

    expected_frames = sum(x["expected_frames"] for x in manifest)

    print("episodes =", len(manifest))
    print("expected VQ frames =", expected_frames)

    # Canonical small episode identity map.
    episode_source = {}

    for m in manifest:
        ep_key = (
            m["dataset"],
            int(m["split_episode_index"]),
        )

        if ep_key in episode_source:
            raise RuntimeError(
                f"duplicate canonical episode key: {ep_key}"
            )

        episode_source[ep_key] = m["source_episode_id"]

    # ------------------------------------------------------------------
    # Load VQGAN tokens
    # ------------------------------------------------------------------

    vision_map = {}

    loaded_episodes = 0
    loaded_frames = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:

        for batch_start in range(
            0,
            len(manifest),
            args.episode_batch,
        ):
            batch = manifest[
                batch_start:
                batch_start + args.episode_batch
            ]

            futures = [
                ex.submit(load_one_vq_episode, m)
                for m in batch
            ]

            for fut in as_completed(futures):
                frame_map, n = fut.result()

                before = len(vision_map)

                vision_map.update(frame_map)

                after = len(vision_map)

                # With canonical keys, there must NEVER be overwrite.
                if after - before != n:
                    raise RuntimeError(
                        "CANONICAL VQ KEY COLLISION DETECTED: "
                        f"received={n}, unique_added={after-before}"
                    )

                loaded_episodes += 1
                loaded_frames += n

            print(
                f"loaded episodes: "
                f"{loaded_episodes} / {len(manifest)} "
                f"frames: {loaded_frames}",
                flush=True,
            )

    print()
    print("VQ loaded frames =", len(vision_map))
    print("expected frames  =", expected_frames)

    if loaded_frames != expected_frames:
        raise RuntimeError(
            f"loaded frame count mismatch: "
            f"{loaded_frames} != {expected_frames}"
        )

    if len(vision_map) != expected_frames:
        raise RuntimeError(
            f"UNIQUE VQ KEY COUNT MISMATCH: "
            f"{len(vision_map)} != {expected_frames}"
        )

    if full_run and len(vision_map) != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError(
            f"FULL VQ COUNT MISMATCH: "
            f"{len(vision_map)} != {EXPECTED_TOTAL_FRAMES}"
        )

    print("VQ CANONICAL LOAD CHECK: PASS", flush=True)

    # ------------------------------------------------------------------
    # Join LAQ
    # ------------------------------------------------------------------

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tmp_output = Path(str(output) + ".tmp")

    written = 0
    miss = 0
    laq_seen = 0

    with open(args.laq, "r") as fi, \
         open(tmp_output, "w") as fo:

        for line in fi:
            if not line.strip():
                continue

            x = json.loads(line)
            laq_seen += 1

            dataset = x["dataset"].split("/")[0]
            ep = int(x["split_episode_index"])
            frame = int(x["frame_index"])

            episode_key = (dataset, ep)

            canonical_source = episode_source.get(episode_key)

            # In smoke mode we loaded only the first N VQ episodes.
            # Once LAQ reaches data outside that subset, stop.
            if canonical_source is None:
                if not full_run:
                    break

                raise RuntimeError(
                    f"LAQ episode not present in canonical manifest: "
                    f"{episode_key}"
                )

            if x["source_episode_id"] != canonical_source:
                raise RuntimeError(
                    "LAQ SOURCE ID MISMATCH\n"
                    f"dataset={dataset}\n"
                    f"episode={ep}\n"
                    f"canonical={canonical_source}\n"
                    f"laq={x['source_episode_id']}"
                )

            key = (dataset, ep, frame)

            vision = vision_map.get(key)

            if vision is None:
                miss += 1

                if miss <= 20:
                    print(
                        "MISS",
                        dataset,
                        ep,
                        canonical_source,
                        frame,
                        flush=True,
                    )

                continue

            delta = [int(v) for v in x["delta"]]

            if len(delta) != 4:
                raise RuntimeError(
                    f"delta length != 4: {delta}"
                )

            if min(delta) < 0 or max(delta) > 7:
                raise RuntimeError(
                    f"delta outside [0,7]: {delta}"
                )

            row = {
                "id": x["id"],
                "instruction": make_prompt(x["instruction"]),
                "vision": vision,
                "delta": delta,
                "fields": FIELDS,
            }

            fo.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            written += 1

            if written % 100_000 == 0:
                print(
                    "written",
                    written,
                    "miss",
                    miss,
                    flush=True,
                )

            if (
                args.max_samples > 0
                and written >= args.max_samples
            ):
                break

    print()
    print("=" * 100)
    print("JOIN SUMMARY")
    print("=" * 100)
    print("LAQ rows seen =", laq_seen)
    print("written       =", written)
    print("miss          =", miss)

    if miss != 0:
        raise RuntimeError(
            f"JOIN FAILED: miss={miss}"
        )

    if (
        full_run
        and args.max_samples <= 0
        and written != EXPECTED_TOTAL_FRAMES
    ):
        raise RuntimeError(
            f"FULL SAMPLE COUNT MISMATCH: "
            f"{written} != {EXPECTED_TOTAL_FRAMES}"
        )

    # Only expose final output after every assertion passes.
    os.replace(tmp_output, output)

    print()
    print("========================================")
    print("CANONICAL LAPA JSONL BUILD: PASS")
    print("========================================")
    print("samples =", written)
    print("miss    =", miss)
    print("output  =", output)


if __name__ == "__main__":
    main()
