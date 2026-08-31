#!/usr/bin/env python3

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MISMATCH = Path(
    "/mnt/data/laixianxin/LAPA/data/egoverse_vqgan_tokens/"
    "repair_inventory/current_internal_split_mismatches.jsonl"
)

WORKERS = 64

with MISMATCH.open() as f:
    items = [json.loads(x) for x in f if x.strip()]

print("episodes to deep-check =", len(items))


def check(x):

    n = 0
    errors = []

    canonical_idx = x["canonical_split_episode_index"]
    stored_idx = x["stored_split_episode_index"]
    source_id = x["canonical_source_episode_id"]
    expected = x["expected_frames"]

    first_frame = None
    last_frame = None
    stored_indices = set()
    source_ids = set()

    with open(x["vqgan_path"]) as f:
        for line in f:

            r = json.loads(line)

            fi = int(r["frame_index"])

            if first_frame is None:
                first_frame = fi

            last_frame = fi

            stored_indices.add(int(r["split_episode_index"]))
            source_ids.add(r["source_episode_id"])

            if len(r["vision"]) != 256:
                errors.append(
                    f"vision len != 256 at frame {fi}"
                )
                break

            n += 1

    if n != expected:
        errors.append(
            f"row count {n} != expected {expected}"
        )

    if first_frame != 0:
        errors.append(
            f"first frame {first_frame} != 0"
        )

    if last_frame != expected - 1:
        errors.append(
            f"last frame {last_frame} != {expected - 1}"
        )

    if source_ids != {source_id}:
        errors.append(
            f"source ids mismatch: {source_ids}"
        )

    if stored_indices != {stored_idx}:
        errors.append(
            f"stored split indices vary: {stored_indices}"
        )

    return {
        "dataset": x["dataset"],
        "canonical": canonical_idx,
        "stored": stored_idx,
        "source_id": source_id,
        "rows": n,
        "expected": expected,
        "errors": errors,
    }


bad = []
total_rows = 0

with ThreadPoolExecutor(max_workers=WORKERS) as ex:

    fs = [ex.submit(check, x) for x in items]

    for i, fut in enumerate(as_completed(fs), 1):

        r = fut.result()

        total_rows += r["rows"]

        if r["errors"]:
            bad.append(r)

        if i % 100 == 0 or i == len(fs):
            print(
                f"checked {i}/{len(fs)} "
                f"rows={total_rows:,} "
                f"errors={len(bad)}",
                flush=True,
            )


print()
print("=" * 80)
print("DEEP CHECK RESULT")
print("=" * 80)
print("episodes =", len(items))
print("rows     =", total_rows)
print("errors   =", len(bad))

if not bad:
    print("PASS")
else:
    print("FAIL")
    for x in bad[:20]:
        print(x)
