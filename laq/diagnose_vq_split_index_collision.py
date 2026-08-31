#!/usr/bin/env python3

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MANIFEST = Path(
    "/mnt/data/laixianxin/LAPA/data/egoverse_vqgan_tokens/"
    "canonical_vqgan_manifest.jsonl"
)

OUT = Path(
    "/mnt/data/laixianxin/LAPA/data/egoverse_vqgan_tokens/"
    "repair_inventory/current_internal_split_mismatches.jsonl"
)

WORKERS = 64

KNOWN_BUILDER_COUNT = 114_572_528


with MANIFEST.open() as f:
    manifest = [json.loads(x) for x in f if x.strip()]

print("=" * 100)
print("VQGAN STORED split_episode_index DIAGNOSTIC")
print("manifest episodes =", len(manifest))
print("=" * 100)


def inspect(m):
    path = m["vqgan_path"]

    with open(path) as f:
        row = json.loads(f.readline())

    return {
        "dataset": m["dataset"],
        "canonical_split_episode_index": int(m["split_episode_index"]),
        "stored_split_episode_index": int(row["split_episode_index"]),
        "canonical_source_episode_id": m["source_episode_id"],
        "stored_source_episode_id": row["source_episode_id"],
        "expected_frames": int(m["expected_frames"]),
        "vqgan_path": path,
        "source": m.get("source"),
    }


results = []

with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = [ex.submit(inspect, m) for m in manifest]

    for i, fut in enumerate(as_completed(futures), 1):
        results.append(fut.result())

        if i % 5000 == 0 or i == len(futures):
            print(f"checked {i} / {len(futures)}", flush=True)


wrong_idx = [
    x for x in results
    if x["canonical_split_episode_index"]
    != x["stored_split_episode_index"]
]

wrong_source = [
    x for x in results
    if x["canonical_source_episode_id"]
    != x["stored_source_episode_id"]
]


OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w") as f:
    for x in sorted(
        wrong_idx,
        key=lambda z: (
            z["dataset"],
            z["canonical_split_episode_index"]
        )
    ):
        f.write(json.dumps(x) + "\n")


print()
print("=" * 100)
print("IDENTITY CHECK")
print("=" * 100)

print("episodes total                    =", len(results))
print("wrong stored split_episode_index  =", len(wrong_idx))
print("wrong source_episode_id           =", len(wrong_source))
print("frames in wrong-index episodes    =", sum(x["expected_frames"] for x in wrong_idx))


# ---------------------------------------------------------------------
# 模拟旧 builder:
#
# key = (dataset, STORED split_episode_index, frame_index)
#
# 一个 episode 的 frame_index 是:
# 0 ... expected_frames-1
#
# 如果两个 episode 映射到了相同 stored episode index，
# Python dict 最终只保留这些 frame key 的并集。
#
# 因为都从 frame 0 开始，所以一个 collision group 的
# unique key 数 = max(frame_count)
# ---------------------------------------------------------------------

groups = defaultdict(list)

for x in results:
    key = (
        x["dataset"],
        x["stored_split_episode_index"],
    )
    groups[key].append(x)


expected_total = sum(x["expected_frames"] for x in results)

predicted_unique_keys = 0
collision_groups = []

for key, xs in groups.items():
    max_frames = max(x["expected_frames"] for x in xs)

    predicted_unique_keys += max_frames

    if len(xs) > 1:
        collision_groups.append((key, xs))


predicted_lost = expected_total - predicted_unique_keys


print()
print("=" * 100)
print("OLD BUILDER KEY COLLISION SIMULATION")
print("=" * 100)

print("expected total frames       =", expected_total)
print("stored episode-key groups   =", len(groups))
print("collision groups            =", len(collision_groups))
print("predicted dict unique keys  =", predicted_unique_keys)
print("predicted overwritten keys  =", predicted_lost)

print()
print("actual builder reported     =", KNOWN_BUILDER_COUNT)
print("actual builder lost         =", expected_total - KNOWN_BUILDER_COUNT)

print()

if predicted_unique_keys == KNOWN_BUILDER_COUNT:
    print("============================================================")
    print("EXACT MATCH: split_episode_index collision explains builder")
    print("============================================================")
else:
    print("============================================================")
    print("NOT EXACT MATCH: additional issue exists")
    print("difference =",
          predicted_unique_keys - KNOWN_BUILDER_COUNT)
    print("============================================================")


print()
print("First 20 wrong-index episodes:")
for x in sorted(
    wrong_idx,
    key=lambda z: (
        z["dataset"],
        z["canonical_split_episode_index"]
    )
)[:20]:

    print(
        x["dataset"],
        "canonical=", x["canonical_split_episode_index"],
        "stored=", x["stored_split_episode_index"],
        "frames=", x["expected_frames"],
        "source_id=", x["canonical_source_episode_id"]
    )


print()
print("First 20 collision groups:")

for key, xs in sorted(collision_groups)[:20]:

    print()
    print("KEY", key)

    for x in xs:
        print(
            "   canonical=",
            x["canonical_split_episode_index"],
            "stored=",
            x["stored_split_episode_index"],
            "frames=",
            x["expected_frames"],
            "source=",
            x["canonical_source_episode_id"]
        )


print()
print("mismatch manifest =", OUT)
