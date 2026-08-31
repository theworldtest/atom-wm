#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def build_vq_index(manifest):

    index = {}

    with open(manifest) as f:
        for line in f:
            x = json.loads(line)

            key = (
                x["dataset"],
                int(x["split_episode_index"])
            )

            index[key] = x["vqgan_path"]

    return index



def load_episode_tokens(path):

    frames = {}

    with open(path) as f:
        for line in f:

            x = json.loads(line)

            frames[
                int(x["frame_index"])
            ] = x["vision"]

    return frames



def process_chunk(args):

    (
        lines,
        vq_index,
        output
    ) = args


    episode_cache = {}

    count = 0
    miss = 0


    with open(output,"w") as fo:

        for line in lines:

            x=json.loads(line)


            dataset = x["dataset"].split("/")[0]

            ep = int(
                x["split_episode_index"]
            )

            key=(dataset,ep)


            if key not in episode_cache:

                path=vq_index.get(key)

                if path is None:
                    miss += 1
                    continue

                episode_cache[key]=load_episode_tokens(path)


            vision = episode_cache[key].get(
                int(x["frame_index"])
            )


            if vision is None:
                miss += 1
                continue


            y={
                "id":x["id"],
                "instruction":x["instruction"],
                "vision":vision,
                "delta":[int(i) for i in x["delta"]]
            }


            fo.write(
                json.dumps(y,separators=(",",":"))
                + "\n"
            )

            count+=1


    return count,miss



def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--manifest",
        default="/mnt/data/laixianxin/LAPA/data/egoverse_vqgan_tokens/canonical_vqgan_manifest.jsonl"
    )

    parser.add_argument(
        "--laq",
        default="/mnt/data/laixianxin/LAPA/data/egoverse_laq_labels/latent_action_egoverse_train.jsonl"
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=96
    )


    parser.add_argument(
        "--max-lines",
        type=int,
        default=0
    )


    args=parser.parse_args()


    print("="*100)
    print("LAPA LAZY JOIN BUILDER")
    print("workers =",args.workers)
    print("="*100)



    print("building VQ index")

    vq_index=build_vq_index(
        args.manifest
    )

    print(
        "vq episodes =",
        len(vq_index)
    )



    print("loading LAQ")

    with open(args.laq) as f:
        lines=f.readlines(
            args.max_lines
            if args.max_lines>0 else -1
        )


    print(
        "LAQ lines =",
        len(lines)
    )


    size=len(lines)//args.workers+1


    chunks=[]

    tmp=Path(
        args.output+".parts"
    )

    tmp.mkdir(
        exist_ok=True
    )


    for i in range(args.workers):

        c=lines[
            i*size:
            (i+1)*size
        ]

        if c:
            chunks.append(
                (
                    c,
                    vq_index,
                    str(tmp/f"{i:05d}.jsonl")
                )
            )


    print(
        "chunks =",
        len(chunks)
    )



    total=0
    miss=0


    with ProcessPoolExecutor(
        args.workers
    ) as ex:

        for i,(n,m) in enumerate(
            ex.map(
                process_chunk,
                chunks
            )
        ):

            total+=n
            miss+=m

            print(
                "finished",
                i+1,
                "/",
                len(chunks),
                "samples=",
                total,
                "miss=",
                miss
            )


    print("merging")

    with open(args.output,"w") as out:

        for p in sorted(
            tmp.glob("*.jsonl")
        ):

            with open(p) as f:

                for line in f:
                    out.write(line)



    print("="*100)
    print("DONE")
    print("samples =",total)
    print("miss =",miss)
    print("output =",args.output)
    print("="*100)



if __name__=="__main__":
    main()
