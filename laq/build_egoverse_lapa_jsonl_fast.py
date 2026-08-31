#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing


def load_vq_episode(ep):
    path = Path(ep["vqgan_path"])

    out = {}

    with open(path) as f:
        for line in f:
            x = json.loads(line)

            key = (
                ep["dataset"],
                int(x["frame_index"])
            )

            out[key] = x["vision"]

    return out


def build_cache(args):

    with open(args.manifest) as f:
        episodes=[json.loads(x) for x in f]

    cache_dir=Path(args.cache)
    cache_dir.mkdir(parents=True,exist_ok=True)

    print("episodes =",len(episodes))

    worker=args.workers

    with ProcessPoolExecutor(worker) as ex:

        for i,result in enumerate(
            ex.map(load_vq_episode,episodes)
        ):

            p=cache_dir/f"{i:06d}.json"

            with open(p,"w") as f:
                json.dump(result,f)

            if i%500==0:
                print(
                    "cached",
                    i,
                    "/",
                    len(episodes)
                )


def load_cache(args):

    cache={}

    for p in Path(args.cache).glob("*.json"):

        with open(p) as f:
            x=json.load(f)

        cache.update(x)

    return cache



def worker_laq(lines,cache,out):

    n=0

    with open(out,"w") as fo:

        for line in lines:

            x=json.loads(line)

            key=(
                x["dataset"].split("/")[0],
                int(x["frame_index"])
            )

            vision=cache.get(key)

            if vision is None:
                continue

            y={
                "id":x["id"],
                "instruction":x["instruction"],
                "vision":vision,
                "delta":list(
                    map(int,x["delta"])
                )
            }

            fo.write(
                json.dumps(y)
                +"\n"
            )

            n+=1

    return n



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
        default=64
    )

    parser.add_argument(
        "--cache",
        default="/mnt/data/laixianxin/LAPA/data/vq_cache"
    )

    args=parser.parse_args()


    print("="*100)
    print("FAST LAPA JSONL BUILDER")
    print("workers =",args.workers)
    print("="*100)


    build_cache(args)

    print("loading cache")

    cache=load_cache(args)

    print(
        "cache frames =",
        len(cache)
    )


    with open(args.laq) as f:
        lines=f.readlines()


    chunks=[]

    n=args.workers

    size=len(lines)//n+1

    for i in range(n):

        chunks.append(
            lines[
                i*size:
                (i+1)*size
            ]
        )


    tmp=Path(args.output+".parts")
    tmp.mkdir(exist_ok=True)


    func=partial(
        worker_laq,
        cache=cache,
    )


    with ProcessPoolExecutor(n) as ex:

        futures=[]

        for i,c in enumerate(chunks):

            futures.append(
                ex.submit(
                    worker_laq,
                    c,
                    cache,
                    str(tmp/f"{i:04d}.jsonl")
                )
            )


        total=0

        for f in futures:
            total+=f.result()


    print(
        "samples =",
        total
    )


    print("merging")

    with open(args.output,"w") as out:

        for p in sorted(tmp.glob("*.jsonl")):

            with open(p) as f:
                for line in f:
                    out.write(line)


    print("DONE")
    print(args.output)



if __name__=="__main__":
    main()
