#!/usr/bin/env python3

import json
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def build_vq_index(path):

    idx={}

    with open(path) as f:
        for line in f:
            x=json.loads(line)

            key=(
                x["dataset"],
                int(x["split_episode_index"])
            )

            idx[key]=x["vqgan_path"]

    return idx



def load_episode(path):

    frames={}

    with open(path) as f:

        for line in f:

            x=json.loads(line)

            frames[int(x["frame_index"])]=x["vision"]

    return frames



def worker(args):

    (
        worker_id,
        start,
        end,
        laq_path,
        vq_index,
        out
    )=args


    episode_cache={}

    count=0
    miss=0


    with open(laq_path) as f:

        f.seek(start)

        # 对齐行首
        if start!=0:
            f.readline()


        while f.tell()<end:

            line=f.readline()

            if not line:
                break


            x=json.loads(line)


            dataset=x["dataset"].split("/")[0]

            ep=int(
                x["split_episode_index"]
            )

            key=(dataset,ep)


            if key not in episode_cache:

                vq=vq_index.get(key)

                if vq is None:
                    miss+=1
                    continue

                episode_cache[key]=load_episode(vq)



            vision=episode_cache[key].get(
                int(x["frame_index"])
            )


            if vision is None:
                miss+=1
                continue


            y={
                "id":x["id"],
                "instruction":x["instruction"],
                "vision":vision,
                "delta":[int(i) for i in x["delta"]]
            }


            with open(out,"a") as fo:
                fo.write(
                    json.dumps(
                        y,
                        separators=(",",":")
                    )
                    +"\n"
                )


            count+=1


    return worker_id,count,miss



def main():

    p=argparse.ArgumentParser()


    p.add_argument(
        "--workers",
        type=int,
        default=160
    )

    p.add_argument(
        "--laq",
        default="/mnt/data/laixianxin/LAPA/data/egoverse_laq_labels/latent_action_egoverse_train.jsonl"
    )


    p.add_argument(
        "--manifest",
        default="/mnt/data/laixianxin/LAPA/data/egoverse_vqgan_tokens/canonical_vqgan_manifest.jsonl"
    )


    p.add_argument(
        "--output",
        required=True
    )


    args=p.parse_args()


    print("="*100)
    print("LAPA 184 CPU BUILDER")
    print("workers =",args.workers)
    print("="*100)



    print("building VQ index")

    vq_index=build_vq_index(
        args.manifest
    )


    print(
        "VQ episodes=",
        len(vq_index)
    )


    size=os.path.getsize(args.laq)


    print(
        "LAQ size GB=",
        size/1024**3
    )


    parts=Path(
        args.output+".parts"
    )

    parts.mkdir(
        exist_ok=True
    )


    jobs=[]


    step=size//args.workers


    for i in range(args.workers):

        start=i*step

        end=size if i==args.workers-1 else (i+1)*step


        jobs.append(
            (
                i,
                start,
                end,
                args.laq,
                vq_index,
                str(parts/f"{i:04d}.jsonl")
            )
        )



    total=0
    miss=0


    with ProcessPoolExecutor(
        args.workers
    ) as ex:


        for r in ex.map(
            worker,
            jobs
        ):

            wid,n,m=r

            total+=n
            miss+=m

            print(
                "worker",
                wid,
                "samples",
                n,
                "miss",
                m
            )


    print("merging")


    with open(args.output,"w") as out:

        for p in sorted(
            parts.glob("*.jsonl")
        ):

            with open(p) as f:

                for line in f:
                    out.write(line)



    print("="*100)
    print("DONE")
    print("samples=",total)
    print("miss=",miss)
    print("output=",args.output)
    print("="*100)



if __name__=="__main__":
    main()
