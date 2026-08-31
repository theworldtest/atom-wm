#!/usr/bin/env python3

import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


VQROOT = Path(
    "/mnt/data/laixianxin/LAPA/data/"
    "egoverse_vqgan_tokens"
)


MANIFEST = VQROOT / "canonical_vqgan_manifest.jsonl"


LAQ = Path(
    "/mnt/data/laixianxin/LAPA/data/"
    "egoverse_laq_labels/"
    "latent_action_egoverse_train.jsonl"
)



def load_one_episode(ep):

    result={}

    path=Path(ep["vqgan_path"])

    with open(path) as f:

        for line in f:

            x=json.loads(line)

            key=(

                x["dataset"],

                int(x["split_episode_index"]),

                int(x["frame_index"])

            )

            result[key]=x["vision"]


    return result



def main():


    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--max-vq-episodes",
        type=int,
        default=-1
    )


    parser.add_argument(
        "--workers",
        type=int,
        default=32
    )


    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1
    )


    parser.add_argument(
        "--output",
        required=True
    )


    args=parser.parse_args()



    print("="*100)
    print("LAPA EGOVERSE JSONL BUILDER")
    print("workers =",args.workers)
    print("="*100)



    ##################################################
    # manifest
    ##################################################

    episodes=[]


    with open(MANIFEST) as f:

        for i,line in enumerate(f):

            episodes.append(json.loads(line))

            if (
            args.max_vq_episodes > 0
            and len(episodes)>=args.max_vq_episodes
            ):
                break


    print(
        "episodes =",
        len(episodes)
    )



    ##################################################
    # parallel VQ loading
    ##################################################

    vision_map={}


    start=time.time()


    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as pool:


        from concurrent.futures import as_completed


        chunk_size = 500


        with ProcessPoolExecutor(
            max_workers=args.workers
        ) as pool:


            for start in range(
                0,
                len(episodes),
                chunk_size
            ):

                chunk = episodes[
                    start:start+chunk_size
                ]


                futures = [
                    pool.submit(
                        load_one_episode,
                        ep
                    )
                    for ep in chunk
                ]


                for f in as_completed(futures):

                    vision_map.update(
                        f.result()
                    )


                print(
                    "loaded episodes:",
                    min(
                        start+chunk_size,
                        len(episodes)
                    ),
                    "/",
                    len(episodes),
                    "frames:",
                    len(vision_map)
                )


        for i,f in enumerate(
            as_completed(futures),
            1
        ):

            vision_map.update(
                f.result()
            )


            if i%1000==0:

                print(
                    "loaded",
                    i,
                    "/",
                    len(episodes),
                    "frames=",
                    len(vision_map),
                    "time=",
                    round(time.time()-start,1)
                )



    print(
        "VQ loaded frames=",
        len(vision_map)
    )


    ##################################################
    # LAQ merge
    ##################################################

    out=Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    count=0
    miss=0


    with open(LAQ) as fin, open(out,"w") as fout:


        for line in fin:


            x=json.loads(line)


            key=(

                x["dataset"].split("/")[0],

                int(x["split_episode_index"]),

                int(x["frame_index"])

            )


            vision=vision_map.get(key)


            if vision is None:

                miss+=1
                continue



            y={

                "id":
                    x["source_episode_id"]
                    +"_"
                    +str(x["frame_index"]),


                "instruction":
                    x["instruction"],


                "vision":
                    vision,


                "delta":
                    [
                        int(a)
                        for a in x["delta"]
                    ]

            }


            fout.write(
                json.dumps(y)
                +"\n"
            )


            count+=1


            if count%100000==0:

                print(
                    "written",
                    count,
                    "miss",
                    miss
                )


            if (
                args.max_samples>0
                and count>=args.max_samples
            ):
                break



    print("="*100)
    print("DONE")
    print("samples =",count)
    print("miss =",miss)
    print("output =",out)
    print("="*100)



if __name__=="__main__":

    main()

