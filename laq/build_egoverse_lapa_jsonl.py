#!/usr/bin/env python3

import json
import time
import argparse
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()

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


    LAQ = Path(
        "/mnt/data/laixianxin/LAPA/data/"
        "egoverse_laq_labels/"
        "latent_action_egoverse_train.jsonl"
    )


    MANIFEST = Path(
        "/mnt/data/laixianxin/LAPA/data/"
        "egoverse_vqgan_tokens/"
        "canonical_vqgan_manifest.jsonl"
    )


    VQROOT = Path(
        "/mnt/data/laixianxin/LAPA/data/"
        "egoverse_vqgan_tokens"
    )


    print("="*100)
    print("BUILD LAPA JSONL")
    print("="*100)


    ########################################
    # 1. build frame -> vision map
    ########################################

    print("Loading VQGAN episodes...")


    vision_map={}


    with open(MANIFEST) as f:

        for i,line in enumerate(f):

            ep=json.loads(line)


            vq_file=Path(ep["vqgan_path"])


            with open(vq_file) as vf:

                for frame_line in vf:

                    frame=json.loads(frame_line)


                    key=(
                        frame["dataset"],
                        int(frame["split_episode_index"]),
                        int(frame["frame_index"])
                    )


                    vision_map[key]=frame["vision"]


            if (i+1)%1000==0:
                print(
                    "episodes loaded",
                    i+1
                )


    print(
        "vision frames =",
        len(vision_map)
    )


    ########################################
    # 2. merge LAQ
    ########################################


    out=Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    print("Streaming LAQ...")


    count=0
    miss=0

    start=time.time()


    with open(LAQ) as fin, open(out,"w") as fout:


        for line in fin:

            laq=json.loads(line)


            key=(
                laq["dataset"].split("/")[0],
                int(laq["split_episode_index"]),
                int(laq["frame_index"])
            )


            vision=vision_map.get(key)


            if vision is None:
                miss+=1
                continue


            sample={

                "id":
                    laq["source_episode_id"]
                    +"_"
                    +str(laq["frame_index"]),


                "instruction":
                    laq["instruction"],


                "vision":
                    vision,


                "delta":
                    [
                        int(x)
                        for x in laq["delta"]
                    ]
            }


            fout.write(
                json.dumps(sample)
                +"\n"
            )


            count+=1


            if count%100000==0:
                print(
                    "written",
                    count,
                    "miss",
                    miss,
                    "time",
                    round(time.time()-start,1)
                )


            if args.max_samples>0 and count>=args.max_samples:
                break



    print("="*100)
    print("DONE")
    print("samples",count)
    print("miss",miss)
    print("output",out)
    print("="*100)



if __name__=="__main__":
    main()

