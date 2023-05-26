import os
from shutil import copyfile
import random
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Prune FaceForensics dataset.")
    parser.add_argument(
        "--datapath",
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Split to prune",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        help="Max number of images to keep per video",
    )


if __name__ == "__main__":
    random.seed(42)
    args = parse_arguments()
    datapath = args.datapath
    split = args.split
    CUTOFF = args.cutoff

    datapath= os.path.join(datapath, f"{split}_set")
    destination = os.path.join(datapath, f"{split}_set_pruned")
    for real_or_fake in os.listdir(datapath):
        for imgfolder in os.listdir(os.path.join(datapath, real_or_fake)):
            t = 0
            ls = os.listdir(os.path.join(datapath, real_or_fake, imgfolder))
            random.shuffle(ls)
            cutoff = min(CUTOFF, len(ls))
            while(t<cutoff):
                src = os.path.join(datapath, real_or_fake, imgfolder, ls[t])
                dst = os.path.join(destination, real_or_fake, imgfolder, ls[t])
                if not os.path.exists(os.path.join(destination, real_or_fake, imgfolder)):
                    os.makedirs(os.path.join(destination, real_or_fake, imgfolder))
                copyfile(src, dst)
                t += 1

