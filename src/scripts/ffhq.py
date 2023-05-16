"""
    This script splits the FFHQ dataset into train
    and validation using a 95/5 split
"""
import os
import glob
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", help="the FFHQ dataset path", required=True)

args = parser.parse_args()

random.seed(42)

data_files = glob.glob(os.path.join(args.dataset_path, "*.png"))
validation_files = random.sample(data_files, k=int(0.05 * len(data_files)))
validation_folder = os.path.join(args.dataset_path, os.path.join("validation", "regular"))

for file_path in validation_files:
    filename = file_path.split("/")[-1]

    os.rename(file_path, os.path.join(os.path.join(validation_folder), filename))
