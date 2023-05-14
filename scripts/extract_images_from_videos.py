"""
Extracts images from (compressed) videos, used for the FaceForensics++ dataset

Usage: see -h or https://github.com/ondyari/FaceForensics

Author: Andreas Roessler
Date: 25.01.2019
"""

# Source: https://github.com/ondyari/FaceForensics/blob/master/dataset/extract_compressed_videos.py

import os
from os.path import join
import argparse
import subprocess
import cv2
from tqdm import tqdm


DATASET_PATHS = {
    "original_youtube": "original_sequences/youtube",
    "original_actors": "original_sequences/actors",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "DeepFakeDetection": "manipulated_sequences/DeepFakeDetection",
    # "Face2Face": "manipulated_sequences/Face2Face",
    # "FaceSwap": "manipulated_sequences/FaceSwap",
}
COMPRESSION = ["raw", "c23", "c40"]


def extract_frames(data_path, output_path, method="cv2"):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    os.makedirs(output_path, exist_ok=True)
    if method == "ffmpeg":
        subprocess.check_output(
            "ffmpeg -i {} {}".format(data_path, join(output_path, "%04d.png")),
            shell=True,
            stderr=subprocess.STDOUT,
        )
    elif method == "cv2":
        reader = cv2.VideoCapture(data_path)
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            cv2.imwrite(join(output_path, "{:04d}.png".format(frame_num)), image)
            frame_num += 1
        reader.release()
    else:
        raise Exception("Wrong extract frames method: {}".format(method))


def extract_method_videos(data_path, dataset, compression, method):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(data_path, DATASET_PATHS[dataset], compression, "videos")
    images_path = join(data_path, DATASET_PATHS[dataset], compression, "images")
    for video in tqdm(os.listdir(videos_path)):
        image_folder = video.split(".")[0]
        extract_frames(
            join(videos_path, video), join(images_path, image_folder), method
        )


def extract_masks(data_path, dataset, method):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(data_path, DATASET_PATHS[dataset], "masks", "videos")
    images_path = join(data_path, DATASET_PATHS[dataset], "masks", "images")
    for video in tqdm(os.listdir(videos_path)):
        image_folder = video.split(".")[0]
        extract_frames(
            join(videos_path, video), join(images_path, image_folder), method
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=list(DATASET_PATHS.keys()) + ["all"],
        default="all",
    )
    p.add_argument("--compression", "-c", type=str, choices=COMPRESSION, default="raw")
    p.add_argument(
        "--method", "-m", type=str, choices=["cv2", "ffmpeg"], default="ffmpeg"
    )
    args = p.parse_args()

    if args.dataset == "all":
        for dataset in DATASET_PATHS.keys():
            print("Extracting frames for Dataset:", dataset)
            args.dataset = dataset
            extract_method_videos(**vars(args))
    else:
        extract_method_videos(**vars(args))

    if args.dataset == "all" or args.dataset == "Deepfakes":
        print("Extracting Deepfakes masks")
        extract_masks(args.data_path, "Deepfakes", args.method)
        print("Extracting DeepFakeDetection masks")
        extract_masks(args.data_path, "DeepFakeDetection", args.method)
