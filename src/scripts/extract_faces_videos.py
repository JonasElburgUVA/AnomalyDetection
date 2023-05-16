"""
    Extract faces from a directory containing multiple videos
    in .mp4 format. This script requires the following libraries:

    - mmcv
    - facenet-pytorch
"""
import os
import cv2
import glob
import mmcv
import torch
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN


parser = argparse.ArgumentParser(
    prog="Face Extractor",
    description="This program extracts all faces in the source directory"
)

parser.add_argument("--source_directory", type=str, help="The directory with the video files to extract faces from")
parser.add_argument("--output_directory", type=str, help="The directory where extracted images should be saved")
parser.add_argument("--min_size", type=int, default=128, help="The minimum size length in pixels")
parser.add_argument("--min_confidence", type=int, default=0.995, help="The minimum confidence for a face to be saved")
parser.add_argument("--padding", type=int, default=16, help="Padding added to frames to ensure cropped images are squares")

args = parser.parse_args()

os.makedirs(args.output_directory, exist_ok=True)

device = torch.device("cuda")
min_image_size = args.min_size ** 2
mtcnn = MTCNN(keep_all=True, device=device)
videos = glob.glob(os.path.join(args.source_directory, "*.mp4"))

for video_path in tqdm(videos):
    filename = video_path.split("/")[-1]
    video_id = filename.split(".")[0]

    video = mmcv.VideoReader(video_path)
    frames = [
        Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ) for frame in video
    ]

    for i, frame in enumerate(frames):
        # Boxes are returned as a list in the form
        # [x0, y0, x1, y1]
        boxes, confidences = mtcnn.detect(frame)

        # Pad the image to ensure square crops
        w, h = frame.size
        w, h = w + args.padding, h + args.padding

        padded_frame = Image.new("RGB", (w, h))
        padded_frame.paste(
            frame,
            tuple((n - o) // 2 for n, o in zip((w, h), frame.size))
        )

        array_frame = np.array(padded_frame)

        for face_id, (box, confidence) in enumerate(zip(boxes, confidences)):
            if confidence < args.min_confidence:
                continue

            x0, y0 = np.ceil(box[:2]).astype(int)
            x1, y1 = np.floor(box[2:]).astype(int)

            # Make sure that the coordinates differences are always even
            if (y1 - y0) % 2 != 0:
                # Make sure we are not going out of bounds
                if y1 < array_frame.shape[0]:
                    y1 += 1
                else:
                    y0 -= 1

            if (x1 - x0) % 2 != 0:
                if x1 < array_frame.shape[1]:
                    x1 += 1
                else:
                    x0 -= 1

            # Calculate the difference between the two sides of the image
            coords_diff = abs((x1 - x0) - (y1 - y0))

            # Make sure the two sides have the same length, and
            # keep the image center by adding half of the length
            # difference to one side and half to the other
            if (y1 - y0) > (x1 - x0):
                x0 = max(0, int(x0 - coords_diff / 2))
                x1 = min(array_frame.shape[1], int(x1 + coords_diff / 2))
            elif (x1 - x0) > (y1 - y0):

                y0 = max(0, int(y0 - coords_diff / 2))
                y1 = min(array_frame.shape[0], int(y1 + coords_diff / 2))

            face_slice = array_frame[y0:y1, x0:x1, :]
            confidence = str(round(confidence, 3)).replace(".", "")

            # Filter out images that are too small
            if np.prod(face_slice.shape[:2]) < min_image_size:
                continue

            face_image = Image.fromarray(face_slice)
            face_image.save(os.path.join(
                args.output_directory,
                f"face_{video_id}_{i}_{face_id}_{confidence}.jpg"
            ))
