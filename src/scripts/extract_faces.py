import os
import sys
import glob
import argparse

from PIL import Image
from tqdm import tqdm

try:
    import face_recognition
except ImportError:
    print(f"face_recognition is required to run this script. Install it with pip install face_recognition")
    sys.exit(-1)


parser = argparse.ArgumentParser(
    prog="Face Extractor",
    description="This program extracts all faces in the source directory"
)

parser.add_argument("--source_directory", type=str, help="The directory with the images to extract faces from")
parser.add_argument("--output_directory", type=str, help="The directory where extracted images should be saved")
parser.add_argument("--min_size", type=int, default=128, help="The minimum size length in pixels")

args = parser.parse_args()

os.makedirs(args.output_directory, exist_ok=True)

images = glob.glob(os.path.join(args.source_directory, "*.png"))

for file_path in tqdm(images):
    filename = file_path.split("/")[-1]

    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)

    for i, (top, right, bottom, left) in enumerate(face_locations):
        face = Image.fromarray(image[top:bottom, left:right, :])
        width, height = face.size

        if width < args.min_size:
            continue

        face.save(os.path.join(args.output_directory, f"{filename}_{i}.png"))
