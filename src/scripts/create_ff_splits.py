import os
import shutil
import json
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Move directories based on splits.")
    parser.add_argument(
        "--split_dir",
        type=str,
        help="Path to the split file directory (in JSON format)",
    )
    parser.add_argument("--image_path", type=str, help="Path to the image directory")
    parser.add_argument(
        "--is_deepfake", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    split_files = os.listdir(args.split_dir)

    for split_file in split_files:
        with open(os.path.join(args.split_dir, split_file)) as f:
            splits = json.load(f)

        split_name = os.path.splitext(os.path.basename(split_file))[0]
        print(
            f"Moving {split_name} set into {os.path.join(args.image_path, split_name)}"
        )
        # Create the train folder if it doesn't exist
        destination_dir = os.path.join(args.image_path, split_name)

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Move directories from image folder to split folder based on splits JSON
        for data in splits:
            if args.is_deepfake:
                data = ["_".join(data), "_".join(data[::-1])]

            for src_dir in data:
                source_dir = os.path.join(args.image_path, src_dir)
                if os.path.exists(source_dir):
                    shutil.move(source_dir, destination_dir)
                    # print(f"Moved directory {data} to train folder.")
