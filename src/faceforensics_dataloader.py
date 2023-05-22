import os

import torch
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.transforms.functional as tf

from PIL import Image
import random
import numpy as np


"""
|-- downloaded_videos
    < contains all original downloaded videos, video information files and their extracted sequences
      which can be used to extract the original sequences used in the dataset >
|-- original_sequences
    |-- youtube
        < c0/raw original sequence images/videos of the FaceForensics++ dataset >
        < c23/hq original sequence images/videos >
        < c40/lq original sequence images/videos >
    |-- actors
        < images/videos from the DeepFakeDetection dataset >
|-- manipulated_sequences
    |-- Deepfakes
        < images/videos of all three compression degrees as well as models and masks after poisson image editing>
        < based on the original_sequences/youtube/ folder>
    |-- DeepFakeDetection
        < images/videos ... as well as masks based on the original_sequences/actors/ folder>

"""


class FaceForensicsDataset(Dataset):
    """Images are sized 720x1280 (HxW)"""

    def __init__(
        self, data_dir, split, compression="raw", sample_number=64, deepfakes=False
    ) -> None:
        super().__init__()

        if split not in ["train", "dev", "test"]:
            raise NotImplemented("Invalid split. Available split are: train, test, dev")

        if deepfakes:
            self.path = os.path.join(
                data_dir,
                "manipulated_sequences",
                "Deepfakes",
                compression,
                "images",
                split,
            )
        else:
            self.path = os.path.join(
                data_dir, "original_sequences", "youtube", compression, "images", split
            )
        self.files = sorted(os.listdir(self.path))
        self.sample_number = sample_number
        self.length_limit = min(
            [len(os.listdir(os.path.join(self.path, f))) for f in self.files]
        )
        assert (
            self.sample_number <= self.length_limit
        ), f"Sample number exceeds shortest video length {self.length_limit}"

        self.transform = transform
        self.to_tensor = transforms.PILToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns batch of frames of a Deepfaked video shaped BxLxCxHxW, where L is the sample_number
        """
        fname = self.files[idx]
        dir = os.path.join(self.path, fname)
        frames = os.listdir(dir)
        frames = random.sample(os.listdir(dir), k=self.sample_number)
        img_batch = torch.stack(
            [self.to_tensor(Image.open(os.path.join(dir, f))) for f in frames]
        )

        if self.transform is not None:
            img_batch = self.transform(img_batch)

        return img_batch


def collate_fn(batch):
    """Transform the input batch of shape BxLxCxHxW to (B*L)xCxHxW"""
    # Return the flattened batch
    stacked_batch = torch.stack(batch)
    flattened_batch = stacked_batch.view(-1, *stacked_batch.shape[2:])
    # Shuffle the flattened batch
    indices = torch.randperm(flattened_batch.size(0))
    batch = flattened_batch[indices]

    return flattened_batch


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import torchvision

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    # NOTE: Random transformation won't be consistent between outputs. So don't include them here
    transform = transforms.Compose(
        [
            transforms.Resize(
                (126, 224), antialias=None
            ),  # resize the image to whatever
        ]
    )

    dataset = FaceForensicsDataset(
        "./data", split="train", compression="raw", sample_number=10, deepfakes=False
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for batch in loader:
        print("Batch_Size:", batch.shape)
        plt.imshow(batch[0].permute(1, 2, 0))
        plt.show()
        break
