import os
import random

import torch

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


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
    def __init__(self, data_dir, split, sample_number=64, transform=None) -> None:
        super().__init__()

        if split not in ["train", "dev", "test"]:
            raise NotImplemented("Invalid split. Available split are: train, test, dev")

        self.path = os.path.join(data_dir, split)
        self.files = sorted(os.listdir(self.path))
        self.sample_number = sample_number
        self.length_limit = min(
            [len(os.listdir(os.path.join(self.path, f))) for f in self.files]
        )

        assert (
            self.sample_number <= self.length_limit
        ), f"Sample number exceeds shortest video length {self.length_limit}"

        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns batch of frames of a Deepfaked video shaped BxLxCxHxW, where L is the sample_number
        """
        fname = self.files[idx]
        video_dir = os.path.join(self.path, fname)

        frames = os.listdir(video_dir)
        frames = random.sample(os.listdir(video_dir), k=self.sample_number)

        batch_transform = self.transform if self.transform else self.to_tensor     

        return torch.stack([
            batch_transform(
                Image.open(os.path.join(video_dir, f))
            ) for f in frames
        ])


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
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    # NOTE: Random transformation won't be consistent between outputs. So don't include them here
    transform = transforms.Compose(
        [
            transforms.Resize(
                (126, 224), antialias=None
            ),  # resize the image to whatever
            transforms.ToTensor()
        ]
    )

    dataset = FaceForensicsDataset(
        "./data", split="train", sample_number=10, transform=transform
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for batch in loader:
        print("Batch_Size:", batch.shape)

        plt.imshow(batch[0].permute(1, 2, 0))
        plt.show()

        break
