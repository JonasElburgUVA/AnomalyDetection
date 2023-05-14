import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image
import random


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
        self,
        data_dir,
        task,
        frame_step=1,
        transform=None,
        compression="raw",
        face_only=False,
    ) -> None:
        super().__init__()

        if task not in ["Deepfakes"]:  # "DeepFakeDetection"
            raise NotImplemented(
                "Invalid task. choose: 'Deepfakes'"  # or 'DeepFakeDetection'"
            )
        self.task = task
        self.path = os.path.join(data_dir, "manipulated_sequences", task)

        dir_name = task == "Deepfakes" and "youtube" or "actors"
        self.source_path = os.path.join(
            data_dir, "original_sequences", dir_name, compression, "images"
        )

        self.image_path = os.path.join(self.path, compression, "images")
        self.mask_path = os.path.join(self.path, "masks", "images")

        self.files = sorted(os.listdir(self.image_path))
        self.frame_step = frame_step
        self.face_only = face_only

        self.length_limit = min(
            [len(os.listdir(os.path.join(self.image_path, f))) for f in self.files]
        )
        print("Shortest video in frames:", self.length_limit)
        self.transform = transform
        self.transform_img = transforms.PILToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns batch of frames of a Deepfaked video shaped BxLxCxHxW
        and the masks outlining the faces.

        """
        fname = self.files[idx]
        combined_dir = os.path.join(self.image_path, fname)
        frames = sorted(os.listdir(combined_dir))

        mask_dir = os.path.join(self.mask_path, fname)
        masks = sorted(os.listdir(mask_dir))

        # TODO: Naming scheme of DeepFakeDetection is different
        source, _ = fname.split("_")[:2]

        source_dir = os.path.join(self.source_path, source)
        src_frames = sorted(os.listdir(source_dir))
        # target_dir = os.path.join(self.source_path, target)

        # Limiting the length to the shortest video to allow batching
        # and masks start on the second frame so we skip the first
        frames = frames[1 : self.length_limit : self.frame_step]
        masks = masks[: self.length_limit : self.frame_step]
        src_frames = src_frames[1 : self.length_limit : self.frame_step]

        assert len(frames) == len(
            masks
        ), f"{len(frames)} != {len(masks)}. Length mismatch between mask and manipulated {fname}"

        assert len(frames) == len(
            src_frames
        ), f"{len(frames)} != {len(src_frames)}. Length mismatch between original {source} and manipulated {fname}"

        img_batch = []
        origin_batch = []
        mask_batch = []

        for i in range(len(frames)):
            img = self.transform_img(Image.open(os.path.join(combined_dir, frames[i])))
            origin = self.transform_img(
                Image.open(os.path.join(source_dir, src_frames[i]))
            )
            msk = self.transform_img(
                Image.open(os.path.join(mask_dir, masks[i])).convert("L")
            )
            # if self.face_only:
            #     img = self.apply_mask(img, msk)
            #     origin = self.apply_mask(origin, msk)
            img_batch.append(img)
            origin_batch.append(origin)
            mask_batch.append(msk)

        if self.transform is not None:
            # Random operations are not the same for each batch
            img_batch = torch.cat(
                [self.transform(img).unsqueeze(0) for img in img_batch]
            )
            origin_batch = torch.cat(
                [self.transform(img).unsqueeze(0) for img in origin_batch]
            )
            mask_batch = torch.cat(
                [self.transform(img).unsqueeze(0) for img in mask_batch]
            )

        return img_batch, origin_batch, mask_batch

    # def apply_mask(self, img, mask):
    #     """
    #     Return a crop that contains the face.
    #     """
    #     pass


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import torchvision

    seed = 40
    torch.manual_seed(seed)
    random.seed(seed)

    transform = transforms.Compose(
        [
            transforms.Resize(
                (126),
            ),  # resize the image to whatever
            #
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            # transforms.GaussianBlur((7, 7)),
        ]
    )

    task = "Deepfakes"

    dataset = FaceForensicsDataset(
        "./data", task, frame_step=10, transform=transform, compression="raw"
    )

    out, src, mask = dataset[0]
    out, src, mask = out[0], src[0], mask[0]

    # Create grid of images & turn mask in 3 channel image for visualization
    grid = torchvision.utils.make_grid(
        [out, src, mask.expand(3, -1, -1)],
        nrow=2,
    )
    # permute to PIL HxWxC format
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    loader = DataLoader(dataset, batch_size=5, num_workers=8)

    for img, src, msk in loader:
        print(img[0][0].shape)
        print(src[0][0].shape)
        print(msk[0][0].shape)
        break
