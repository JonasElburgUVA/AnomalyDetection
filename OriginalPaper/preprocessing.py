import os
import glob
import numpy as np
import random

from torch.utils.data import Dataset

import utils
import pickle

import albumentations as A

class img_dataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform_shape=None,
        transform_color=None,
        sample=False,
        sample_number=4,
        sample_anomaly=None,
        sample_cond_threshold=0,
        slice_offset=20,
        total_slices=256
    ):
        """
        Args:
        :param data_dir: Directory of images (str)
        :param transform_shape: Albumentations transforms
        :param transform_color: Albumentations transforms
        :param sample: Return sampled slices
        :param sample: Number of slices to be sampled
        :param sample_anomally: [None, normal, abnormal]
        :param sample_cond_threshold: Threshold to apply to the label so define anomaly (e.g. labels in segmentation > 1 are anomalies)
        """
        self.data_dir = data_dir
        self.set = glob.glob(data_dir+'/*.nt')
        self.transform_shape = transform_shape
        self.transform_color = transform_color
        self.sample = sample
        self.sample_number = sample_number
        self.sample_anomaly = sample_anomaly
        self.slice_offset = slice_offset
        self.sample_cond_threshold = sample_cond_threshold
        self.total_slices = total_slices

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):
        file_name = os.path.join(self.data_dir,self.set[item])
        sample = pickle.load(open(file_name, 'rb'))

        if self.sample:
            # FIXME: This assumes that self.sample_anomaly is always None
            choices = np.arange(len(sample.empty_mask))[sample.empty_mask]
            sample_idx = np.array(random.choices(choices, k=self.sample_number))

            img = sample.img[sample_idx].astype(np.float32)

            # Position of the slice within the volume, encoded in a variable in the range [-0.5,0.5].
            coord = sample_idx[:, np.newaxis] / self.total_slices
            coord = coord - 0.5

            img_batch = utils.mri_sample(
                img,
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                coord,
                np.array([sample.cid] * self.sample_number),
                np.zeros(0)
            )
        else: # If no ssampling is required, just reverse the order: list of img_ext to img_ext of arrays
            img_batch = utils.mri_sample(*map(np.array, zip(*[sample])))

        if self.transform_shape is not None:
            img_batch = utils.mri_sample(
                np.stack([self.transform_shape(image=img)["image"] for img in img_batch.img]),
                img_batch.seg,
                img_batch.k,
                img_batch.t,
                img_batch.coord,
                img_batch.cid,
                img_batch.empty_mask
            )

        if self.transform_color is not None:
            cero_mask = img_batch.img == 0

            # Set to range [0,1], clipping any value further than 3 sigma
            img_aug = np.clip((img_batch.img + 3.) / 6., 0, 1)
            img_aug = np.stack([self.transform_color(image=i)["image"] for i in img_aug])
            img_aug = img_aug * 6. - 3.
            img_aug[cero_mask] = 0

            img_batch = utils.mri_sample(
                img_aug,
                img_batch.seg,
                img_batch.k,
                img_batch.t,
                img_batch.coord,
                img_batch.cid,
                img_batch.empty_mask
            )

        return img_batch


def collate_fn(batches):
    return utils.mri_sample(*map(np.concatenate, zip(*batches)))


class brain_dataset(img_dataset):
    def __init__(self, data_dir, train = True,  **kwargs):

        if train:
            transform_shape = A.Compose([
                A.ElasticTransform(alpha=2, sigma=5, alpha_affine=5),
                A.RandomScale((-.15, .1)),
                A.PadIfNeeded(160, 160, value=0, border_mode=1),
                A.CenterCrop(160, 160),
                A.HorizontalFlip(),
                A.Rotate(limit=5),
            ])

            transform_color = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=.15, contrast_limit=.15),
                A.GaussianBlur(blur_limit=7),
                A.GaussNoise(var_limit=.001)
            ])
        else:
            transform_shape = transform_color = None

        super(brain_dataset,self).__init__(
            data_dir,
            sample=True,
            transform_shape=transform_shape,
            transform_color=transform_color,
            **kwargs
        )
