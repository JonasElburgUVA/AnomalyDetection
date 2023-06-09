# VQ-VAE DeepFake Anomaly Detection

This repository contains an application of [Anomaly Detection Through Latent Space Restoration Using Vector Quantized Variational Autoencoders](https://ieeexplore.ieee.org/abstract/document/9433778) to the task of DeepFake detection.

## Setup

### Environment

Create a conda environment and run `setup.sh` to install the required dependencies and download the pretrained models

```sh
conda create -n anomaly-vqvae python=3.10
conda activate anomaly-vqvae
sh setup_env.sh
```

The training scrips are set up to log metrics to [wandb](https://wandb.ai/site). The command line tool is installed by default, and if you want to log metrics simply run `wandb login` and provide your API key.

### Datasets

#### MOOD (Original Paper)

The original paper's workd was made in the context of the [MOOD challenge](http://medicalood.dkfz.de/web/) whose goal is identifying anomalies in brain MRI and abdominal CT scans. The data is not directly available, but can be requested by filling a form on the challenge website

#### FFHQ

We used a subset of ~52k images of FFHQ to train our models. The dataset is [hosted on Kaggle](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) and can be downloaded and preprocessed with the following command

```sh
# See https://github.com/Kaggle/kaggle-api for instructions on
# how to find your Kaggle credentials
export KAGGLE_USERNAME=username
export KAGGLE_KEY=kaggle-api-key

sh src/scripts/download_ffhq.sh
```

#### FaceForensics

Face Forensics is not directly available, but access can be requested via the google form [mentioned here](https://github.com/ondyari/FaceForensics/tree/master/dataset). We used the `original` and `Deepfakes` videos, in `raw` quality.

Once you have downloaded the videos run the face extraction script with the following commands

```sh
# Original samples
python -u src/scripts/extract_images_from_videos.py \
    --data_path /path/to/original/videos \
    --dataset original \
    --compression c0 \
    --extract_every 4

# Deepfakes
python -u src/scripts/extract_images_from_videos.py \
    --data_path /path/to/deepfake/videos \
    --dataset original \
    --compression c0 \
    --extract_every 10
```

To split the extracted faces into `train,val,test` use the `create_ff_splits.py` script as follows. This assumes you have downloaded the `json` split files [available here](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits) into the `./data/splits/` folder.

```sh
python src/scripts/create_ff_splits.py \
    --split_dir "./data/splits/" \
    --image_path "./data/original_sequences/youtube/raw/images/"

python src/scripts/create_ff_splits.py \
    --split_dir "./data/splits/" \
    --image_path "./data/manipulated_sequences/Deepfakes/raw/images/" \
    --is_deepfake
```

#### Data Folder setup

Once you've downloaded and preprocessed the datasets you should organize them so that the directories are structured like this:

```
data
├── faceforensics
│   ├── test
│   │   ├── fake
│   │   │   └── XXX_YYY
│   │   └── real
│   │       └── ZZZ
│   ├── val
│   │   ├── fake
│   │   │   └── XXX_YYY
│   │   └── real
│   │       └── ZZZ
│   └── val_pruned
│       ├── fake
│       │   └── XXX_YYY
│       └── real
│           └── ZZZ
└── ffhq
    ├── test
    │   ├── easy
    │   │   ├── 0
    │   │   └── 1
    │   ├── hard
    │   │   ├── 0
    │   │   └── 1
    │   └── medium
    │       ├── 0
    │       └── 1
    └── val
        ├── 0
        └── 1
```

Note on 'val_pruned': Since the validation set is only used for finding the optimal threshold value, the set was decreased in size in our experiments. The script for this can be found in the repository. However, this step is optional.

You will also need an output folder with the following structure. This can be placed within the data folder.

```
output
├── faceforensics
└── ffhq
```

Finally, the checkpoints should be in the same folder, and should follow the naming convention {dataset}_{model}.pt. Here dataset can be either 'ffhq' or 'faceforensics', and model can be either 'vqvae' or 'ar'.

## Model Training

We provide scripts to train the VQ-VAE and AR models both on FFHQ and FaceForensics. For FFHQ you can train the models as follows

```sh
# You can optionally resume training from a checkpoint by providing
# a path using the --vqvae_checkpoint argument.

# This script trains the VQ-VAE model
python src/DeepFake/train.py \
    --training_directory "data/ffhq/train" \
    --holdout_directory "data/ffhq/holdout" \
    --epochs 30 

# This script traing the AR model
# Provide the --ar_checkpoint argument to resume training
python src/DeepFake/train_ar.py \
    --training_directory "data/ffhq/train" \
    --holdout_directory "data/ffhq/holdout" \
    --vqvae_checkpoint "checkpoints/ffhq/vqvae.pt" \
    --epochs 30
```

The scripts for FaceForensics follow a similar usage pattern, and they are respectively named `faceforensics_train.py` and `faceforensics_train_ar.py`.

## Running Experiments

Assuming you have downloaded the `brain_toy.zip` data you can unpack it with `unzip brain_toy.zip -d data/toy` and run experiments using the following commands:

```sh
python src/OriginalPaper/docker/scripts/pred.py \
    -i "data/brain_toy/toy" \
    -o "output/full/sample" \
    -mode "sample" \
    -d "brain" \
    --checkpoint_features "checkpoints/brain/vqvae.pt" \
    --checkpoint_latent "checkpoints/brain/ar.pt"

python src/OriginalPaper/docker/scripts/pred.py \
    -i "data/brain_toy/toy" \
    -o "output/full/pixel" \
    -mode "pixel" \
    -d "brain" \
    --checkpoint_features "checkpoints/brain/vqvae.pt" \
    --checkpoint_latent "checkpoints/brain/ar.pt"
```

For the **real/deepfake experiments** refer to the notebook