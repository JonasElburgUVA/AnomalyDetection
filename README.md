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

Once you've downloaded and preprocessed the datasets you should organize them in the following folder structure:

```
# FILL ME
```

## Running Experiments

Assuming you have downloaded the `brain_toy.zip` data you can unpack it with `unzip brain_toy.zip -d data/toy` and run experiments using the following commands:

```sh
python docker/scripts/pred.py \
    -i "data/brain_toy/toy" \
    -o "output/full/sample" \
    -mode "sample" \
    -d "brain"

python docker/scripts/pred.py \
    -i "data/brain_toy/toy" \
    -o "output/full/pixel" \
    -mode "pixel" \
    -d "brain"
```

For the **real/deepfake experiments** refer to the notebook
