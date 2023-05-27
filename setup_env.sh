pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

pip install h5py
pip install tqdm
pip install wandb
pip install nibabel
pip install matplotlib
pip install torchmetrics
pip install scikit-learn
pip install scikit-image
pip install albumentations
pip install tensorboard
pip install gdown
pip install kaggle

mkdir -p checkpoints/brain
mkdir -p checkpoints/ffhq
mkdir -p checkpoints/faceforensics

# Brain checkpoints
gdown 1WaECO8w_NkR0sgp6z6itGgn-U1NyaxET -O checkpoints/brain/vqvae.pt
gdown 1oYxI0gPqKwWjP-oV6cJ9wPN-y0ScU8aH -O checkpoints/brain/ar.pt

# FFHQ Checkpoints
gdown 10BWRqmzS6k7ZfXTlebh9FK0oa-AbJ3e6 -O checkpoints/ffhq/vqvae.pt
gdown 1lwTMQphz3WZnQSOdkqgqUliF-Xs_Vm4l -O checkpoints/ffhq/ar.pt

# Face Forensics checkpoints
gdown 1p_0RtYlVEOalJ9_VqmpJbrUf-CQ84mNx -O checkpoints/faceforensics/vqvae.pt
gdown 1qNx_0qLKgK2lo7EZhPlmTEitH8So-75n -O checkpoints/faceforensics/ar.pt

# Datasets
mkdir data
mkdir output

# Output folders
mkdir -p output/full/sample
mkdir -p output/full/pixel

mkdir -p output/faceforensics
mkdir -p output/ffhq
