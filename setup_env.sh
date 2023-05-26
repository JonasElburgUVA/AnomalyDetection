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
gdown 1nAh81s142xzE5VUf-tZibq-pIUaqoLiU -O checkpoints/brain/vqvae.pt
gdown 1RgQ1bpRn0zrJRSC2sqG5GyywNgYdw4Zx -O checkpoints/brain/ar.pt

# FFHQ Checkpoints
gdown 1VsFMxSJEsxjhcXVseDTK6dydBKdUZ1w1 -O checkpoints/ffhq/vqvae.pt
gdown 1bEtV6NktUtgyOqd-rpuWS6pA4VJ12Eke -O checkpoints/ffhq/ar.pt

# Face Forensics checkpoints
gdown 1orYcmEfAiSBzMiuSJeAOGBK1kKN3_fG4 -O checkpoints/faceforensics/vqvae.pt
gdown 1MXiYTViwgUMo0pwwMR4kkuRS6ZU9Wez4 -O checkpoints/faceforensics/ar.pt

# Datasets
mkdir data
mkdir output

# Output folders
mkdir -p output/full/sample
mkdir -p output/full/pixel

mkdir -p output/faceforensics
mkdir -p output/ffhq
