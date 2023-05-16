# This requires the kaggle python package to be installed
# and the credentials to be available
kaggle datasets download arnaud58/flickrfaceshq-dataset-ffhq

# Unzip the dataset
mkdir -p ffhq
unzip flickrfaceshq-dataset-ffhq.zip -d ffhq

# split between validation and training
mkdir -p ffhq/train/regular
mkdir -p ffhq/validation/regular

python ffhq.py --dataset_path ffhq

mv ffhq/*.png ffhq/train/regular
