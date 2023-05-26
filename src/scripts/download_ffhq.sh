# This requires the kaggle python package to be installed
# and the credentials to be available
kaggle datasets download arnaud58/flickrfaceshq-dataset-ffhq

# Unzip the dataset
mkdir -p data/ffhq
unzip flickrfaceshq-dataset-ffhq.zip -d data/ffhq

# Split between validation and training
mkdir -p data/ffhq/train/regular
mkdir -p data/ffhq/validation/regular
mkdir -p data/ffhq/fake/

# Test dataset directory
mkdir -p data/ffhq/test/0
mkdir -p data/ffhq/test/1

python src/scripts/ffhq.py --dataset_path data/ffhq

mv data/ffhq/*.png data/ffhq/train/regular

# Download the real and fake faces dataset
kaggle datasets download ciplab/real-and-fake-face-detection

# Unpack the dataset
unzip real-and-fake-face-detection.zip -d data/ffhq/fake

# Remove the extra unpacked directory
rm -rf data/ffhq/fake/real_and_fake_face_detection
rm -rf data/ffhq/fake/real_and_fake_face/training_real

# Split the images by difficulty
mkdir -p data/ffhq/fake/easy
mkdir -p data/ffhq/fake/medium
mkdir -p data/ffhq/fake/hard

mv data/ffhq/fake/real_and_fake_face/training_fake/easy_* data/ffhq/fake/easy
mv data/ffhq/fake/real_and_fake_face/training_fake/mid_* data/ffhq/fake/medium
mv data/ffhq/fake/real_and_fake_face/training_fake/hard_* data/ffhq/fake/hard

rm -rf data/ffhq/fake/real_and_fake_face

# Create the dataset for lambda tuning
mkdir -p data/ffhq/lambda_tuning/0
mkdir -p data/ffhq/lambda_tuning/1

# Copy the first 252 images from the holdout directory to the lambda tuning real folder
find data/ffhq/validation/regular/*.png -maxdepth 1 -type f | head -252 | xargs cp -t "data/ffhq/lambda_tuning/0"
# Copy 84 images per difficulty to the lambda tuning fake directory
find data/ffhq/fake/easy/*.jpg -maxdepth 1 -type f | head -84 | xargs cp -t "data/ffhq/lambda_tuning/1"
find data/ffhq/fake/medium/*.jpg -maxdepth 1 -type f | head -84 | xargs cp -t "data/ffhq/lambda_tuning/1"
find data/ffhq/fake/hard/*.jpg -maxdepth 1 -type f | head -84 | xargs cp -t "data/ffhq/lambda_tuning/1"

# Create the testing dataset using the holdout and fake images.
# The fake images total 960 samples
cp -r data/ffhq/fake/easy/*.jpg data/ffhq/test/1
cp -r data/ffhq/fake/medium/*.jpg data/ffhq/test/1
cp -r data/ffhq/fake/hard/*.jpg data/ffhq/test/1

find data/ffhq/validation/regular/*.png -maxdepth 1 -type f | head -960 | xargs cp -t "data/ffhq/test/0"
