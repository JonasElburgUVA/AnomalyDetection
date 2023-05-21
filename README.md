# AnomalyDetection

Environment setup

```sh
# Module loading
module load 2022
module load Anaconda3/2022.05
nvidia-smi

# Download brain checkpoints
gdown "https://drive.google.com/uc?id=1a_Zgy4tStAIV9YaXZvnd4IUt2Yv5JEwL"
gdown "https://drive.google.com/uc?id=1-0rR_md7VSwt9tghPRgyNfcJNH93B4_C"

# Download the toy data for the brain
wget http://medicalood.dkfz.de/QtVaXgs8il/brain_toy.zip
```

Run experiments (evaluation)

```sh
python docker/scripts/pred.py -i "/home/lcur1720/dl2/AnomalyDetection/brain_toy/toy" -o "/home/lcur1720/dl2/AnomalyDetection/output/full/sample" -mode "sample" -d "brain"
python docker/scripts/pred.py -i "/home/lcur1720/dl2/AnomalyDetection/brain_toy/toy" -o "/home/lcur1720/dl2/AnomalyDetection/output/full/pixel" -mode "pixel" -d "brain"
```


```
python src/scripts/create_ff_splits.py --split_dir "./data/splits/" --image_path "./data/original_sequences/youtube/raw/images/"

python src/scripts/create_ff_splits.py --split_dir "./data/splits/" --image_path "./data/manipulated_sequences/Deepfakes/raw/images/" --is_deepfake

python src/scripts/create_ff_splits.py --split_dir "./data/splits/" --image_path "./data/manipulated_sequences/Deepfakes/masks/images/" --is_deepfake

```