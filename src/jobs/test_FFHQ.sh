#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=predict_anomalies_FFHQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=job_logs/test_FFHQ_slurm_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

source activate vqvae

python -u AnomalyDetection/src/DeepFake/pred.py --data_dir data --cp_dir data/checkpoints -o data/output -dataset ffhq --split test --ld 7
python -u AnomalyDetection/src/DeepFake/eval.py --pred_dir data/output --output_dir data/output --dataset ffhq --split test