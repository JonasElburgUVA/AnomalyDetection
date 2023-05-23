#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=predict_anomalies_FFHQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=job_logs/predict_anomalies_FFHQ_slurm_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

source activate vqvae

# run from user home directory
python -u AnomalyDetection/src/DeepFake/pred.py -i data/FaceForensics/test_set -o data/output