#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=install_faceForensics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=job_logs/faceForensics_download_slurm_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

source activate vqvae

srun python -u scripts/download_faceforensics.py data/ --server EU2 -d original -c raw

srun python -u scripts/download_faceforensics.py data/ --server EU2 -d Deepfakes -c raw
srun python -u scripts/download_faceforensics.py data/ --server EU2 -d Deepfakes -t masks -c raw

# srun python -u scripts/download_faceforensics.py data/ --server EU2 -d DeepFakeDetection_original -c raw
# srun python -u scripts/download_faceforensics.py data/ --server EU2 -d DeepFakeDetection -c raw
# srun python -u scripts/download_faceforensics.py data/ --server EU2 -d DeepFakeDetection -t masks -c raw
