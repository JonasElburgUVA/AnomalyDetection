#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=extract_faceForensics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=job_logs/faceForensics_extract_slurm_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

source activate vqvae

srun python -u scripts/extract_images_from_videos.py --data_path data/original_sequences/youtube/raw/videos/ --dataset original --compression c0