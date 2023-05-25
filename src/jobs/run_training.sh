#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=training_vqvae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=32000M
#SBATCH --output=training_slurm_%A.out

module purge

module load 2022
module load Anaconda3/2022.05

source activate vae

srun python DeepFake/train.py
