#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=64000M
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=cs
#SBATCH --partition=cs
#SBATCH --time=4:00:00
nvidia-smi --list-gpus
nvidia-smi --query-gpu=memory.total --format=csv
module load python/3.11
cd ~/compute
source gptenv/bin/activate
cd ~/minGPT
python3 train.py