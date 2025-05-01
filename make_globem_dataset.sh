#!/bin/bash
#SBATCH --job-name=mk_globem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mk_globem_%j.log
#SBATCH --error=logs/mk_globem_%j.err

module load conda/3
source "$(conda info --base)/etc/profile.d/conda.sh"

# create just once
conda create -y -n healthllm_env python=3.9
conda activate healthllm_env

# install project packages
pip install --upgrade pip
pip install -r requirements.txt  

python make_globem_dataset.py \
       --root   /groups/xx2489_gp/kw3215/Datasets/globem \
       --subtask depression \
       --out_dir datasets