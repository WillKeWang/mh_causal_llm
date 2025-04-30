#!/bin/bash
#SBATCH --job-name=mk_globem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mk_globem_%j.log
#SBATCH --error=logs/mk_globem_%j.err

module load conda/3
conda activate healthllm_env

python gen_dataset.py \
       --dataset globem \
       --task stress \
       --in_dir  data/globem_raw \
       --out_dir datasets \
       --val_split 0.1