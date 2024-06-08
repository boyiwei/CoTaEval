#!/bin/bash
#SBATCH --job-name=cre_bf   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=20G              # required memory
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)


export HF_HOME=""
export HF_DATASETS_CACHE=""
export TRANSFORMERS_CACHE=""

module purge
module load anaconda3/2023.3
conda activate copyright

cd data-portraits

python create_bf.py --n 6 --datatype "newsqa" --tokenized

