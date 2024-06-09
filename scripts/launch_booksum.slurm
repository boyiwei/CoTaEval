#!/bin/bash
#SBATCH --job-name=books7b  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=20G   # memory per cpu-core
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)

export HF_HOME=""
export HF_DATASETS_CACHE=""
export TRANSFORMERS_CACHE=""

module purge
module load anaconda3/2023.3
conda activate copyright



completion_len=200
model='llama2-7b-chat-hf'

n=500



for datatype in "booksum"
do
    python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --datatype $datatype  --use_low_ppl  --eval_zero_shot --eval_general

done
