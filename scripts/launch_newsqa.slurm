#!/bin/bash
#SBATCH --job-name=7bnews  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=30G   # memory per cpu-core
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)

export HF_HOME=""
export HF_DATASETS_CACHE=""
export TRANSFORMERS_CACHE=""

module purge
module load anaconda3/2023.3
# conda activate jailbreak
conda activate copyright



completion_len=200
model='llama2-7b-chat-hf'

n=1000


for datatype in "newsqa" 
do
    python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --datatype $datatype   --eval_zero_shot --eval_general --eval_infringement
done
