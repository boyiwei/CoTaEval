#!/bin/bash
#SBATCH --job-name=7bnews  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=30G   # memory per cpu-core
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=wby@princeton.edu
#SBATCH --partition=pli-c

export HF_HOME="/scratch/gpfs/bw1822/cache"
export HF_DATASETS_CACHE="/scratch/gpfs/bw1822/cache"
export TRANSFORMERS_CACHE="/scratch/gpfs/bw1822/cache"

module purge
module load anaconda3/2023.3
# conda activate jailbreak
conda activate copyright



completion_len=200
model='llama2-7b-chat-hf'
# model='llama2-7b-chat-hf_newsqa_STEP1000'
# model='l'

n=1000



# for cont in 200 #10 20 50 100 200 300 400 500 
# do
#     python main.py --model_name $model --num_test $n --context_len $cont --datatype $datatype --completion_len $completion_len --use_low_ppl # --no_context
#     python main.py --model_name $model --num_test $n --context_len $cont --datatype $datatype --completion_len $completion_len --use_low_ppl --no_context
#     # python main.py --model_name $model --num_test $n --context_len $cont --datatype $datatype --completion_len $completion_len --use_low_ppl --intervention "sys_prompt-sys_a" #--no_context
#     # python main.py --model_name $model --num_test $n --context_len $cont --datatype $datatype --completion_len $completion_len --use_low_ppl --intervention "sys_prompt-sys_b" #--no_context
#     # python main.py --model_name $model --num_test $n --context_len $cont --datatype $datatype --completion_len $completion_len --use_low_ppl --intervention "sys_prompt-sys_c" #--no_context
# done
# for model in 'llama2-7b-chat-hf' 'llama2-13b-chat-hf'
# do
# for i in 1 2 3
# do
# for model in  "llama2-7b-hf_booksum_STEP1000" "llama2-7b-hf_booksum_STEP3000"
# for model in "llama2-7b-hf_newsqa_grad_diff_2e-06_31" "llama2-7b-hf_newsqa_grad_ascent_2e-06_31" "llama2-7b-hf_newsqa_KL_2e-06_31"
# for model in 'llama2-7b-hf_newsqa_STEP1000'
# do
for datatype in "newsqa" # "newsqa"
do
    # Intervention=None
    # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --datatype $datatype  --use_low_ppl  --eval_zero_shot --no_context
    python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --datatype $datatype   --eval_zero_shot --eval_general --eval_infringement
    # # # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "none_greedy" --datatype $datatype  --use_low_ppl  --eval_zero_shot --no_overwrite

    # # # # ## #Intervention=Topk
    # # # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "top_k" --datatype $datatype --use_low_ppl  --eval_zero_shot --std 0.5 --eval_general
    # # # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "top_k" --datatype $datatype --use_low_ppl  --eval_zero_shot --std 1 --eval_general
    # for i in 1 2 3 4 5
    # do
    # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "top_k" --datatype $datatype --use_low_ppl  --eval_zero_shot --std 3 --eval_general --no_overwrite
    # done

    # # # # # #Intervention=Sys_prompt
    # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "sys_prompt-dbrx" --datatype $datatype --use_low_ppl  --eval_zero_shot --eval_general
    # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "sys_prompt-bing" --datatype $datatype --use_low_ppl  --eval_zero_shot --eval_general
    # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "sys_prompt-copilot" --datatype $datatype --use_low_ppl  --eval_zero_shot --eval_general
    # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "sys_prompt-sys_a" --datatype $datatype --use_low_ppl  --eval_zero_shot --eval_general
    # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "sys_prompt-sys_b" --datatype $datatype --use_low_ppl  --eval_zero_shot --eval_general
    # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "sys_prompt-sys_c" --datatype $datatype --use_low_ppl  --eval_zero_shot --eval_general
    #  --eval_general
    # # # #Intervention=mem_free-consecutive
    # # for ngram in 50
    # # do 
    # # cd data-portraits
    # # python easy_redis.py --shutdown
    # # python easy_redis.py --start-from-dir /scratch/gpfs/bw1822/bloom_filters/$datatype/$ngram
    # # cd ..
    # # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "mem_free-consecutive" --datatype $datatype --use_low_ppl  --eval_zero_shot --n $ngram
    # # done

    # ### Intervention=mem_free_tokenized-consecutive and mem_free_new
    # for ngram in 12 24
    # do 
    # cd data-portraits
    # python easy_redis.py --shutdown
    # python easy_redis.py --start-from-dir /scratch/gpfs/bw1822/bloom_filters/${datatype}_tokenized/$ngram
    # cd ..
    # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "mem_free_tokenized-consecutive" --datatype $datatype --use_low_ppl --eval_general --eval_zero_shot --n $ngram
    # # for skip_tokens in 1 5 10 15 20
    # # do
    # # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "mem_free_new" --datatype $datatype --use_low_ppl  --eval_zero_shot --n $ngram  --skip_tokens $skip_tokens --eval_general
    # # done
    # done
    # # ### Intervention=mem_free-non_consecutive

    # # # #### cad
    # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "cad" --datatype $datatype --use_low_ppl  --eval_zero_shot --context_aware_decoding_alpha 0 --eval_general
    # python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "cad" --datatype $datatype --use_low_ppl  --eval_zero_shot --context_aware_decoding_alpha 1 --eval_general

     # # #### unlearning
    #  python main.py --model_name $model --num_test $n --context_len 200 --completion_len 200 --intervention "unlearning-gradient_ascent" --datatype $datatype --use_low_ppl  --eval_zero_shot


# done
done
