# Copyright Takedown Evaluation (CoTaEval)

This repository provides an original implementation of *Evaluating Copyright Takedown Methods for Language Models* by Boyi Wei*, Weijia Shi*, Yangsibo Huang*, Noah A. Smith, Chiyuan Zhang, Luck Zettlemoyer, Kai Li and Peter Henderson.

## Content

## Setup

You can use the following instruction to create conda environment
```bash
conda env create -f environment.yml
```
Please notice that you need to specify your environment path inside ``environment.yml``

After initializing the environment, we need to install a modified ``transformers`` library for R-CAD deployment. You can use the following instruction for installation:

```bash
cd cad/transformers_cad
pip install -e .
```

## Evaluate Infringement

### Quick Start
The main entry for infringement evaluation is ``main.py``.

For example, in RAG setting under news articles domain, if we want to evaluate the infringement risk and utility of top-k perturbation with std=3, using Llama-2-7B-chat model, we can use the following command:
```bash
python main.py --model_name llama2-7b-chat-hf --num_test 1000 --context_len 200 --completion_len 200 --datatype newsqa --intervention top_k --std 3   --eval_zero_shot --eval_general --eval_infringement
```
### Argument Details
Important parameters are:
1. ``--model_name``: To specify the model for evaluation. The model name and the path for loading the model can be specified via ``modeltype2path``: Dictionary in ``main.py``
2. ``--datatype``: To specify the evaluation domain. Available options are ``newsqa`` (for news articles domain) and ``booksum`` (for books domain).
3. ``--num_test``: Number of examples to test for infringement evaluation. For news articles, we set 1000 as default. For books, we set 500 as default.
4. ``--context_len``: The length of hint. We set 200 as default.
5. ``--completion_len``: The maximum number of generated tokens. We set 200 as default
6. ``--eval_infringement``: If true, perform infrigement test.
9. ``--intervention``: Intervention methods. Available options are ``none, memfree_tokenized_consecutive, top_k, sys_prompt-dbrx, sys_prompt-bing, sys_prompt-copilot, sys_prompt-sys_a, sys_prompt-sys_b, sys_prompt_sys_c, cad``. ``memfree_tokenized_consecutive`` corresponds to Memfree Decoding, `sys_prompt-*` corresponds to the system prompt methods, and `*` refers to which type of system prompt are we using. ``sys_a``, ``sys_b``, and ``sys_c`` correpond to the three manually created system prompt in Appendix C.1
13. ``--no_context``: If true, we don't provide the context in the infringement and utility evaluation. For memorization settings.
14. 
### Specific Takedown Methods Implementation
#### MemFree
When evaluating Memfree, please make sure that the redis has been started. For example, if we want to evaluate Memfree with $n$-gram size equals 6, we can use the following code:
```bash
cd data-portraits
python easy_redis.py --shutdown
python easy_redis.py --start-from-dir your/path/to/bloom_filter/${datatype}_tokenized/$ngram
cd ..
python main.py --model llama2-7b-chat-hf --num_test 1000 --context_len 200 --completion_len 200 --datatype newsqa --intervention memfree_tokenized_consecutive --n 6 --eval_zero_shot --eval_general --eval_infringement
```
For details on how to create the bloom filter using Data Portraits, please refer to 

1.  ``--n``: The $n$-gram stored in the bloom filter for Memfree decoding
#### Top-K perturbation

1.  ``--std``: The std of the Gaussian noise in Top-k perturbation

#### R-CAD
12. ``--context_aware_decoding_alpha``: The weight of adjustment $\alpha$ in R-CAD.


#### Unlearning

For unlearning methods, we use the framework provided in [TOFU](https://github.com/locuslab/tofu). When perform unlearning, you need to use our dataset as forget set and retain set. After having the unlearned the model, we can evaluate their performence following the procedure above, with ``--intervention none``.


### Adding custom takedown methods

## Evaluate Utility

7. ``--eval_zero_shot``: If true, evaluate the blocklisted and indomain utility
8. ``--eval_general``: If true, evaluate the MMLU score
   
We use the [FastChat](https://github.com/lm-sys/FastChat) to compute the MT-bench score. The code is in ``eval/FastChat_new``. To run MT-Bench, use the following code
``bash
cd eval/FastChat_new/fastchat/llm_judge

python gen_model_answer.py --model-path your/path/to/the/model --model-id llama2-7b-chat-hf_none --intervention none
``

You can specify the intervention type following ``--intervention``

After having the model's answer, we need to run ``python gen_judgment.py --model-list <the model list you want to eval> --parallel 5`` to generate the judgement. Then we can use ``python show_result.py`` to show the MT-Bench score for all the models.

## Evaluate efficiency

The main function for evaluating the efficiency is ``main_efficiency.py``. The key difference between ``main_efficiency.py`` and ``main.py`` is in ``main_efficiency.py`` we set ``max_new_tokens=min_new_toknes=200`` for fair comparison. For example, to test the effiency of top-k perturbation, we can use the following codes:
```bash
python main_efficiency.py --model_name llama2-7b-chat-hf --num_test 1000 --context_len 200 --completion_len 200 --datatype newsqa --intervention top_k --std 3
```

## Result Analysis
### Infringenet analysis
#### Compute all the metrics
To facilitate the win rate computation, we need to reformat the output file, and compute all the 8 metrics (ROUGE-1, ROUGE-L, LCS(character), LCS(word), ACS(word), Levenshitein Distance, Semantic Similarity, MinHash Similarity) for each example. We use ``process.py`` to do so. After running ``main.py``, it will output a ``.csv`` file in ``res/output_res``, which contains the raw output of the infringement test. After having the file in ``res/output_res``, we can process the raw output to another ``.csv`` file with all metrics. For example, if we have a raw output file in ``res/output_res/newsqa_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_none_no_context_False.csv``, we can process it by using:
```bash
python process.py --input_dir res/output_res --output_dir res/output_res_processed --file_name newsqa_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_none_no_context_False.csv
```
It output a ``.csv`` file to ``res/output_res_processed``, which contains all the 8 metrics for infringement evaluation. We also need to notice that sometimes there are few examples that will lead the model output nothing in the vanilla case, we call them as "invalid ids". For fair comparison purposes, we also need to zero out all the "invalid ids" for different takedown methods under the same model and domain.
#### Compute the win rate

After having the processed `.csv` file, we can use ``winrate_compute.py`` to compute the winrate for different intervention methods. For example, if we want to compute the win rate for different takedown methods for Llama2-7B-chat model, under news articles domain and RAG scenario, we can use the following command:
```bash
python winrate_compute.py --data_type news --model_name llama2_7b_chat --scenario rag
```
It will output ``win_rate_memorization.csv``, with per-metric and average winrate for each takedown methods inside it.

### Utility and Efficiency analysis
The data of utility and efficiency will be logged in ``res/utility_res``, where you can see the F1/ROUGE score, MMLU score, tokens/sec in ``log_*.txt``, here ``*`` refers to the model name.




   