# Copyright Takedown Evaluation (CoTaEval)

This repository provides an original implementation of *Evaluating Copyright Takedown Methods for Language Models* by Boyi Wei*, Weijia Shi*, Yangsibo Huang*, Noah A. Smith, Chiyuan Zhang, Luck Zettlemoyer, Kai Li and Peter Henderson.

# 1. Setup

## 1.1 Environments
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
## 1.2 Create Bloom-filter (for MemFree)
1. Go to ``data-portraits/``, install the library:
    ```sh
    pip install -e . #install editable package
    ```

2. Install Redis and Redis-Bloom
   1. Install Redis: https://redis.io/docs/install/install-redis/install-redis-from-source/
        ```sh
        wget https://download.redis.io/redis-stable.tar.gz

        tar -xzvf redis-stable.tar.gz
        cd redis-stable
        make
        ```
        ```sh
        make install PREFIX="$(pwd)" # install to redis-stable/bin
        ```
   2. Install Redis-Bloom (For successful compilation, the GNU version should >4.0)
        ```
        git clone https://github.com/RedisBloom/RedisBloom.git
        cd RedisBloom
        git checkout tags/v2.4.3 
        git submodule update --init --recursive
        make -j 4
        cd ..
        ```
   3. Add the path where your redis is intalled to your ``.bashrc`` file. Add ``export PATH=$PATH:/your/path/to/data-portraits/redis-stable/bin`` at the end of your ``.bashrc`` file.
   4. Restart bash, try to run command ```redis-server``` on bash to see whether it will work.
3.  Create the Bloom Filter (.bf file)
   1. go to ``data-portraits`` folder
   2. use ``create_bf.py`` to create the .bf file, specify the dataset and the number of example you want to feed into bloom filter. We also provide a script in ``scripts/create_bf.slurm`` for reference.
4. Before running Data Portraits, we need to initialize redis and load the dataset. For example, load the Bloom filter which stores 6-gram news articles, and is stored in ``bloom_filters/newsqa_tokenized/6``:
   ```
   python easy_redis.py --start-from-dir bloom_filters/newsqa_tokenized/6
   ```
   If Redis-Bloom is successfully installed it won't raise error.
5. When finish the experiment. Use
   ```
   python easy_redis.py --shutdown
   ```
   to shutdown the redis daemon server.

# 2. Evaluate Infringement and utility

The main entry for infringement/utility evaluation is ``main.py``. Important parameters are:
Important parameters are:
1. ``--model_name``: To specify the model for evaluation. The model name and the path for loading the model can be specified via ``modeltype2path``: Dictionary in ``main.py``
2. ``--datatype``: To specify the evaluation domain. Available options are ``newsqa`` (for news articles domain) and ``booksum`` (for books domain).
3. ``--num_test``: Number of examples to test for infringement evaluation. For news articles, we set 1000 as default. For books, we set 500 as default.
4. ``--context_len``: The length of hint. We set 200 as default.
5. ``--completion_len``: The maximum number of generated tokens. We set 200 as default
6. ``--eval_infringement``: If true, perform infrigement test.
7. ``--eval_zero_shot``: If true, evaluate the blocklisted and indomain utility
8. ``--eval_general``: If true, evaluate the MMLU score
9. ``--intervantion``: Intervention methods. Available options are ``none, memfree_tokenized_consecutive, top_k, sys_prompt-dbrx, sys_prompt-bing, sys_prompt-copilot, sys_prompt-sys_a, sys_prompt-sys_b, sys_prompt_sys_c, cad``. ``memfree_tokenized_consecutive`` corresponds to Memfree Decoding, `sys_prompt-*` corresponds to the system prompt methods, and `*` refers to which type of system prompt are we using. ``sys_a``, ``sys_b``, and ``sys_c`` correpond to the three manually created system prompt in Appendix C.1
10. ``--std``: The std of the Gaussian noise in Top-$k$ perturbation
11. ``--n``: The $n$-gram stored in the bloom filter for Memfree decoding
12. ``--context_aware_decoding_alpha``: The weight of adjustment $\alpha$ in R-CAD.
13. ``--no_context``: If true, we don't provide the context in the infringement and utility evaluation. For memorization settings.

For example, in RAG setting under news articles domain, if we want to evaluate the infringement risk and utility of top-$k$ perturbation with std$=3$, using Llama-2-7B-chat model, we can use the following code:
```bash
python main.py --model_name llama2-7b-chat-hf --num_test 1000 --context_len 200 --completion_len 200 --datatype newsqa --intervention top_k --std 3   --eval_zero_shot --eval_general --eval_infringement
```

When evaluating Memfree, please make sure that the redis has been started. For example, if we want to evaluate Memfree with $n$-gram size equals 6, we can use the following code:
```bash
cd data-portraits
python easy_redis.py --shutdown
python easy_redis.py --start-from-dir your/path/to/bloom_filter/${datatype}_tokenized/$ngram
cd ..
python main.py --model llama2-7b-chat-hf --num_test 1000 --context_len 200 --completion_len 200 --datatype newsqa --intervention memfree_tokenized_consecutive --n 6 --eval_zero_shot --eval_general --eval_infringement
```

For unlearning methods, we use the framework provided in [TOFU](https://github.com/locuslab/tofu). When perform unlearning, you need to use our dataset as forget set and retain set. After having the unlearned the model, we can evaluate their performence following the procedure above, with ``--intervention none``.


# 3. Evaluate MT-Bench Score
We use the [FastChat](https://github.com/lm-sys/FastChat) to compute the MT-bench score. The code is in ``eval/FastChat_new``. To run MT-Bench, use the following code
``bash
cd eval/FastChat_new/fastchat/llm_judge

python gen_model_answer.py --model-path your/path/to/the/model --model-id llama2-7b-chat-hf_none --intervention none
``

You can specify the intervention type following ``--intervention``

After having the model's answer, we need to run ``python gen_judgment.py --model-list <the model list you want to eval> --parallel 5`` to generate the judgement. Then we can use ``python show_result.py`` to show the MT-Bench score for all the models.

# 4. Evaluate the efficiency

The main function for evaluating the efficiency is ``main_efficiency.py``. The key difference between ``main_efficiency.py`` and ``main.py`` is in ``main_efficiency.py`` we set ``max_new_tokens=min_new_toknes=200`` for fair comparison. For example, to test the effiency of top-$k$ perturbation, we can use the following codes:
```bash
python main_efficiency.py --model_name llama2-7b-chat-hf --num_test 1000 --context_len 200 --completion_len 200 --datatype newsqa --intervention top_k --std 3
```

# 5. Analysis
## 5.1 Infringenet analysis
### 5.1.1 Compute all the metrics
To facilitate the win rate computation, we need to reformat the output file, and compute all the 8 metrics (ROUGE-1, ROUGE-L, LCS(character), LCS(word), ACS(word), Levenshitein Distance, Semantic Similarity, MinHash Similarity) for each example. We use ``process.py`` to do so. After running ``main.py``, it will output a ``.csv`` file in ``res/output_res``, which contains the raw output of the infringement test. After having the file in ``res/output_res``, we can process the raw output to another ``.csv`` file with all metrics. For example, if we have a raw output file in ``res/output_res/newsqa_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_none_no_context_False.csv``, we can process it by using:
```bash
python process.py --input_dir res/output_res --output_dir res/output_res_processed --file_name newsqa_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_none_no_context_False.csv
```
It output a ``.csv`` file to ``res/output_res_processed``, which contains all the 8 metrics for infringement evaluation. We also need to notice that sometimes there are few examples that will lead the model output nothing in the vanilla case, we call them as "invalid ids". For fair comparison purposes, we also need to zero out all the "invalid ids" for different takedown methods under the same model and domain.
### 5.1.2 Compute the winrates

After having the processed `.csv` file, we can use ``winrate_compute.py`` to compute the winrate for different intervention methods. For example, if we want to compute the win rate for different takedown methods for Llama2-7B-chat model, under news articles domain and RAG scenario, we can use the following command:
```bash
python winrate_compute.py --data_type news --model_name llama2_7b_chat --scenario rag
```
It will output ``win_rate_memorization.csv``, with per-metric and average winrate for each takedown methods inside it.

## 5.2 Utility and Efficiency analysis
The data of utility and efficiency will be logged in ``res/utility_res``, where you can see the F1/ROUGE score, MMLU score, tokens/sec in ``log_*.txt``, here ``*`` refers to the model name.




   