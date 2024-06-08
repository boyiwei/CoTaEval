import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from rouge_score import rouge_scorer, scoring
import evaluate
# # from ipdb import set_trace as bp
import torch
import tqdm
import numpy as np
import pandas as pd
import os
import lib.utils as utils
from lib.eval import eval_ppl, eval_zero_shot
from lib.decoding_intervention import DataPortraitsLogitsProcessor
import timeit
import dataportraits
import subprocess
import re
from evaluate import load


def calc_lcs(ref, pref):
    rougeL = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = rougeL.score(ref, pref)['rougeL'].recall
    return score * len(ref.split(' '))

modeltype2path = {
    'llama2-7b-hf': "/home/samyakg/scratch/nlp_checkpoints/llama-2/Llama-2-7b-hf",
    'llama2-7b-chat-hf': "/home/samyakg/scratch/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf",
    'llama2-70b-chat-hf': "/home/samyakg/scratch/nlp_checkpoints/llama-2/Llama-2-70b-chat-hf"
}
    
def load_models(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        modeltype2path[model_name],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(modeltype2path[model_name], use_fast=False)  
    model.seqlen = model.config.max_position_embeddings
    return model, tokenizer


def run_completions_chat(model_name, data_type, model, tokenizer, testing_chunks, completion_len, agg_res, gt_key, args, hints=False):
    prompts = []
    best_verbatim_matching_comps, matching_sequences= [], []
    best_verbatim_matching_ids, max_lengths, total_lengths = [], [], []
    inference_time = []
    if hints:
        testing_chunks_prompt = testing_chunks['prompt_hint']
    else:
        testing_chunks_prompt = testing_chunks['prompt_nohint']
    for prompt, gt in tqdm.tqdm(zip(testing_chunks_prompt, testing_chunks[gt_key]), total=len(testing_chunks)):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate text completions
        time_start = timeit.default_timer()
        generate_ids = model.generate(inputs.input_ids, max_new_tokens=completion_len, do_sample=True, num_return_sequences=10)
        time_end = timeit.default_timer()
        inference_time.append(time_end - time_start)
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # begin compute time
        # time_start = timeit.default_timer()
        
        
        # time_end = timeit.default_timer()
        # print(f"Time taken for verbatim matching: {time_end - time_start}")
        
        prompt = prompt.replace("<|im_start|>", "").replace("<|im_end|>", "") # For dbrx, because it won't output special token during generation
        outputs = [o.replace(prompt, '') for o in outputs]
        best_verbatim_matching_id, matching_sequence, max_length, total_length = utils.find_common_sequences(outputs, gt)
        best_verbatim_matching = outputs[best_verbatim_matching_id]
        # Compute the length of the Longest Common Subsequence (LCS) between two strings
        
        prompts.append(prompt)
        best_verbatim_matching_comps.append(best_verbatim_matching)
        best_verbatim_matching_ids.append(best_verbatim_matching_id)
        max_lengths.append(max_length)
        total_lengths.append(total_length)
        
        matching_sequences.append(matching_sequence)
        
    df = pd.DataFrame({'prompt': prompts,
                       'best_verbatim_matching': best_verbatim_matching_comps,
                       'matching_sequence': matching_sequences,
                       'max_length': max_lengths,
                       'total_length': total_lengths,
                       })
    if args.use_low_ppl:
        data_type = f'{data_type}_low_ppl'
    if hints:
        df.to_csv(f'res/{data_type}_comp_{model_name}_completion_len_{completion_len}_hints.csv')
    else:
        df.to_csv(f'res/{data_type}_comp_{model_name}_completion_len_{completion_len}_nohints.csv')
    
    return agg_res


def run_evaluation(outputs, prompts, gts):
    rouge = evaluate.load('rouge')
    agg_res = {}
    rouge_1, rouge_l, lcses, prompts = [], [], [], []
    best_rouge1_comps, best_rougeL_comps, best_verbatim_matching_comps, gts, matching_sequences = [], [], [], [], []
    best_rouge1_ids, best_rougeL_ids, best_verbatim_matching_ids = [], [], []
    best_verbatim_matching_ids, max_lengths, total_lengths = [], [], []

    perplexity = load("perplexity", module_type="metric")
    ppls_raw = perplexity.compute(predictions=outputs, model_id='gpt2')
    # bp()
    ppls = ppls_raw['perplexities']
    mean_ppl = ppls_raw['mean_perplexity']

    # time_end = timeit.default_timer()
    # print(f"Time taken for verbatim matching: {time_end - time_start}")

    for output, prompt, gt in zip(outputs, prompts, gts):
        best_verbatim_matching_id, matching_sequence, max_length, total_length = utils.find_common_sequences(outputs, gt)
        results = rouge.compute(predictions=output, references=[gt]*len(output), use_aggregator=False) 
        max_rougeL = max(results['rougeL'])
        max_rouge1 = max(results['rouge1'])
        best_rougeL = output[results['rougeL'].index(max_rougeL)]
        best_rouge1 = output[results['rouge1'].index(max_rouge1)]
        best_verbatim_matching = output[best_verbatim_matching_id]
        # Compute the length of the Longest Common Subsequence (LCS) between two strings
        lcs = calc_lcs(gt, best_rougeL)
    
        prompts.append(prompt)
        rouge_1.append(max_rouge1)
        rouge_l.append(max_rougeL)
        best_rouge1_comps.append(best_rouge1)
        best_rougeL_comps.append(best_rougeL)
        best_verbatim_matching_comps.append(best_verbatim_matching)
        best_rouge1_ids.append(results['rouge1'].index(max_rouge1))
        best_rougeL_ids.append(results['rougeL'].index(max_rougeL))
        best_verbatim_matching_ids.append(best_verbatim_matching_id)
        max_lengths.append(max_length)
        total_lengths.append(total_length)
        gts.append(gt)
        lcses.append(lcs)
        matching_sequences.append(matching_sequence)
    # bp()
    df = pd.DataFrame({'prompt': prompts, 'gt': gts, 'rouge1': rouge_1, 'rougeL': rouge_l, 'lcs': lcses,
                       'best_rouge1': best_rouge1_comps, 'best_rougeL': best_rougeL_comps, 'best_verbatim_matching': best_verbatim_matching_comps,
                       'matching_sequence': matching_sequences,
                       'max_length': max_lengths, 'total_length': total_lengths,
                       'best_rouge1_ids': best_rouge1_ids, 'best_rougeL_ids': best_rougeL_ids, "best_verbatim_matching_ids": best_verbatim_matching_ids, "ppls": ppls})
 
    # agg res
    agg_res['max_rouge1'] = df['rouge1'].max()
    agg_res['max_rougeL'] = df['rougeL'].max()
    agg_res['max_lcs'] = df['lcs'].max()
    # agg_res['min_rouge1'] = df['rouge1'].min()
    # agg_res['min_rougeL'] = df['rougeL'].min()
    agg_res['mean_rouge1'] = df['rouge1'].mean()
    agg_res['mean_rougeL'] = df['rougeL'].mean()
    # agg_res['min_lcs'] = df['lcs'].min()
    agg_res['mean_lcs'] = df['lcs'].mean()
    # agg_res['inference_time'] = sum(inference_time) / len(inference_time)
    agg_res["mean_ppl"] = mean_ppl
    return df
    # output_file = "output/"
    # df.to_csv(f'res/output_res/{data_type}_comp_{model_name}_context_len_{context_len}_completion_len_{completion_len}_intervention_{intervention}_{n}.csv')
        


def run_completions(model_name, data_type, model, tokenizer, testing_chunks, context_len, completion_len, agg_res, args, intervention=False, bloom_filter=None):
    rouge = evaluate.load('rouge')
    rouge_1, rouge_l, lcses, prompts = [], [], [], []
    best_rouge1_comps, best_rougeL_comps, best_verbatim_matching_comps, gts, matching_sequences = [], [], [], [], []
    best_rouge1_ids, best_rougeL_ids, best_verbatim_matching_ids = [], [], []
    best_verbatim_matching_ids, max_lengths, total_lengths = [], [], []
    inference_time = []
    match = re.search(r'\d+', bloom_filter)
    n = int(match[0])
    agg_res = {}
    
    portrait = dataportraits.RedisBFSketch('localhost', 8899, bloom_filter, int(n))
    prior_processor = model._get_logits_processor

    
    for prompt, gt in tqdm.tqdm(zip(testing_chunks['prompt_autocomplete'], testing_chunks['gt_autocomplete']), total=len(testing_chunks)):
        prompt = ' '.join(prompt.split(' ')[-context_len:])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # ========================modified part=========================
        width=2 * int(n) - 1
        if intervention:
            def new_logits_processor(*args, **kwargs):
                prior = prior_processor(*args, **kwargs)
                prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait))
                return prior
            model._get_logits_processor = new_logits_processor
            # ========================modified part=========================
            # Generate text completions
        time_start = timeit.default_timer()
        generate_ids = model.generate(inputs.input_ids, max_new_tokens=completion_len, do_sample=False, num_return_sequences=1)
        time_end = timeit.default_timer()
        print(f"Time taken for inference: {time_end - time_start}")
        inference_time.append(time_end - time_start)
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Compute ROUGE scores
        prompt = prompt.replace("<|im_start|>", "").replace("<|im_end|>", "") # For dbrx, because it won't output special token during generation
        outputs = [o.replace(prompt, '') for o in outputs]
        # begin compute time
        # time_start = timeit.default_timer()
        best_verbatim_matching_id, matching_sequence, max_length, total_length = utils.find_common_sequences(outputs, gt)
        # time_end = timeit.default_timer()
        # print(f"Time taken for verbatim matching: {time_end - time_start}")
        results = rouge.compute(predictions=outputs, references=[gt]*len(outputs), use_aggregator=False) 
        max_rougeL = max(results['rougeL'])
        max_rouge1 = max(results['rouge1'])
        best_rougeL = outputs[results['rougeL'].index(max_rougeL)]
        best_rouge1 = outputs[results['rouge1'].index(max_rouge1)]
        best_verbatim_matching = outputs[best_verbatim_matching_id]
        # Compute the length of the Longest Common Subsequence (LCS) between two strings
        lcs = calc_lcs(gt, best_rougeL)
        
        prompts.append(prompt)
        rouge_1.append(max_rouge1)
        rouge_l.append(max_rougeL)
        best_rouge1_comps.append(best_rouge1)
        best_rougeL_comps.append(best_rougeL)
        best_verbatim_matching_comps.append(best_verbatim_matching)
        best_rouge1_ids.append(results['rouge1'].index(max_rouge1))
        best_rougeL_ids.append(results['rougeL'].index(max_rougeL))
        best_verbatim_matching_ids.append(best_verbatim_matching_id)
        max_lengths.append(max_length)
        total_lengths.append(total_length)
        
        
        gts.append(gt)
        lcses.append(lcs)
        matching_sequences.append(matching_sequence)
    
    if args.use_low_ppl:
        data_type = f'{data_type}_low_ppl'
    df = pd.DataFrame({'prompt': prompts, 'gt': gts, 'rouge1': rouge_1, 'rougeL': rouge_l, 'lcs': lcses,
                       'best_rouge1': best_rouge1_comps, 'best_rougeL': best_rougeL_comps, 'best_verbatim_matching': best_verbatim_matching_comps,
                       'matching_sequence': matching_sequences,
                       'max_length': max_lengths, 'total_length': total_lengths,
                       'best_rouge1_ids': best_rouge1_ids, 'best_rougeL_ids': best_rougeL_ids, "best_verbatim_matching_ids": best_verbatim_matching_ids, "inference_time": inference_time}) 
    df.to_csv(f'res/output_res/{data_type}_comp_{model_name}_context_len_{context_len}_completion_len_{completion_len}_intervention_{intervention}_{n}.csv')
    
    agg_res['max_rouge1'] = df['rouge1'].max()
    agg_res['max_rougeL'] = df['rougeL'].max()
    agg_res['max_lcs'] = df['lcs'].max()
    # agg_res['min_rouge1'] = df['rouge1'].min()
    # agg_res['min_rougeL'] = df['rougeL'].min()
    agg_res['mean_rouge1'] = df['rouge1'].mean()
    agg_res['mean_rougeL'] = df['rougeL'].mean()
    # agg_res['min_lcs'] = df['lcs'].min()
    agg_res['mean_lcs'] = df['lcs'].mean()
    agg_res['inference_time'] = sum(inference_time) / len(inference_time)
    return agg_res, df
   

def main(args):
    num_tests = args.num_tests
    completion_len = args.completion_len
    context_len = args.context_len
    model_name = args.model_name
    n = args.n

    if args.datatype == 'lyrics':
        if args.use_low_ppl:
            testing_chunks = pd.read_csv('eval_data/ppl_lyrics_llama2-7b-hf.csv')
            testing_chunks = testing_chunks.sort_values('ppl').head(num_tests)
        else:
            testing_chunks = pd.read_csv('eval_data/lyrics.csv', nrows=num_tests)
        gt_key = 'lyrics'
    elif args.datatype == 'lyrics_author':
        testing_chunks = pd.read_csv('eval_data/lyrics_author.csv', nrows=num_tests)
        bloom_filter = f'lyrics_author.{n}-{n}.bf'
        gt_key = 'lyrics'
    elif args.datatype == 'lyrics_popular':
        testing_chunks = pd.read_csv('eval_data/lyrics_popular_filtered.csv', nrows=num_tests)
        bloom_filter = f'lyrics_popular.{n}-{n}.bf'
        gt_key = 'lyrics'
    elif args.datatype == 'news':
        if args.use_low_ppl:
            testing_chunks = pd.read_csv(f'eval_data/ppl_{args.datatype}_{args.model_name}.csv')
            testing_chunks = testing_chunks.sort_values('ppl').head(num_tests)
        else:
            testing_chunks = pd.read_csv('eval_data/news.csv', nrows=num_tests)
        bloom_filter = f'news_full.{n}-{n}.bf'
    elif args.datatype == 'books':
        if args.use_low_ppl:
            testing_chunks = pd.read_csv(f'eval_data/ppl_{args.datatype}_{args.model_name}.csv')
            testing_chunks = testing_chunks.sort_values('ppl').head(num_tests)
            bloom_filter = f'books_top.{n}-{n}.bf'
        else:
            testing_chunks = pd.read_csv('eval_data/books.csv', nrows=num_tests)
    else:
        raise NotImplementedError
    model, tokenizer = load_models(model_name)

    
    agg_res = {}
    agg_res['model'] = model_name
    agg_res['num_tests'] = num_tests
    agg_res['context_len'] = context_len
    agg_res['completion_len'] = completion_len
    if model_name in ['llama2-7b-hf']:
        agg_res = run_completions(model_name, args.datatype, model, tokenizer, testing_chunks, context_len, completion_len, agg_res, args, intervention=args.intervention, bloom_filter=bloom_filter)
    else:
        assert args.intervention == False
        agg_res = run_completions_chat(model_name, args.datatype, model, tokenizer, testing_chunks, completion_len, agg_res, gt_key, args, hints=args.hints)
    if os.path.exists(args.log_dir):
        df = pd.read_csv(args.log_dir)
        df = pd.concat([df, pd.DataFrame(agg_res, index=[0])])
        df.to_csv(args.log_dir, index=False)
    else:
        df = pd.DataFrame(agg_res, index=[0])
        df.to_csv(args.log_dir, index=False)
    
    # evaluate the model's PPL
    if args.eval_ppl:
        ppl_test = eval_ppl(args, model, tokenizer, device=model.device)
        print(f"wikitext perplexity {ppl_test}")
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.model_name}_{args.datatype}.txt")
        if not os.path.exists(save_filepath):
            with open(save_filepath, 'w') as f:
                print("model\tintervention\tdataset\tcontext_len\tcompletion_len\tmetric\tscore", file=f, flush=True)
                print(
                    f"{args.model_name}\t{args.intervention}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\tPPL\t{ppl_test:.4f}",
                    file=f,
                    flush=True,
                )
                print(
                    f"{args.model_name}\t{args.intervention}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\tInference Time\t{agg_res['inference_time']:.4f}",
                    file=f,
                    flush=True,
                )
        else:
            with open(save_filepath, 'a') as f:
                print(
                    f"{args.model_name}\t{args.intervention}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\tPPL\t{ppl_test:.4f}",
                    file=f,
                    flush=True,
                )
                print(
                    f"{args.model_name}\t{args.intervention}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\tInference Time\t{agg_res['inference_time']:.4f}",
                    file=f,
                    flush=True,
                )
    
    # evaluate the model's utility
    if args.eval_zero_shot:
        accelerate=False
        task_list = ["boolq", "rte","hellaswag","winogrande","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(modeltype2path[args.model_name], model, tokenizer, task_list, num_shot, accelerate, limit=200)
        print("********************************")
        print("zero_shot evaluation results")
        sum_acc = 0
        with open(save_filepath, "a") as f:
            for k,v in results['results'].items():
                print(f"{args.model_name}\t{args.intervention}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\t{k}\t{v['acc']:.4f}", file=f, flush=True)
                sum_acc += v['acc']
            print(f"{args.model_name}\t{args.intervention}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\taveraged\t{sum_acc/len(task_list):.4f}", file=f, flush=True)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tests", type=int, default=100)
    parser.add_argument("--completion_len", type=int, default=500)
    parser.add_argument("--context_len", type=int, default=100)
    parser.add_argument("--model_name", type=str, default='llama2-7b-hf')
    parser.add_argument("--datatype", type=str, default='lyrics_author')
    parser.add_argument("--log_dir", type=str, default='res/log_completion.csv')
    parser.add_argument("--hints", type=bool, default=False)
    parser.add_argument("--intervention", action="store_true")
    parser.add_argument("--save", type=str, default='res/utility_res')
    parser.add_argument("--use_low_ppl", action='store_true')
    parser.add_argument("--eval_ppl", action='store_true')
    parser.add_argument("--eval_zero_shot", action='store_true')
    parser.add_argument("--n", type=int, default=50)
    
    args = parser.parse_args()
    main(args)
