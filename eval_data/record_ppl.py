import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
import torch
import os
from tqdm import tqdm
import pandas as pd
# from ipdb import set_trace as bp
import numpy as np
from pathlib import Path
import sys
sys.path.append("./eval_data/booksum/raw_version")
from preprocess import read_jsonl

def calc_ppl(model, tokenizer, dataset, args):
    max_length = model.config.max_length
    ppls = []
    prompts = []
    gts = []
    
    if args.datatype == 'books':
        key = 'snippet'
    elif args.datatype == 'news':
        key = 'article'
    elif 'newsqa' in args.datatype:
        key = 'story_text'
    elif args.datatype == 'lyrics':
        key = 'lyrics'
    elif 'booksum' in args.datatype:
        key = 'document'        

    res = []
    Path(f"eval_data/{args.datatype}").mkdir(parents=True, exist_ok=True)
    for example in tqdm(dataset):
        words = example[key].split(' ')
        if len(words) < args.context_len + args.completion_len:
            continue
        if 'newsqa' in args.datatype:
            answer_token_range = example['answer_token_ranges']
            if ',' in answer_token_range:
                continue
            text = example['story_text']
            text_split = text.split(' ')
            slice_obj = slice(*map(int, answer_token_range.split(':')))
            example['answer'] = ' '.join(text_split[slice_obj])
            
        prompt = ' '.join(words[:args.context_len])
        gt =  ' '.join(words[args.context_len:args.context_len+args.completion_len])
        input_ids = tokenizer(example[key], return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss  # Ensure this is correctly calculated
        ppl = torch.exp(neg_log_likelihood)
        
        example['ppl'] = ppl.item()
        example['gt_autocomplete'] = gt
        example['prompt_autocomplete'] = prompt
        # bp()
        res.append(example)
        # if len(res) % 100 == 0:
        #     df = pd.DataFrame(res)
        #     df.to_csv(f'eval_data/{args.datatype}/ppl_{args.datatype}_{args.model_name}.csv', index=False)
    df = pd.DataFrame(res)
    df.to_csv(f'eval_data/{args.datatype}/ppl_{args.datatype}_{args.model_name}.csv', index=False)


    # # read csv
    # df = pd.read_csv(f'eval_data/{args.datatype}/ppl_{args.datatype}_{args.model_name}.csv')

    # save low_ppl, sort df ppl column and select top 500
    df = df.sort_values(by='ppl')
    # bp()
    df.to_csv(f'eval_data/{args.datatype}/ppl_{args.datatype}_{args.model_name}_sorted.csv', index=False)
    return ppls  # Corrected to return the list of perplexities

def load_models(model_name):
    if model_name == 'llama2-7b-hf':
        model_path = "/home/samyakg/scratch/nlp_checkpoints/llama-2/Llama-2-7b-hf"
        tokenizer_path = "/home/samyakg/scratch/nlp_checkpoints/llama-2/Llama-2-7b-hf"
    elif model_name == 'llama2-7b-chat-hf':
        model_path = "/home/samyakg/scratch/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
        tokenizer_path = "/home/samyakg/scratch/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
    else:
        model_path = "/home/samyakg/scratch/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"

    # for weijia, hardcode the path
    if not os.path.exists(model_path):
        model_path = tokenizer_path = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    return model, tokenizer


def main(args):
    model_name = args.model_name

    if args.datatype == 'lyrics':
        dataset = pd.read_csv('eval_data/lyrics_author_large.csv').to_dict('records')
    elif args.datatype == 'news':
        dataset = load_from_disk('/scratch/gpfs/yangsibo/cache/datasets_cnn_dailymail')
        dataset = dataset['train'][:50000]
    elif args.datatype == 'newsqa':
        dataset = load_dataset("boyiwei/newsqa")['train']
    elif args.datatype == 'newsqa_test':
        dataset = load_dataset("boyiwei/newsqa")['test']
    elif args.datatype == 'books':
        # dataset = load_dataset("swj0419/BookMIA")
        dataset = load_from_disk('/scratch/gpfs/yangsibo/cache/datasets_bookmia')
        indices = np.where(np.asarray(dataset['train']['label']) == 1)[0]
        dataset = dataset['train'].select(indices)
    elif args.datatype == "booksum":
        dataset = read_jsonl("eval_data/booksum/raw_version/train.jsonl")
    elif args.datatype == 'booksum_test':
        dataset = read_jsonl("eval_data/booksum/raw_version/test.jsonl")
    else:
        raise NotImplementedError
    model, tokenizer = load_models(model_name)

    calc_ppl(model, tokenizer, dataset, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama2-7b-hf')
    parser.add_argument("--datatype", type=str, default='booksum_train')
    parser.add_argument("--context_len", type=int, default=200)
    parser.add_argument("--completion_len", type=int, default=200)
    args = parser.parse_args()
    main(args)
