import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import torch
import tqdm
import numpy as np
import pandas as pd
import os
from lib.decoding_intervention import DataPortraitsLogitsProcessor, TopKPerturbationLogitsProcessor, DataPortraitsSkipLogitsProcessor
import timeit
import dataportraits
import re
from lib.prompt_utils import apply_prompt_template



modeltype2path = {
    'llama2-7b-hf': "meta-llama/Llama-2-7b-hf",
    'llama2-7b-chat-hf': "meta-llama/Llama-2-7b-chat-hf",
    'llama2-70b-chat-hf': "meta-llama/Llama-2-70b-chat-hf",
    'llama3-8b-chat-hf': "meta-llama/Meta-Llama-3-8B-Instruct",
    'dbrx': "databricks/dbrx-instruct",
}


def load_models(model_name):
    if 'llama2' in model_name:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
        if model_name in ['llama2-70b-chat-hf']:
            model = AutoModelForCausalLM.from_pretrained(
            modeltype2path[model_name],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        else:
            model = AutoModelForCausalLM.from_pretrained(
            modeltype2path[model_name],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    elif 'llama3' in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            modeltype2path[model_name],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=False)

    elif model_name in ['dbrx']:
        model = AutoModelForCausalLM.from_pretrained(modeltype2path['dbrx'], device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True, token="hf_YOUR_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(modeltype2path['dbrx'], trust_remote_code=True, token="hf_YOUR_TOKEN")
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.pad_token_id = tokenizer.eos_token_id # TODO noqa
    else:
        model = AutoModelForCausalLM.from_pretrained(
            modeltype2path[model_name],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(modeltype2path[model_name], use_fast=False)  
    model.seqlen = model.config.max_position_embeddings
    return model, tokenizer


def run_completions(model_name, model, tokenizer, testing_chunks, context_len, completion_len, args, bloom_filter=None, rewrite_text=[]):
    token_sec_list = [], [], [], []
    prior_processor = model._get_logits_processor
    if args.intervention != 'cad':
        model.generation_config.context_aware_decoding_alpha = None # Here we add this to avoid error for the non-cad situation.
    model.generation_config.mem_free_new = False
    for i, (prompt, gt) in tqdm.tqdm(enumerate(zip(testing_chunks['prompt_autocomplete'], testing_chunks['gt_autocomplete'])), total=len(testing_chunks)):
        prompt = ' '.join(prompt.split(' ')[-context_len:])
        
        if 'mem_free' in args.intervention:
            bf_is_tokenized = "tokenized" in args.intervention
            choice = args.intervention.split('-')[-1]
            context = f"Context: {prompt + ' ' + gt}\n"
            if model_name in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf', 'llama3-8b-chat-hf']:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context)[0]
            elif 'llama3' in model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context, model="llama3")[0]
            match = re.search(r'\d+', bloom_filter)
            n = int(match[0])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            portrait = dataportraits.RedisBFSketch('localhost', 8899, bloom_filter, int(n))
            if choice == 'consecutive':
                if "tokenized" in args.intervention:
                    width = 2 * int(n) - 6
                else:
                    width = 2 * int(n) - 1 
                def new_logits_processor(*args, **kwargs):
                    prior = prior_processor(*args, **kwargs)
                    prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait, bf_is_tokenized=bf_is_tokenized, tokenized_prompt=inputs, n=int(n), consecutive=True))
                    return prior
            else:
                width = 3 * int(n)
                acs_threshold = args.acs_threshold
                def new_logits_processor(*args, **kwargs):
                    prior = prior_processor(*args, **kwargs)
                    prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait, bf_is_tokenized=bf_is_tokenized, tokenized_prompt=inputs, n=int(n), consecutive=False, acs_threshold=acs_threshold))
                    return prior
            model._get_logits_processor = new_logits_processor
            # Generate text completions
            time_start = timeit.default_timer()
            generate_ids = model.generate(inputs.input_ids, min_new_tokens=completion_len, max_new_tokens=completion_len, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
            time_end = timeit.default_timer()
        
        elif args.intervention == 'top_k':
            context = f"Context: {prompt + ' ' + gt}\n"
            if model_name in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf', 'llama3-8b-chat-hf']:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context)[0]
            elif 'llama3' in model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context, model='llama3')[0]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            std = args.std
            def new_logits_processor(*args, **kwargs):
                prior = prior_processor(*args, **kwargs)
                prior.append(TopKPerturbationLogitsProcessor(tokenizer, model, std))
                return prior
            model._get_logits_processor = new_logits_processor
            time_start = timeit.default_timer()
            generate_ids = model.generate(inputs.input_ids, min_new_tokens=completion_len, max_new_tokens=completion_len, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
            time_end = timeit.default_timer()
        
        elif 'cad' in args.intervention:
            assert 'llama2-7b-hf' in model_name
            assert args.no_context==True
            null_prompt = prompt + ' ' + gt
            tokenized_null_input = tokenizer(null_prompt, return_tensors="pt").to(model.device)
            tokenized_input = tokenizer(prompt, return_tensors="pt").to(model.device)
            inputs = tokenized_input
            generation_config = GenerationConfig(do_sample=False, num_return_sequences=1, context_aware_decoding_alpha=args.context_aware_decoding_alpha, max_new_tokens=completion_len)
            time_start = timeit.default_timer()
            with torch.no_grad():
                generate_ids = model.generate(input_ids=tokenized_input.input_ids, null_inputs=tokenized_null_input.input_ids, attention_mask=tokenized_input.attention_mask, pad_token_id=tokenizer.eos_token_id, generation_config=generation_config)
            time_end = timeit.default_timer()
        else: # includes vanilla case as well as FT base model case.
            if args.no_context:
                context = ""
            else:
                context = f"Context: {prompt + ' ' + gt}\n"
            if any(element in model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf', 'dbrx', 'llama3-8b-chat-hf']):
                if "llama2" in model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context)[0]
                elif "dbrx" in model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context, model="dbrx")[0]
                elif 'llama3' in model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context, model="llama3")[0]
            elif "llama2-7b-hf" in model_name: # For base model case and we only evaluate non-context situation.
                assert args.no_context==True
            else:
                raise NotImplementedError
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            time_start = timeit.default_timer()
            generate_ids = model.generate(inputs.input_ids, min_new_tokens=completion_len, max_new_tokens=completion_len, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask) # None but greedy decoding
            time_end = timeit.default_timer()
        generation_time = time_end - time_start
        num_tokens = len(generate_ids[0]) - len(inputs.input_ids[0]) #TODO(wby)
        tokens_per_second = num_tokens / generation_time
        token_sec_list.append(tokens_per_second)
        print("tokens_per_second: ", tokens_per_second)
        # bp()
    return token_sec_list
    
def main(args):
    print(f"Model Name: {args.model_name}\nDataset: {args.datatype}\nIntervention: {args.intervention}")
    num_tests = args.num_tests
    completion_len = args.completion_len
    context_len = args.context_len
    model_name = args.model_name
    n = args.n

    if args.datatype == 'newsqa':
        testing_chunks = pd.read_csv(f'eval_data/newsqa/newsqa_blocklisted_infringement.csv') #TODO(wby) in the future we need to change to different model name.
        testing_chunks = testing_chunks.sort_values('ppl').head(num_tests)
        if "tokenized" in args.intervention:
            bloom_filter = f'newsqa_tokenized.{6*n}-{6*n}.bf'
        else:
            bloom_filter = f'newsqa.{n}-{n}.bf'
    elif args.datatype == 'booksum':
        testing_chunks = pd.read_csv(f'eval_data/booksum/booksum_blocklisted.csv', nrows=num_tests)
        if "tokenized" in args.intervention:
            bloom_filter = f'booksum_tokenized.{6*n}-{6*n}.bf'
        else:
            bloom_filter = f'booksum.{n}-{n}.bf'
    else:
        raise NotImplementedError
    if "unlearning" in args.intervention:
        model_name_new = model_name + "_" + args.intervention
        model, tokenizer = load_models(model_name_new)
    else:
        model, tokenizer = load_models(model_name)


    agg_res = {}
    agg_res['model'] = model_name
    agg_res['num_tests'] = num_tests
    agg_res['context_len'] = context_len
    agg_res['completion_len'] = completion_len
    
    if args.intervention == 'uncopy' or args.intervention == 'uncopy_merge':
        if args.datatype == 'newsqa':
            rewrite_text = pd.read_csv(f'eval_data/newsqa/ppl_newsqa_llama2-7b-hf_1000.csv')['rewrite'].tolist()
        elif args.datatype == 'booksum':
            rewrite_text = pd.read_csv(f'eval_data/booksum/ppl_booksum_llama2-7b-hf_500.csv')['rewrite'].tolist() 
    elif 'uncopy_fact' in args.intervention:
        if args.datatype == 'newsqa':
            rewrite_text = pd.read_csv(f'eval_data/newsqa/ppl_newsqa_llama2-7b-hf_1000.csv')['rewrite_fact'].tolist()
        elif args.datatype == 'booksum':
            rewrite_text = pd.read_csv(f'eval_data/booksum/ppl_booksum_llama2-7b-hf_500.csv')['rewrite_fact'].tolist()
    elif 'uncopy_summary' in args.intervention:
        if args.datatype == 'newsqa':
            rewrite_text = pd.read_csv(f'eval_data/newsqa/ppl_newsqa_llama2-7b-hf_1000.csv')['rewrite_summary'].tolist()
        elif args.datatype == 'booksum':
            rewrite_text = pd.read_csv(f'eval_data/booksum/ppl_booksum_llama2-7b-hf_500.csv')['rewrite_summary'].tolist()
    else:
        rewrite_text = []
    
    # swj change

    token_sec_list = run_completions(model_name, model, tokenizer, testing_chunks, context_len, completion_len, args, bloom_filter=bloom_filter, rewrite_text=rewrite_text)

    average_inference_speed = np.mean(token_sec_list)
    # print("agg_res: ", agg_res)
    # exit()
    if 'mem_free' in args.intervention:
        if args.intervention == "mem_free_new":
            intervention_type = args.intervention + '_' + f'{args.n}' + '_' + f'{args.skip_tokens}'
        else:
            intervention_type = args.intervention + '_' + f'{args.n}'
    elif args.intervention == 'top_k':
        intervention_type = args.intervention + '_' + f'{args.std}'
    elif args.intervention == 'cad':
        intervention_type = args.intervention + '_' + f'{args.context_aware_decoding_alpha}'
    else:
        intervention_type = args.intervention
        
    if os.path.exists(args.log_dir):
        df = pd.read_csv(args.log_dir)
        df = pd.concat([df, pd.DataFrame(agg_res, index=[0])])
        df.to_csv(args.log_dir, index=False)
    else:
        df = pd.DataFrame(agg_res, index=[0])
        df.to_csv(args.log_dir, index=False)
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.model_name}_{args.datatype}.txt")
    if args.datatype == 'newsqa':
        with open(save_filepath, "a") as f:
            print(f"{args.model_name}\t{intervention_type}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\ttokens_per_sec\t{average_inference_speed:.4f}", file=f, flush=True)
    elif args.datatype == 'booksum':
        with open(save_filepath, "a") as f:
            print(f"{args.model_name}\t{intervention_type}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\ttokens_per_sec\t{average_inference_speed:.4f}", file=f, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tests", type=int, default=100)
    parser.add_argument("--completion_len", type=int, default=500)
    parser.add_argument("--context_len", type=int, default=100)
    parser.add_argument("--model_name", type=str, default='llama2-7b-hf')
    parser.add_argument("--datatype", type=str, default='newsqa')
    parser.add_argument("--log_dir", type=str, default='res/log_completion.csv')
    parser.add_argument("--hints", type=bool, default=False)
    parser.add_argument("--intervention", type=str, default="none", choices=["none", "none_greedy",
                                                                             "top_k", 
                                                                             "mem_free-consecutive", "mem_free-non_consecutive",
                                                                             "mem_free_tokenized-consecutive", "mem_free_tokenized-non_consecutive",
                                                                             "mem_free_new",
                                                                             "sys_prompt-sys_a", "sys_prompt-sys_b", "sys_prompt-sys_c", 
                                                                             "sys_prompt-bing", "sys_prompt-copilot",  "sys_prompt-dbrx", 
                                                                             "uncopy", "uncopy_merge",
                                                                             "uncopy_fact", "uncopy_summary",
                                                                             "cad",
                                                                             "unlearning-gradient_ascent", "unlearning-dpo"])
    parser.add_argument("--save", type=str, default='res/utility_res')
    parser.add_argument("--use_low_ppl", action='store_true')
    parser.add_argument("--eval_zero_shot", action='store_true')
    parser.add_argument("--eval_general", action='store_true') # We evalaute the model's in-domain utility by default
    parser.add_argument("--no_context", action='store_true')
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--acs_threshold", type=int, default=50) # for mem-free non_consecutive situation
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument("--context_aware_decoding_alpha", type=float, default=0.0)
    parser.add_argument("--no_overwrite", action='store_true')
    parser.add_argument("--skip_tokens", type=int, default=10)
    
    args = parser.parse_args()
    main(args)
