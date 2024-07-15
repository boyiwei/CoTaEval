import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from ipdb import set_trace as bp
import evaluate
from transformers import GenerationConfig
import torch
from sentence_transformers import SentenceTransformer, util
import tqdm
import pandas as pd
import os
import lib.utils as utils
from lib.eval import eval_newsqa, eval_booksum, eval_mmlu
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
        if model_name in ['llama2-7b-chat-hf', 'llama2-70b-chat-hf', 'llama2-13b-chat-hf']:
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


def run_completions(model_name, model, tokenizer, testing_chunks, context_len, completion_len, args, bloom_filter=None):
    output_list, prompt_list, gt_list, inference_time_list = [], [], [], []
    prior_processor = model._get_logits_processor
    if args.intervention != 'cad':
        model.generation_config.context_aware_decoding_alpha = None # Here we add this to avoid error for the non-cad situation.
    model.generation_config.mem_free_new = False
    for i, (prompt, gt) in tqdm.tqdm(enumerate(zip(testing_chunks['prompt_autocomplete'], testing_chunks['gt_autocomplete'])), total=len(testing_chunks)):
        prompt = ' '.join(prompt.split(' ')[-context_len:])
        
        if 'mem_free' in args.intervention:
            bf_is_tokenized = "tokenized" in args.intervention
            choice = args.intervention.split('-')[-1]
            if args.no_context:
                context = ""
            else:
                context = f"Context: {prompt + ' ' + gt}\n"
            if any(element in args.model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context)[0]
            elif 'llama3' in model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context, model="llama3")[0]
            match = re.search(r'\d+', bloom_filter)
            n = int(match[0])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            portrait = dataportraits.RedisBFSketch('localhost', 8899, bloom_filter, int(n))
            if choice == 'consecutive':
                if "tokenized" in args.intervention:
                    # width = 2 * int(n) - 5
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
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=completion_len, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
            time_end = timeit.default_timer()
        
        elif args.intervention == 'top_k':
            if args.no_context:
                context = ""
            else:
                context = f"Context: {prompt + ' ' + gt}\n"
            if any(element in args.model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
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
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=completion_len, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
            time_end = timeit.default_timer()
        
        elif 'sys_prompt' in args.intervention:
            if args.no_context:
                context = ""
            else:
                context = f"Context: {prompt + ' ' + gt}\n"
            system_prompt_choice = args.intervention.split('-')[-1]
            if 'llama2' in model_name:
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], context=context)[0]
            elif 'dbrx' in model_name:
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], context=context, model='dbrx')[0]
            elif 'llama3' in model_name:
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], context=context, model='llama3')[0]
            if args.no_cnn:
                prompt = prompt.replace("CNN", "")   
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            time_start = timeit.default_timer()
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=completion_len, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask) # The only difference is do_sample
            time_end = timeit.default_timer()
        
        elif 'cad' in args.intervention: # only applicatble to memorization setting
            assert args.no_context==True
            null_prompt = prompt + ' ' + gt
            prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context="")[0]
            tokenized_null_input = tokenizer(null_prompt, return_tensors="pt").to(model.device)
            tokenized_input = tokenizer(prompt, return_tensors="pt").to(model.device)
            generation_config = GenerationConfig(do_sample=False, num_return_sequences=1, context_aware_decoding_alpha=args.context_aware_decoding_alpha, max_new_tokens=completion_len)
            time_start = timeit.default_timer()
            with torch.no_grad():
                generate_ids = model.generate(input_ids=tokenized_input.input_ids, null_inputs=tokenized_null_input.input_ids, attention_mask=tokenized_input.attention_mask, generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)
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
            if args.no_cnn:
                prompt = prompt.replace("CNN", "")    
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            time_start = timeit.default_timer()
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=completion_len, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask) # None but greedy decoding
            time_end = timeit.default_timer()
        print(f"Time taken for inference: {time_end - time_start}")
        inference_time_list.append(time_end - time_start)
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # bp()
        # Compute ROUGE scores
        prompt = prompt.replace("<|im_start|>", "").replace("<|im_end|>", "") # For dbrx, because it won't output special token during generation
        prompt = prompt.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "").replace("<|start_header_id|>", "").replace("<|end_header_id|>", "") # For llama3
        outputs = [o.replace(prompt, '') for o in outputs]
        
        # bp()
        output_list.append(outputs)
        prompt_list.append(prompt)
        gt_list.append(gt)
        
    return output_list, prompt_list, gt_list, inference_time_list
    
    
def eval_infringement(model_name, data_type, intervention, prompt_list, gt_list, output_list, inference_time_list, agg_res, context_len, completion_len, args):  
    rouge = evaluate.load('rouge')
    rouge_1, rouge_l, prompts = [], [], []

    # eval semantic similarity
    semantic_sim = []
    model = SentenceTransformer("all-MiniLM-L6-v2")

    best_rouge1_comps, best_rougeL_comps, best_verbatim_matching_comps, gts, matching_sequences = [], [], [], [], []
    best_rouge1_ids, best_rougeL_ids, best_verbatim_matching_ids = [], [], []
    best_verbatim_matching_ids, max_lengths, total_lengths = [], [], [] 
    # begin compute time
    # time_start = timeit.default_timer()
    for prompt, gt, outputs in zip(prompt_list, gt_list, output_list):
        best_verbatim_matching_id, matching_sequence, max_length, total_length = utils.find_common_sequences(outputs, gt)
        results = rouge.compute(predictions=outputs, references=[gt]*len(outputs), use_aggregator=False) 

        # semantic simlarity
        ref_embeddings = model.encode([gt])
        pred_embeddings = model.encode(outputs)
        cos_sim = util.cos_sim(pred_embeddings, ref_embeddings).cpu().numpy().squeeze().tolist()
        if isinstance(cos_sim, float):
            cos_sim = [cos_sim]
        max_cos_sim = max(cos_sim)
        semantic_sim.append(max_cos_sim)
        # bp()

        max_rougeL = max(results['rougeL'])
        max_rouge1 = max(results['rouge1'])
        best_rougeL = outputs[results['rougeL'].index(max_rougeL)]
        best_rouge1 = outputs[results['rouge1'].index(max_rouge1)]
        best_verbatim_matching = outputs[best_verbatim_matching_id]
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
        matching_sequences.append(matching_sequence)
    
    data_type = f'{data_type}_low_ppl'
    df = pd.DataFrame({'prompt': prompts, 'gt': gts, 'rouge1': rouge_1, 'rougeL': rouge_l, 'semantic_sim': semantic_sim,
                    'best_rouge1': best_rouge1_comps, 'best_rougeL': best_rougeL_comps, 'best_verbatim_matching': best_verbatim_matching_comps,
                    'matching_sequence': matching_sequences,
                    'max_length': max_lengths, 'total_length': total_lengths,
                    'best_rouge1_ids': best_rouge1_ids, 'best_rougeL_ids': best_rougeL_ids, "best_verbatim_matching_ids": best_verbatim_matching_ids, "inference_time": inference_time_list}) 
    
    if 'mem_free' in intervention:
        path = f'res/output_res/{data_type}_comp_{model_name}_context_len_{context_len}_completion_len_{completion_len}_intervention_{intervention}_{args.n}_no_context_{args.no_context}.csv'
    elif intervention == 'top_k':
        path = f'res/output_res/{data_type}_comp_{model_name}_context_len_{context_len}_completion_len_{completion_len}_intervention_{intervention}_{args.std}_no_context_{args.no_context}.csv'
    elif intervention == 'cad':
        path = f'res/output_res/{data_type}_comp_{model_name}_context_len_{context_len}_completion_len_{completion_len}_intervention_{intervention}_{args.context_aware_decoding_alpha}_no_context_{args.no_context}.csv'
    elif args.no_cnn:
        path = f'res/output_res/{data_type}_comp_{model_name}_context_len_{context_len}_completion_len_{completion_len}_intervention_{intervention}_no_context_{args.no_context}_no_cnn.csv'
    else:
        path = f'res/output_res/{data_type}_comp_{model_name}_context_len_{context_len}_completion_len_{completion_len}_intervention_{intervention}_no_context_{args.no_context}.csv'
    
    if args.no_overwrite:
        counter = 1
        new_path = path
        while os.path.exists(new_path):
            base, extension = os.path.splitext(path)
            new_path = f"{base}_{counter}{extension}"
            counter += 1
        df.to_csv(new_path)
    else:  
        df.to_csv(path)
    
    agg_res['max_rouge1'] = df['rouge1'].max()
    agg_res['max_rougeL'] = df['rougeL'].max()
    agg_res['max_semantic_sim'] = df['semantic_sim'].max()
    agg_res['mean_rouge1'] = df['rouge1'].mean()
    agg_res['mean_rougeL'] = df['rougeL'].mean()
    agg_res['inference_time'] = sum(inference_time_list) / len(inference_time_list)
    return agg_res


def main(args):
    print(f"Model Name: {args.model_name}\nDataset: {args.datatype}\nIntervention: {args.intervention}")
    num_tests = args.num_tests
    completion_len = args.completion_len
    context_len = args.context_len
    model_name = args.model_name
    n = args.n

    if args.datatype == 'newsqa':
        testing_chunks = pd.read_csv(f'eval_data/newsqa/newsqa_blocklisted_infringement.csv')
        testing_chunks = testing_chunks.head(num_tests)
        if "tokenized" in args.intervention:
            bloom_filter = f'newsqa_tokenized.{6*n}-{6*n}.bf'
        else:
            bloom_filter = f'newsqa.{n}-{n}.bf'
    elif args.datatype == 'booksum':
        testing_chunks = pd.read_csv(f'eval_data/booksum/booksum_blocklisted_infringement.csv', nrows=num_tests)
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
    
    
    # Evaluate infringement
    if args.eval_infringement:
        output_list, prompt_list, gt_list, inference_time_list = run_completions(model_name, model, tokenizer, testing_chunks, context_len, completion_len, args, bloom_filter=bloom_filter)
        agg_res = eval_infringement(model_name, args.datatype, args.intervention, prompt_list, gt_list, output_list, inference_time_list, agg_res, context_len, completion_len, args)
    
    
    if 'mem_free' in args.intervention:
        intervention_type = args.intervention + '_' + f'{args.n}'
    elif args.intervention == 'top_k':
        intervention_type = args.intervention + '_' + f'{args.std}'
    elif args.intervention == 'cad':
        intervention_type = args.intervention + '_' + f'{args.context_aware_decoding_alpha}'
    else:
        intervention_type = args.intervention
    if args.no_cnn:
        intervention_type = intervention_type + "_no_cnn"
    
    if args.eval_general or args.eval_zero_shot:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.model_name}_{args.datatype}.txt")
    
    if args.eval_general:
        mmlu_score = eval_mmlu(args, model, tokenizer, num_sampled=100, bloom_filter=bloom_filter)
        with open(save_filepath, "a") as f:
            print(f"{args.model_name}\t{intervention_type}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\tmmlu\t{mmlu_score:.4f}", file=f, flush=True)
    
    if args.eval_zero_shot:
        if args.datatype == 'newsqa':
            f1_score_train = eval_newsqa(args, model, tokenizer, num_sampled=500, bloom_filter=bloom_filter, split='train') # in-domain eval
            f1_score_test = eval_newsqa(args, model, tokenizer, num_sampled=500, bloom_filter=bloom_filter, split='test') # in-distribution eval
            with open(save_filepath, "a") as f:
                print(f"{args.model_name}\t{intervention_type}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\tf1_news_train\t{f1_score_train:.4f}", file=f, flush=True)
                print(f"{args.model_name}\t{intervention_type}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\tf1_news_test\t{f1_score_test:.4f}", file=f, flush=True)
        elif args.datatype == 'booksum':
            results_train = eval_booksum(args, model, tokenizer, num_sampled=200, bloom_filter=bloom_filter, split='train')
            results_test = eval_booksum(args, model, tokenizer, num_sampled=200, bloom_filter=bloom_filter, split='test')
            with open(save_filepath, "a") as f:
                print(f"{args.model_name}\t{intervention_type}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\trougeL_recall_booksum_train\t{results_train['rougeL_recall']:.4f}", file=f, flush=True)
                print(f"{args.model_name}\t{intervention_type}\t{args.datatype}\t{args.context_len}\t{args.completion_len}\trougeL_recall_booksum_test\t{results_test['rougeL_recall']:.4f}", file=f, flush=True)
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tests", type=int, default=100)
    parser.add_argument("--completion_len", type=int, default=500)
    parser.add_argument("--context_len", type=int, default=100)
    parser.add_argument("--model_name", type=str, default='llama2-7b-hf')
    parser.add_argument("--datatype", type=str, default='newsqa')
    parser.add_argument("--log_dir", type=str, default='res/log_completion.csv')
    parser.add_argument("--intervention", type=str, default="none", choices=["none",
                                                                             "top_k", 
                                                                             "mem_free_tokenized-consecutive",
                                                                             "sys_prompt-sys_a", "sys_prompt-sys_b", "sys_prompt-sys_c", 
                                                                             "sys_prompt-bing", "sys_prompt-copilot",  "sys_prompt-dbrx", 
                                                                             "cad"])
    parser.add_argument("--save", type=str, default='res/utility_res')
    parser.add_argument("--eval_infringement", action='store_true')
    parser.add_argument("--eval_zero_shot", action='store_true')
    parser.add_argument("--eval_general", action='store_true')
    parser.add_argument("--no_context", action='store_true')
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--acs_threshold", type=int, default=50) # for mem-free non_consecutive situation
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument("--context_aware_decoding_alpha", type=float, default=0.0)
    parser.add_argument("--no_overwrite", action='store_true')
    parser.add_argument("--skip_tokens", type=int, default=10)
    parser.add_argument("--no_cnn", action='store_true')
    
    args = parser.parse_args()
    main(args)
