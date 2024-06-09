# Import necessary modules
import numpy as np
import pandas as pd
import string
from ipdb import set_trace as bp
from tqdm import tqdm
import re
from tqdm import tqdm
import dataportraits
from lib.decoding_intervention import DataPortraitsLogitsProcessor
import sys
import json
sys.path.append("./lib")
from utils_cad import *
from .prompt_utils import apply_prompt_template


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def find_empty_positions(lst):
    empty_positions = []
    for index, element in enumerate(lst):
        if element is None or element == "" or str(element).isspace():
            empty_positions.append(index)
    return empty_positions


def text_processor(text):
    # delete <s> and </s> appearing in the text
    text = text.replace("<s>", "").replace("</s>", "")
    # remove the leading and trailing whitespaces
    text = text.strip().rstrip()
    # remove the trailing comma and period
    if text != '' and text[-1] in [",", "."]:
        text = text[:-1]
    return text


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    
    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)
    num_common_tokens = len(common_tokens)
    
    if num_common_tokens == 0:
        return 0.0
    
    precision = num_common_tokens / len(prediction_tokens)
    recall = num_common_tokens / len(ground_truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall)


def evaluate_f1_scores(predictions, ground_truths):
    f1_scores = [f1_score(pred, truth) for pred, truth in zip(predictions, ground_truths)]
    return sum(f1_scores) / len(f1_scores)


def eval_booksum(args, model, tokenizer, num_sampled=200, bloom_filter=None, split='train'):
    print("Evaluating on BookSum...")
    if split == 'train':
        booksum_dataset = pd.read_csv('eval_data/booksum/booksum_blocklisted_utility.csv')
    elif split == 'test':
        booksum_dataset = pd.read_csv('eval_data/booksum/booksum_indomain_utility.csv')
    prompt_format = "Summarize the article. Summary:"
    predictions, references, documents = [], [], []
    prior_processor = model._get_logits_processor
    for i, ex in tqdm(booksum_dataset[:num_sampled].iterrows()):
        answer = ex['summary']
        text = ex['document']
        prompt = f"Document: {text}\n\n{prompt_format}"
        
        if 'mem_free' in args.intervention:
            if any(element in args.model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
            elif "llama2-7b-hf" in args.model_name:
                prompt = f"Document: {text}\n\n{prompt_format}"
            elif 'llama3' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model='llama3')[0]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            bf_is_tokenized = "tokenized" in args.intervention
            match = re.search(r'\d+', bloom_filter)
            n = int(match[0])
            portrait = dataportraits.RedisBFSketch('localhost', 8899, bloom_filter, int(n))
            choice = args.intervention.split('-')[-1]
            if choice == 'consecutive':
                if "tokenized" in args.intervention:
                    width =2 * int(n) - 6
                else:
                    width = 2 * int(n) - 1
                def new_logits_processor(*args, **kwargs):
                    prior = prior_processor(*args, **kwargs)
                    if len(prior) == 1:
                        prior.pop() # Remove the existing bloom_filter logits processor
                    prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait, bf_is_tokenized=bf_is_tokenized, tokenized_prompt=inputs, n=int(n), consecutive=True))
                    return prior
            else:
                width = 3 * int(n)
                acs_threshold = args.acs_threshold
                def new_logits_processor(*args, **kwargs):
                    prior = prior_processor(*args, **kwargs)
                    if len(prior) == 1:
                        prior.pop() # Remove the existing bloom_filter logits processor
                    prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait, bf_is_tokenized=bf_is_tokenized, tokenized_prompt=inputs, n=int(n), consecutive=False, acs_threshold=acs_threshold))
                    return prior
            model._get_logits_processor = new_logits_processor
            context_len = inputs.input_ids.shape[1]
            if context_len > 3500:
                print("find examples with context length > 3500, continue")
                continue
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=500, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
        elif args.intervention == 'top_k':
            if any(element in args.model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
            elif "llama2-7b-hf" in args.model_name:
                prompt = f"Document: {text}\n\n{prompt_format}"
            elif 'llama3' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model='llama3')[0]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            context_len = inputs.input_ids.shape[1]
            if context_len > 3500:
                print("find examples with context length > 3500, continue")
                continue
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=500, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
        elif 'sys_prompt' in args.intervention:
            system_prompt_choice = args.intervention.split('-')[-1]
            if 'llama2' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True)[0]
            elif 'dbrx' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True, model='dbrx')[0]
            elif 'llama3' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True, model='llama3')[0]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            context_len = inputs.input_ids.shape[1]
            if context_len > 3500:
                print("find examples with context length > 3500, continue")
                continue
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=500, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
        else:
            if any(element in args.model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
            elif 'dbrx' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model='dbrx')[0]
            elif 'llama3' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model='llama3')[0]
            elif 'llama2-7b-hf' in args.model_name:
                prompt = prompt
            else:
                raise ValueError("Invalid model name")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            context_len = inputs.input_ids.shape[1]
            if context_len > 3500:
                print("find examples with context length > 3500, continue")
                continue
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=500, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prompt = prompt.replace("<|im_start|>", "").replace("<|im_end|>", "") # For dbrx, because it won't output special token during generation
        prompt = prompt.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "").replace("<|start_header_id|>", "").replace("<|end_header_id|>", "") # For llama3
        outputs = [o.replace(prompt, '') for o in outputs]
        prediction = outputs[0].replace(prompt, "").replace("<s>", "").replace("</s>", "").strip().rstrip()
        # bp()
        predictions.append(prediction)
        references.append(answer)
        documents.append(text)
        
    if "llama2-7b-chat" in args.model_name: # Here we zero out the empty predictions in the vanilla model for fair comparison. Different model may have different empty predictions in the vanilla case
        if split == 'train':
            invalid_loc = [0, 29, 49, 65, 76, 90, 148, 166, 182]
        else:
            invalid_loc = [38, 73, 75, 76, 77, 109, 110, 116, 117, 118, 128, 129, 130, 134, 143, 146, 168, 169, 172]
        for loc in invalid_loc:
            predictions[loc] = ""
    elif "llama2-70b-chat" in args.model_name:
        if split == 'train':
            invalid_loc = [1, 8, 10, 13, 17, 19, 24, 29, 32, 33, 36, 41, 46, 62, 68, 73, 74, 75, 76, 79, 81, 84, 86, 89, 99, 102, 105, 108, 111, 112, 120, 128, 134, 137, 149, 150, 153, 155, 166, 185, 186, 187, 188]
        else:
            invalid_loc = [39, 58, 59, 96, 102, 104, 106, 139, 140, 141, 142, 145, 146, 150, 152, 155, 156, 157, 158, 159, 160, 163, 165, 166, 170, 171, 173, 183]
        for loc in invalid_loc:
            predictions[loc] = ""
    if args.intervention == "none": # For find empty predictions in the vanilla model
        empty_positions = find_empty_positions(predictions)
        print(f"{args.model_name}\n{split}\tempty_list={empty_positions}")
    evaluator = Evaluator()
    result_dict = evaluator.evaluate(predictions, references, documents)
    return result_dict


def eval_mmlu(args, model, tokenizer, num_sampled=200, bloom_filter=None):
    print("Evaluating on MMLU...")

    prompt_instruction = ""
    subject2em = {}
    prior_processor = model._get_logits_processor
    for subject in tqdm(os.listdir("./eval_data/mmlu")):
        all_em = []
        train_data = read_jsonl(f"./eval_data/mmlu/{subject}/dev.jsonl")
        # formulate the prompt
        prompt_orig = ""
        for ex in train_data:
            ex_instruction = """Question: {}\nChoices: A: {}, B: {}, C: {}, D: {},\nAnswer: {}\n\n"""
            ex_instruction = ex_instruction.format(ex['question'], ex['choices']['A'], ex['choices']['B'], ex['choices']['C'], ex['choices']['D'], ex['answer'])
            prompt_orig += ex_instruction
        prompt_orig += prompt_instruction

        test_data = read_jsonl(f"./eval_data/mmlu/{subject}/test.jsonl")
        for ex in tqdm(test_data[:50]):
            ex_test_instruction = """Question: {}\nChoices: A: {}, B: {}, C: {}, D: {},\n"""
            answer = ex['answer']
            test_data_prompt = ex_test_instruction.format(ex['question'], ex['choices']['A'], ex['choices']['B'], ex['choices']['C'], ex['choices']['D'])
            prompt = prompt_orig + test_data_prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            context_len = inputs.input_ids.shape[1]
            
            if 'mem_free' in args.intervention:
                if any(element in args.model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
                elif 'llama2-7b-hf' in args.model_name:
                    prompt = prompt
                elif 'llama3' in args.model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model='llama3')[0]
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                bf_is_tokenized = "tokenized" in args.intervention
                match = re.search(r'\d+', bloom_filter)
                n = int(match[0])
                portrait = dataportraits.RedisBFSketch('localhost', 8899, bloom_filter, int(n))
                choice = args.intervention.split('-')[-1]
                if choice == 'consecutive':
                    if "tokenized" in args.intervention:
                        width =2 * int(n) - 6
                    else:
                        width = 2 * int(n) - 1
                    def new_logits_processor(*args, **kwargs):
                        prior = prior_processor(*args, **kwargs)
                        if len(prior) == 1:
                            prior.pop()
                        prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait, bf_is_tokenized=bf_is_tokenized, tokenized_prompt=inputs, n=int(n), consecutive=True))
                        return prior
                else:
                    width = 3 * int(n)
                    acs_threshold = args.acs_threshold
                    def new_logits_processor(*args, **kwargs):
                        prior = prior_processor(*args, **kwargs)
                        if len(prior) == 1:
                            prior.pop() # Remove the existing bloom_filter logits processor
                        prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait, bf_is_tokenized=bf_is_tokenized, tokenized_prompt=inputs, n=int(n), consecutive=False, acs_threshold=acs_threshold))
                        return prior
                model._get_logits_processor = new_logits_processor
                context_len = inputs.input_ids.shape[1]
                if context_len > 3500:
                    print("find examples with context length > 3500, continue")
                    continue
                generate_ids = model.generate(inputs.input_ids, max_new_tokens=5, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
            elif args.intervention == 'top_k':
                if any(element in args.model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
                elif 'llama2-7b-hf' in args.model_name:
                    prompt = prompt
                elif 'llama3' in args.model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model='llama3')[0]
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                context_len = inputs.input_ids.shape[1]
                if context_len > 3500:
                    print("find examples with context length > 3500, continue")
                    continue
                generate_ids = model.generate(inputs.input_ids, max_new_tokens=5, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
            elif 'sys_prompt' in args.intervention:
                system_prompt_choice = args.intervention.split('-')[-1]
                if 'llama2' in args.model_name:
                    prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True)[0]
                elif 'dbrx' in args.model_name:
                    prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True, model='dbrx')[0]
                elif 'llama3' in args.model_name:
                    prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True, model='llama3')[0]
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                context_len = inputs.input_ids.shape[1]
                if context_len > 3500:
                    print("find examples with context length > 3500, continue")
                    continue
                generate_ids = model.generate(inputs.input_ids, max_new_tokens=5, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
            else:
                model.generation_config.context_aware_decoding_alpha = None    
                if any(element in args.model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
                elif 'dbrx' in args.model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model='dbrx')[0]
                elif 'llama3' in args.model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model='llama3')[0]
                elif 'llama2-7b-hf' in args.model_name:
                    prompt = prompt
                else:
                    raise ValueError("Invalid model name")
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                context_len = inputs.input_ids.shape[1]
                if context_len > 3500:
                    print("find examples with context length > 3500, continue")
                    continue
                generate_ids = model.generate(inputs.input_ids, max_new_tokens=5, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)            
            
            outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            prompt = prompt.replace("<|im_start|>", "").replace("<|im_end|>", "")
            prompt = prompt.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "").replace("<|start_header_id|>", "").replace("<|end_header_id|>", "") # For llama3
            # bp()
            outputs = [o.replace(prompt, '') for o in outputs]
            outputs = outputs[0].split("\n")
            selected_outputs = [s for s in outputs if "Answer" in s]
            if len(selected_outputs) == 0:
                all_em.append(0)
                continue
            else:
                outputs = selected_outputs[0]
            outputs = outputs.replace("Answer", "").strip(string.punctuation).strip()
            if (outputs not in ['A', 'B', 'C', 'D']):
                all_em.append(0)
                continue
            em = answer == outputs
            all_em.append(em)
        if len(all_em) == 0:
            continue
        else:
            em_subject = sum(all_em) / len(all_em)
            subject2em[subject] = em_subject
        print(subject2em)
    avg_em = sum(subject2em.values()) / len(subject2em)
    std_em = np.std(list(subject2em.values()))
    confidence_interval = 1.96 * std_em / np.sqrt(len(subject2em))
    print(f"Average EM: {avg_em}, std: {std_em}, confidence interval: {confidence_interval}")
    return avg_em


def eval_newsqa(args, model, tokenizer, num_sampled=200, bloom_filter=None, split='train'):
    print("Evaluating on NewsQA...")
    if split=='train':
        newsqa_dataset = pd.read_csv("eval_data/newsqa/newsqa_blocklisted_utility.csv")
    elif split == 'test':
        newsqa_dataset = pd.read_csv("eval_data/newsqa/newsqa_indomain_utility.csv")
    if args.no_context:
        prompt_examples = """Question: What was the amount of children murdered?
Answer: 19
                    
Question: Where was one employee killed ?
Answer: Sudanese region of Darfur
                    
Question: who did say South Africa did not issue a visa on time ?
Answer: Archbishop Desmond Tutu
                        
Question: How many years old was the businessman ?
Answer: 29-year-old

Question: what Pope used to beat himself ?
Answer: John Paul II"""
    else:  
        prompt_examples = """Context: NEW DELHI, India (CNN) -- A high court in northern India on Friday acquitted a wealthy businessman facing the death sentence for the killing of a teen in a case dubbed "the house of horrors." Moninder Singh Pandher was sentenced to death by a lower court in February. The teen was one of 19 victims -- children and young women -- in one of the most gruesome serial killings in India in recent years. Pandher faces trial in the remaining 18 killings and could remain in custody, the attorney said.
Question: What was the amount of children murdered?
Answer: 19
                    
Context: -LRB- CNN -RRB- -- Fighting in the volatile Sudanese region of Darfur has sparked another wave of refugees into Chad and left a Red Cross employee dead , according to international agencies . Refugee camps in eastern Chad house about 300,000 people who fled violence in the Darfur region of Sudan . The U.N. High Commissioner for Refugees said on Monday that more than 12,000 people have fled militia attacks over the last few days from Sudan 's Darfur region to neighboring Chad , still recovering from a recent attempt by rebels there to topple the government .
Question: Where was one employee killed ?
Answer: Sudanese region of Darfur
                    
Context: Johannesburg -LRB- CNN -RRB- -- Miffed by a visa delay that led the Dalai Lama to cancel a trip to South Africa , Archbishop Desmond Tutu lashed out at his government Tuesday , saying it had acted worse than apartheid regimes and had forgotten all that the nation stood for .The Dalai Lama fled Tibet in 1959 after a failed uprising against Chinese rule , and China pressures governments around the world to deny him any legitimacy . Speculation surfaced Tuesday that this year 's visit was also affected by South Africa 's relationship with China . Kim Norgaard , CNN 's Johannesburg bureau chief , contributed to this report .
Question: who did say South Africa did not issue a visa on time ?
Answer: Archbishop Desmond Tutu
                        
Context: -LRB- CNN -RRB- -- England international footballer Steven Gerrard was found not guilty of affray by a court in his home city on Friday . England international Steven Gerrard was cleared by a court in Liverpool of affray . The jury at Liverpool Crown Court took a little over an hour to clear Gerrard of charges relating to a fracas in a nightclub bar in the north-western of England city on December 29 of last year . They accepted the Liverpool captain 's version that he acted in self defense in punching businessman Marcus McGhee . The 29-year-old was the only one of the seven defendants in the case to be cleared after an incident which was described by judge Henry Globe as an `` explosion of violence . ''
Question: How many years old was the businessman ?
Answer: 29-year-old

Context: -LRB- CNN -RRB- -- Pope John Paul II used to beat himself with a belt and sleep on a bare floor to bring himself closer to Christ , a book published Wednesday says . The late pope had a particular belt for self-flagellation and brought it with him to his summer residence , according to the book , `` Why he is a Saint : The True story of John Paul II . '' `` As some members of his own entourage were able to hear with their own ears , both in Poland and in the Vatican , Karol Wojtyla flagellated himself , '' the book says , using the name the pope was given at birth . `` In the closet , among the cloaks , a particular pant-belt hung from a hook , which he utilized as a whip and one which he always had brought to Castel Gandolfo , '' the book says .
Question: what Pope used to beat himself ?
Answer: John Paul II"""
    valid_answer_gt = []
    valid_output = []
    
    prior_processor = model._get_logits_processor
    
    for i in tqdm(range(len(newsqa_dataset))):
        answer = newsqa_dataset['answer'][i]
        answer = text_processor(answer)
        if args.no_context==False:
            text = newsqa_dataset['story_text'][i]
        question = newsqa_dataset['question'][i]
        if args.no_context:
            prompt = f"{prompt_examples}\n\nQuestion: {question}\n"
        else:
            prompt = f"{prompt_examples}\n\nContext: {text}\nQuestion: {question}\n"
        
        if 'mem_free' in args.intervention:
            bf_is_tokenized = "tokenized" in args.intervention
            match = re.search(r'\d+', bloom_filter)
            n = int(match[0])
            portrait = dataportraits.RedisBFSketch('localhost', 8899, bloom_filter, int(n))
            if any(element in args.model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
            elif "llama2-7b-hf" in args.model_name:
                prompt = f"{prompt_examples}\n\nQuestion: {question}\n"
            elif 'llama3' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model='llama3')[0]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            choice = args.intervention.split('-')[-1]
            if choice == 'consecutive':
                if "tokenized" in args.intervention:
                    width =2 * int(n) - 6
                else:
                    width = 2 * int(n) - 1
                def new_logits_processor(*args, **kwargs):
                    prior = prior_processor(*args, **kwargs)
                    if len(prior) == 1:
                        prior.pop() # Remove the existing bloom_filter logits processor
                    prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait, bf_is_tokenized=bf_is_tokenized, tokenized_prompt=inputs, n=int(n), consecutive=True))
                    return prior
            else:
                width = 3 * int(n)
                acs_threshold = args.acs_threshold
                def new_logits_processor(*args, **kwargs):
                    prior = prior_processor(*args, **kwargs)
                    if len(prior) == 1:
                        prior.pop() # Remove the existing bloom_filter logits processor
                    prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait, bf_is_tokenized=bf_is_tokenized, tokenized_prompt=inputs, n=int(n), consecutive=False, acs_threshold=acs_threshold))
                    return prior
            model._get_logits_processor = new_logits_processor
            context_len = inputs.input_ids.shape[1]
            if context_len > 3900:
                print("find examples with context length > 3900, continue")
                continue
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=20, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
        elif args.intervention == 'top_k':
            if any(element in args.model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
            elif "llama2-7b-hf" in args.model_name:
                prompt = f"{prompt_examples}\n\nQuestion: {question}\n"
            elif 'llama3' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model='llama3')[0]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            context_len = inputs.input_ids.shape[1]
            if context_len > 3900:
                print("find examples with context length > 3900, continue")
                continue
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=20, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
        elif 'sys_prompt' in args.intervention:
            system_prompt_choice = args.intervention.split('-')[-1]
            if 'llama2' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True)[0]
            elif 'dbrx' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True, model='dbrx')[0]
            elif 'llama3' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True, model='llama3')[0]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            context_len = inputs.input_ids.shape[1]
            if context_len > 3900:
                print("find examples with context length > 3900, continue")
                continue
            context_len = inputs.input_ids.shape[1]
            if context_len > 3900:
                print("find examples with context length > 3500, continue")
                continue
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=20, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
        else:
            if any(element in args.model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
            elif 'dbrx' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model='dbrx')[0]
            elif 'llama3' in args.model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model='llama3')[0]
            elif "llama2-7b-hf" in args.model_name:
               prompt = f"{prompt_examples}\n\nQuestion: {question}\n" 
            else:
                raise ValueError("Invalid model name")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            context_len = inputs.input_ids.shape[1]
            if context_len > 3900:
                print("find examples with context length > 3900, continue")
                continue
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=20, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prompt = prompt.replace("<|im_start|>", "").replace("<|im_end|>", "") # For dbrx, because it won't output special token during generation
        prompt = prompt.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "").replace("<|start_header_id|>", "").replace("<|end_header_id|>", "") # For llama3
        outputs = [o.replace(prompt, '') for o in outputs]
        print(outputs[0])
        outputs = outputs[0].split("\n")
        selected_outputs = [s for s in outputs if "Answer" in s]
        if len(selected_outputs) == 0:
            valid_answer_gt.append(answer)
            valid_output.append("")
            continue
        else:
            outputs = selected_outputs[0]
        outputs = outputs.replace("Answer", "").strip(string.punctuation).strip()
        valid_answer_gt.append(answer)
        valid_output.append(text_processor(outputs))
        if (len(valid_answer_gt) == num_sampled):
            break
    # evaluation process:
    # Compute the F1 score.
    print(f"{len(valid_output)}, {len(valid_answer_gt)}")
    f1_score = evaluate_f1_scores(valid_output, valid_answer_gt)
    # Compute the exact match score
    return f1_score
    
    
    