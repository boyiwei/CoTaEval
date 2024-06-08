import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
import dataportraits
from transformers.generation.logits_process import *
import numpy as np

            
class DataPortraitsLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
    Contrarily to [`MinLengthLogitsProcessor`], this processor ignores the prompt.

    Args:
        prompt_length_to_skip (`int`):
            The input tokens length. Not a valid argument when used with `generate` as it will automatically assign the
            input length.
        min_new_tokens (`int`):
            The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, prompt: str, width: int, tokenizer, portrait, tokenized_prompt, consecutive, n, acs_threshold=50, bf_is_tokenized=False):
        self.prompt = prompt
        self.tokenized_prompt = tokenized_prompt 
        self.n = n
        self.width = width
        self.tokenizer = tokenizer
        self.portrait = portrait
        self.bf_is_tokenized = bf_is_tokenized # Whether used tokenized bf or not
        self.consecutive = consecutive # Whether do consecutive decection or not.
        self.acs_threshold = acs_threshold # For non-consecutive situation. When ACS reaches to threshold we do intervention

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = input_ids.cpu().detach().numpy()
        
        # Order the tokens by their likelihood
        order = torch.argsort(-scores, 1)
        order = order.cpu().detach().numpy()
        batch_size = input_ids.shape[0]
        
        # Set the likelihood to 0 for all the modest likely next tokens which would cause copyright infringement
        if self.consecutive:
            for ex in range(batch_size):
                for i in order[ex]:
                    ids_to_check = (input_ids[ex].tolist() + [int(i)])
                    if self.bf_is_tokenized:
                        sequence_to_check = ''.join(f"{num:06d}" for num in ids_to_check)
                        tokenized_prompt_string = ''.join(f"{num:06d}" for num in self.tokenized_prompt.input_ids[0].tolist())
                        res_sequence = sequence_to_check.replace(tokenized_prompt_string, "", 1)
                        report = self.portrait.contains_from_text([res_sequence[-self.width:]], stride=1)
                        # report = self.portrait.contains_from_text([tokenized_prompt_string[-self.width:]], stride=5)
                    else:
                        sequence_to_check = self.tokenizer.decode(ids_to_check, skip_special_tokens = True)
                        res_sequence = sequence_to_check.replace(self.prompt, "", 1) + ' '
                        report = self.portrait.contains_from_text([res_sequence[-self.width:]], stride=1)
                    if report[0]['chain_idxs'] != []: # which means we have detected overlapping
                        print("-----------------------------------------------------------")
                        print(report[0]['chains'])
                        print("-----------------------------------------------------------")
                        scores[ex][i] -= 1000
                    else:
                        break
        else:
            for ex in range(batch_size):
                for i in order[ex]:
                    ids_to_check = (input_ids[ex].tolist() + [int(i)])
                    if self.bf_is_tokenized:
                        sequence_to_check = ''.join(f"{num:06d}" for num in ids_to_check)
                        tokenized_prompt_string = ''.join(f"{num:06d}" for num in self.tokenized_prompt.input_ids[0].tolist())
                        res_sequence = sequence_to_check.replace(tokenized_prompt_string, "", 1)
                        report = self.portrait.contains_from_text([res_sequence[-self.width:]], stride=1)
                    else:
                        sequence_to_check = self.tokenizer.decode(ids_to_check, skip_special_tokens = True)
                        res_sequence = sequence_to_check.replace(self.prompt, "", 1) + ' '
                        report = self.portrait.contains_from_text([res_sequence[-self.width:]], stride=1)
                    if report[0]['chain_idxs'] != []:
                        max_length = (max(len(inner_list) for inner_list in report[0]['chain_idxs'])) * self.n
                        if max_length >= self.acs_threshold: # which means we have detected overlapping
                            # print("-----------------------------------------------------------")
                            # print(report[0]['chains'])
                            # print("-----------------------------------------------------------")
                            scores[ex][i] -= 1000
                        else:
                            break
                    else:
                        break
            
        return scores


class DataPortraitsSkipLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
    Contrarily to [`MinLengthLogitsProcessor`], this processor ignores the prompt.

    Args:
        prompt_length_to_skip (`int`):
            The input tokens length. Not a valid argument when used with `generate` as it will automatically assign the
            input length.
        min_new_tokens (`int`):
            The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, prompt: str, width: int, tokenizer, portrait, tokenized_prompt, consecutive, n, skip_tokens=10, acs_threshold=50):
        self.prompt = prompt
        self.tokenized_prompt = tokenized_prompt 
        self.n = n
        self.width = width
        self.tokenizer = tokenizer
        self.portrait = portrait
        self.consecutive = consecutive # Whether do consecutive decection or not.
        self.acs_threshold = acs_threshold # For non-consecutive situation. When ACS reaches to threshold we do intervention
        self.t = skip_tokens #(TODO) The number of tokens to skip detection. need to become a variable in the future, it can also be tied to n and width
        self.logits_list = []
        self.counter = 0 # record the number of being called

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids_orig = input_ids
        input_ids = input_ids.cpu().detach().numpy()
        
        # Order the tokens by their likelihood
        order = torch.argsort(-scores, 1)
        order = order.cpu().detach().numpy()
        batch_size = input_ids.shape[0]
        
        # Set the likelihood to 0 for all the modest likely next tokens which would cause copyright infringement
        for ex in range(batch_size):
            if self.counter < self.t:
                self.logits_list.append(order[0].tolist()[:6])
                self.counter += 1
            else:
                self.counter = 0
                ids_to_check = input_ids[ex].tolist()
                sequence_to_check = ''.join(f"{num:06d}" for num in ids_to_check)
                tokenized_prompt_string = ''.join(f"{num:06d}" for num in self.tokenized_prompt.input_ids[0].tolist())
                res_sequence = sequence_to_check.replace(tokenized_prompt_string, "", 1)
                report = self.portrait.contains_from_text([res_sequence[-self.width:]], stride=1)
                if report[0]['chain_idxs'] != []:
                    matching_ids = np.concatenate(report[0]['chain_idxs']).tolist()
                    base_loc = len(self.tokenized_prompt.input_ids[0].tolist()) + int(len(res_sequence[:-self.width])/6)
                    res_base_loc = (len(res_sequence[:-self.width]) / 6)
                    print(matching_ids)
                    for id in matching_ids:
                        loc = int((id + self.n)/6 - 1) # id is the start of the n-gram, we need to substitude the end of n-gram; loc=0 is the res_sequence[-self.width] token
                        # assert loc > self.width/5 - self.t - 1 # assert the location is in the last t tokens. 
                        # assert input_ids_orig[ex][loc+base_loc] == self.logits_list[int(res_base_loc+loc)][0]
                        idx = self.logits_list[int(res_base_loc+loc)].index(input_ids_orig[ex][loc+base_loc])
                        if idx + 1 < 6:
                            input_ids_orig[ex][loc+base_loc] = self.logits_list[int(res_base_loc+loc)][idx+1] # Directly select the alternative token into input_ids
                        else:
                            input_ids_orig[ex][loc+base_loc] = self.logits_list[int(res_base_loc+loc)][-1]
                # report = self.portrait.contains_from_text([tokenized_prompt_string[-self.width:]], stride=5)
                # self.logits_list = []
                self.logits_list.append(order[0].tolist()[:6])
        return scores, input_ids_orig
    
class TopKPerturbationLogitsProcessor(LogitsProcessor):
    
    def __init__(self, tokenizer, model, std=0.1):
        self.tokenizer = tokenizer
        self.model = model
        self.std = std

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = input_ids.cpu().detach().numpy()
        top_k = self.model.generation_config.top_k
        # Order the tokens by their likelihood
        order = torch.argsort(-scores, 1)
        order = order.cpu().detach().numpy()
        order_top_k = order[:, :top_k]
        batch_size = input_ids.shape[0]
        
        for ex in range(batch_size):
            # add gaussian noise to the top_k scores
            noise = torch.normal(0, self.std, size=(top_k,), device=scores.device)
            scores[ex][order_top_k[ex]] += noise
        return scores