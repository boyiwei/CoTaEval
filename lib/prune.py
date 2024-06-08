import os
import time
import heapq
import torch
import torch.nn as nn
import pickle
from .data import get_loaders
import json
import random
from functools import reduce
import heapq
import re

class ActLinear(nn.Module):
    """
        drop in replacement of nn.Linear
    """
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        #self.register_buffer('activation_norms', torch.zeros([base.in_features], device=self.base.weight.device, requires_grad=False))
        self.activation_norms = torch.zeros([base.in_features], device=self.base.weight.device, requires_grad=False)
        self.n_samples = 0
        self.record_activation = True

    def clear_act_buffer(self):
        self.activation_norms.fill_(0.)
        self.n_samples = 0

    def forward(self, x):

        # DEBUG:
        # print("input zero percentage", (x==0).sum() / x.numel() )

        if self.record_activation:
            if hasattr(self, 'mask') and self.mask is not None:
                x_ = x[self.mask]
            else:
                x_ = x

            bs = x_.nelement() // x_.shape[-1]
            self.activation_norms = self.activation_norms * ( self.n_samples/(self.n_samples+bs) )  +  (x_ * x_).view(-1, x_.shape[-1]).sum(dim=0) * ( 1. / (self.n_samples+bs) )
            self.n_samples += bs

        out = self.base(x)
        return out


class no_act_recording:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for (name, module) in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for (name, module) in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = True


def make_Act(model, verbose=False):
    replace_map = dict()
    for (name, module) in model.named_modules():
        if isinstance(module, nn.Linear):
            replace_map[name] = ActLinear(module)

    for (name, module) in model.named_modules():
        if verbose:
            print("current:", name)
        for k, v in replace_map.items():
            k_ = k.split('.')
            name_prefix, name_suffix = '.'.join(k_[:-1]), k_[-1]
            if name_prefix == "": # outer layer
                if name == name_suffix:
                    if verbose:
                        print(" not modifying ", name_suffix)
                    #setattr(model, name_suffix, v)
            elif name == name_prefix:
                if verbose:
                    print("    modifying ", name_suffix, "inside", name)
                setattr(module, name_suffix, v)
    return model


def revert_Act_to_Linear(model):
    """
    Reverts ActLinear modules back to their original nn.Linear layers.
    """
    for (name, module) in model.named_modules():
        if isinstance(module, ActLinear):
            # Extract the base nn.Linear module from ActLinear
            linear_module = module.base
            # Navigate to the parent module of the ActLinear module
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            print(f'Reverting {name}, parent: {parent_name}')
            parent_module = model if parent_name == '' else reduce(getattr, parent_name.split('.'), model)
            # Replace the ActLinear module with the extracted nn.Linear module
            setattr(parent_module, name.split('.')[-1], linear_module)

    return model


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count)/total_params


def check_sparsity_layerwise(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()
            print(f"{float((W==0).sum().item())/W.numel():.6f},")

    model.config.use_cache = use_cache
    
    
def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    #inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps = []
    tars = []
    attention_mask = []
    position_ids = []
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_mask.append(kwargs['attention_mask'])
            position_ids.append(kwargs['position_ids'])
            #inps[cache['i']] = inp
            #cache['i'] += 1
            #cache['attention_mask'] = kwargs['attention_mask']
            #cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            tars.append(batch[1])
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = [None for _ in range(nsamples)]
    model.config.use_cache = use_cache

    return inps, outs, tars, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_wandg_set_difference(args, model, tokenizer, model_base=None, device=torch.device("cuda:0"), prune_n=0, prune_m=0, prune_data='align_short', p=0.5, q=0.5):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers
    metric1 = 'alpaca_cleaned_no_safety'
    metric2 = prune_data

    print("prune p = {}, q = {}, with metric1 = {}, metric2 = {}".format(p, q, metric1, metric2))
    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if not args.prune_part:
            for name in subset:
                print(f"pruning layer {i} name {name}")
                if args.model == 'llama2-7b-chat-hf':
                    W_metric1 = pickle.load(open(f'out/llama2-7b-chat-hf/unstructured/wandg/{metric1}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl', 'rb'))
                    W_metric2 = pickle.load(open(f'out/llama2-7b-chat-hf/unstructured/wandg/{metric2}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl', 'rb'))
                elif args.model == 'llama2-13b-chat-hf':
                    W_metric1 = pickle.load(open(f'out/llama2-13b-chat-hf/unstructured/wandg/{metric1}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl', 'rb'))
                    W_metric2 = pickle.load(open(f'out/llama2-13b-chat-hf/unstructured/wandg/{metric2}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl', 'rb'))
                elif args.model == 'gemma-7b-hf':
                    W_metric1 = pickle.load(open(f'out/gemma-7b-hf/unstructured/wandg/{metric1}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl', 'rb'))
                    W_metric2 = pickle.load(open(f'out/gemma-7b-hf/unstructured/wandg/{metric2}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl', 'rb'))
                else:
                    raise NotImplementedError

                top_p = int(p * W_metric1.shape[1] * W_metric1.shape[0]) # top_p utility
                top_q = int(q * W_metric2.shape[1] * W_metric2.shape[0]) # top_q safety
                
                
                top_p_indices = torch.topk(W_metric1.flatten(), top_p, largest=True)[1]
                top_q_indices = torch.topk(W_metric2.flatten(), top_q, largest=True)[1]
                unique_p = torch.unique(top_p_indices)
                unique_q = torch.unique(top_q_indices)

                # Create a boolean mask for elements in unique_q that are not in unique_p
                mask = ~torch.isin(unique_q, unique_p)

                # Apply the mask to unique_q to get filtered_indices
                filtered_indices = unique_q[mask]
                weight_dim = subset[name].weight.data.shape[1]
                filtered_indices_rows = filtered_indices // weight_dim
                filtered_indices_cols = filtered_indices % weight_dim
                
                
                assert args.dump_wanda_score == False # Only pruning from the saved score, won't save score again

                W_mask = (torch.zeros_like(subset[name].weight.data) == 1)  
                W_mask[filtered_indices_rows, filtered_indices_cols] = True # prune weights that has relatively high safety while not in top utility scores

                if args.recover_from_base:
                    assert model_base is not None
                    subset[name].weight.data[W_mask] = subset_base[name].weight.data[W_mask]    # patch with the base model's weights
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
        else:
            # args.prune_part == True. We only prune the layer with low jaccard index, which is:
            # layer 0 mlp_down_proj
            # layer 1 self_attn._proj and mlp_down_proj
            # rest of layers: self_attn.o_proj, mlp_gate_proj, mlp_down_proj, mlp_up_proj
            for name in subset:
                condition = ((i == 0) and (name == 'mlp.down_proj')) or \
                            ((i == 1) and ((name == 'self_attn.o_proj') or (name == 'mlp.down_proj'))) or \
                            ((i > 1) and ((name == 'self_attn.o_proj') or (name == 'mlp.gate_proj') or (name == 'mlp.down_proj') or (name == 'mlp.up_proj')))
                if condition:
                    W_metric1 = pickle.load(open(f'out/llama2-7b-chat-hf/unstructured/wandg/{metric1}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl', 'rb'))
                    W_metric2 = pickle.load(open(f'out/llama2-7b-chat-hf/unstructured/wandg/{metric2}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl', 'rb'))
                    top_p = int(p * W_metric1.shape[1] * W_metric1.shape[0])
                    top_q = int(q * W_metric2.shape[1] * W_metric2.shape[0])
                    
                    
                    top_p_indices = torch.topk(W_metric1.flatten(), top_p, largest=True)[1]
                    top_q_indices = torch.topk(W_metric2.flatten(), top_q, largest=True)[1]
                    unique_p = torch.unique(top_p_indices)
                    unique_q = torch.unique(top_q_indices)

                    # Create a boolean mask for elements in unique_p that are not in unique_q
                    mask = ~torch.isin(unique_q, unique_p)

                    # Apply the mask to unique_p to get filtered_indices
                    filtered_indices = unique_q[mask]
                    weight_dim = subset[name].weight.data.shape[1]
                    filtered_indices_rows = filtered_indices // weight_dim
                    filtered_indices_cols = filtered_indices % weight_dim
                    
                    assert args.dump_wanda_score == False # Only pruning from the saved score, won't save score again

                    W_mask = (torch.zeros_like(subset[name].weight.data) == 1)  
                    W_mask[filtered_indices_rows, filtered_indices_cols] = True # prune weights that has relatively high safety while not in top utility scores

                    if args.recover_from_base:
                        assert model_base is not None
                        subset[name].weight.data[W_mask] = subset_base[name].weight.data[W_mask]    # patch with the base model's weights
                    else:
                        subset[name].weight.data[W_mask] = 0  ## set weights to zero


def prune_wandg(args, model, tokenizer, model_base=None, device=torch.device("cuda:0"), prune_n=0, prune_m=0, prune_data='wikitext'):
    model = make_Act(model, verbose=False)

    print(f"loading calibdation data {prune_data}")
    assert prune_data in ['wikitext', 'alpaca', 'alpaca_cleaned', 'alpaca_cleaned_no_safety', 'align', 'align_short', 'misalign']
    dataloader, _ = get_loaders(prune_data, nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer, disentangle=args.disentangle)
    print("dataset loading complete")

    num_hidden_layers = model.config.num_hidden_layers
    saved_grad = {}
    for layer in range(num_hidden_layers):
        layer_filter_fn = lambda x: f'layers.{layer}.' in x 

        model.zero_grad()
        model.requires_grad_(False)
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                print("enabling grad for ", name)
                module.base.requires_grad_(True)
                saved_grad[name] = torch.zeros_like(module.base.weight, device = module.base.weight.device)
                module.base.zero_grad()

        for batch in dataloader:
            inp, tar = batch[0].to(device), batch[1].to(device)
            assert args.disentangle, 'should run in disentangle mode'
            model.zero_grad()
            with no_act_recording(model):
                loss = model(input_ids = inp, labels = tar)[0]
            loss.backward()
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    saved_grad[name] += module.base.weight.grad.abs()

        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                module.base.weight.grad.copy_(saved_grad[name])
                saved_grad.pop(name)
        _prune_core(args, model, model_base, prune_n, prune_m, prune_mode = 'gradient', name_filter_fn = layer_filter_fn)
        #print(torch.cuda.memory_allocated() /1024/1024/1024)

    model = revert_Act_to_Linear(model)
    model.zero_grad() # freeze gradient to save cuda memory

def _prune_core(args, model, model_base=None, prune_n=0, prune_m=0, prune_mode = 'activation', name_filter_fn = None):
    """
        data aware
    """
    assert not args.prune_part, "Warning: prune_part is not supported"
    # assert not args.neg_prune, "Warning: neg_prune is not supported"
    prune_data = args.prune_data
    for name, module in model.named_modules():
        if name_filter_fn is not None and not name_filter_fn(name):
            continue

        if isinstance(module, ActLinear):
            print('pruning:', name)
            
            i = re.search(r'\d+', name)
            if i:
                i = int(i.group())
            else:
                i = 0
                
            print("layer", i)

            if model_base is not None:
                module_base = model_base.get_submodule(name)

            if args.use_diff:
                magnitude = torch.abs(module.base.weight.data - module_base.weight.data)
            else:
                magnitude = torch.abs(module.base.weight.data)

            if prune_mode == 'activation':
                act = (module.activation_norms ** 0.5).unsqueeze(0)
            elif prune_mode == 'gradient':
                act = module.base.weight.grad.abs()
            else:
                raise NotImplemented

            W_metric = magnitude * act
            if args.neg_prune:
                W_metric = -W_metric

            # copied from lib/prune.py prune_wanda:

            if args.dump_wanda_score:
                # Only save the score, no pruning
                save_folder = os.path.join(args.save, f'wanda_score/') # We assume that args.save has contained the information of pruned data.
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                if args.use_diff:
                    target_file = os.path.join(save_folder, f'W_metric_layer_{i}_name_{name}_weight_diff.pkl')
                else:
                    target_file = os.path.join(save_folder, f'W_metric_layer_{i}_name_{name}_weight.pkl')
                with open(target_file, 'wb') as f:
                    print("Writing W_metric in layer {} and name {} with {} to the file".format(i, name, prune_data))
                    pickle.dump(W_metric, f)
                continue


            # log W_metric to the log file

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            if args.recover_from_base:
                module.base.weight.data[W_mask] = module_base.weight.data[W_mask]    # patch with the base model's weights
            else:
                module.base.weight.data[W_mask] = 0  ## set weights to zero


def get_mask(model, neg_prune=False):
    """
    Save mask for the unstructured pruned model (for ft-attack evaluation).
    `neg_prune`:
        - if `args.neg_prune` is False (bottom pruning), save the mask as True for the weights not pruned.
        - if `args.neg_prune` is True (top pruning), save the mask as True for the pruned weights.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    mask = {}
    
    mask_num = 0
    total_num = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            mask[name] = module.weight.data.abs().lt(1e-8).to("cpu").detach()
            if neg_prune is False:
                mask[name] = ~mask[name]
            
            mask_num += mask[name].eq(True).int().sum()
            total_num += mask[name].numel()
    
    print(f"{(100 * mask_num / total_num):.2f}% entries are True in mask.")
    return mask