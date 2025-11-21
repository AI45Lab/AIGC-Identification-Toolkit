from typing import List, Optional, Callable

import time
import random
import math
import torch
import numpy as np

from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor, LogitsProcessorList, set_seed


def tokenize_and_truncate(example: dict,
                          completion_length: int = None,
                          prompt_length: int = None,
                          hf_model_name: str = None,
                          tokenizer=None,
                          model_max_seq_len: int = 4096):
    
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert "text" in example, "expects 'text' field to be present"
    
    inputs = tokenizer.encode(example["text"], return_tensors="pt", truncation=True,
                              max_length=model_max_seq_len)
    example.update({"untruncated_inputs": inputs})

    if (completion_length is not None) and (prompt_length is None):
        
        slice_length = min(inputs.shape[1] - 1, completion_length)
    elif (prompt_length is not None) and (completion_length is None):
        desired_comp_len = (inputs.shape[1] - 1) - prompt_length
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError((
                         f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                         f" but got completion_length:{completion_length},prompt_length:{prompt_length}"))

    
    inputs = inputs[:, :inputs.shape[1] - slice_length]
    
    if "t5" in hf_model_name or "T0" in hf_model_name:
        inputs[0, -1] = 1
    
    example.update({"inputs": inputs})
    return example


def score_sequence(inputs: Tensor = None,
                   outputs: Tensor = None,
                   tokenizer: Tokenizer = None,
                   
                   initial_seed: int = None,
                   dynamic_seed: str = None,
                   bl_proportion: float = None,
                   use_cuda: bool = True,
                   record_hits: bool = False,
                   debug: bool = True,
                   
                   ):
    assert (inputs is not None) and \
           (outputs is not None) and \
           (tokenizer is not None), "output tensor, tokenizer, and bl params req'd"
    

    vocabulary = list(tokenizer.get_vocab().values())
    vocab_size = len(vocabulary)

    model_generations = outputs.tolist()[0]  
    
    toks_generated = model_generations
    num_toks_generated = len(toks_generated)

    
    
    

    

    

    if initial_seed is not None:
        random.seed(initial_seed)

    device = (torch.device("cuda") if use_cuda else torch.device("cpu"))
    g_cuda = torch.Generator(device=device)
    large_prime = 15485863

    bl_hits, hit_list = 0, []

    prev_token = inputs[0][-1].item()
    

    
    for idx, tok_gend in enumerate(toks_generated):

        

        if dynamic_seed == "initial":
            g_cuda.manual_seed(large_prime * initial_seed)
        elif dynamic_seed == "markov_1":
            g_cuda.manual_seed(large_prime * prev_token)
        elif dynamic_seed is None:
            
            pass

        bl_ct = int(vocab_size * bl_proportion)
        posthoc_blacklist = torch.randperm(vocab_size, device=device, generator=g_cuda)[
                            :bl_ct]  

        tok_in_ph_bl = tok_gend in posthoc_blacklist
        if tok_in_ph_bl:
            bl_hits += 1
            hit_list.append(True)
        else:
            hit_list.append(False)

        if debug:
            decoded_token = tokenizer.decode(tok_gend, skip_special_tokens=True)
            print(f"Token generated: '{decoded_token}' was in the blacklist {tok_in_ph_bl}")

        prev_token = tok_gend

    if debug:
        print(
            f"wl hits / num tokens : {num_toks_generated - bl_hits}/{num_toks_generated} = {(num_toks_generated - bl_hits) / num_toks_generated:.02f}")
        print(
            f"bl hits / num tokens : {bl_hits}/{num_toks_generated} = {bl_hits / num_toks_generated:.02f}")

    if record_hits:
        return bl_hits, num_toks_generated, hit_list
    
    return bl_hits, num_toks_generated


def tokenize_for_generation(example: dict,
                            idx: int,
                            max_new_tokens: int = None,
                            min_prompt_tokens: int = None,
                            hf_model_name: str = None,
                            tokenizer: Tokenizer = None,
                            model: torch.nn.Module = None):
    
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    
    example = tokenize_and_truncate(example,
                                    completion_length=max_new_tokens,
                                    prompt_length=min_prompt_tokens,
                                    hf_model_name=hf_model_name,
                                    tokenizer=tokenizer,
                                    
                                    model_max_seq_len=None)
    inputs = example["inputs"]
    
    untruncated_inputs = example["untruncated_inputs"]

    
    re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
    example.update({"truncated_input": re_decoded_input})

    
    decoded_untruncated_input = \
    tokenizer.batch_decode(untruncated_inputs, skip_special_tokens=True)[0]
    example.update({"baseline_completion": decoded_untruncated_input.replace(re_decoded_input, "")})

    example.update({
        "orig_sample_length": untruncated_inputs.shape[1],
        "prompt_length": inputs.shape[1],
        "real_completion_length": untruncated_inputs.shape[1] - inputs.shape[1],
    })
    return example


def generate_completions(example: dict,
                         idx: int,
                         max_new_tokens: int = None,
                         hf_model_name: str = None,
                         tokenizer: Tokenizer = None,
                         model: torch.nn.Module = None,
                         no_bl_partial: Callable = None,
                         w_bl_partial: Callable = None,
                         
                         bl_processor_list: LogitsProcessorList = None):
    
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    
    
    
    
    
    
    
    
    

    
    
    

    
    
    

    inputs = example["inputs"]
    re_decoded_input = example["truncated_input"]

    
    with torch.no_grad():

        samples_taken = 0
        max_retries = 10
        success = False
        while (success is False) and (samples_taken < max_retries):
            samples_taken += 1

            

            start_generation = time.time()
            outputs_no_bl = no_bl_partial(inputs.to(model.device))
            example["no_bl_gen_time"] = time.time() - start_generation

            

            start_generation = time.time()
            outputs_w_bl = w_bl_partial(inputs.to(model.device))
            example["w_bl_gen_time"] = time.time() - start_generation

            
            
            
            
            

            
            
            
            

            if bl_processor_list:
                if bl_processor_list[0].bl_ids is not None:
                    example["bl_ids"] = bl_processor_list[0].get_and_clear_stored_bl_ids()
                if bl_processor_list[0].spike_entropies is not None:
                    example["spike_entropies"] = bl_processor_list[
                        0].get_and_clear_stored_spike_ents()

            try:
                
                no_bl_decoded_output = \
                tokenizer.batch_decode(outputs_no_bl, skip_special_tokens=True)[0]
                example.update({"no_bl_output": no_bl_decoded_output.replace(re_decoded_input, "")})

                w_bl_decoded_output = \
                tokenizer.batch_decode(outputs_w_bl, skip_special_tokens=True)[0]
                example.update({"w_bl_output": w_bl_decoded_output.replace(re_decoded_input, "")})

                success = True

            except:
                
                print(f"Error while trying to decode the outputs of the model...")
                if samples_taken == 1:
                    print(f"truncated_input: {inputs.tolist()}")
                print(f"Result of attempt {samples_taken}")
                print(f"shape outputs_no_bl: {outputs_no_bl.shape}")
                no_bl_toks = outputs_no_bl.tolist()[0]
                print(f"outputs_no_bl: {no_bl_toks}")
                print(f"outputs_no_bl min: {min(no_bl_toks)}")
                print(f"outputs_no_bl max: {max(no_bl_toks)}")

                print(f"shape outputs_w_bl: {outputs_w_bl.shape}")
                w_bl_toks = outputs_w_bl.tolist()[0]
                print(f"outputs_w_bl: {w_bl_toks}")
                print(f"outputs_w_bl min: {min(w_bl_toks)}")
                print(f"outputs_w_bl max: {max(w_bl_toks)}")

        if success is False:
            print(
                f"Unable to get both a no_bl and w_bl output that were decodeable after {samples_taken} tries, returning empty strings.")
            example.update({"no_bl_output": ""})
            example.update({"w_bl_output": ""})
            if bl_processor_list:
                if bl_processor_list[0].bl_ids is not None:
                    example["bl_ids"] = []
                if bl_processor_list[0].spike_entropies is not None:
                    example["spike_entropies"] = []

    
    

    example.update({
        
        "no_bl_num_tokens_generated": outputs_no_bl.shape[1] - inputs.shape[1],
        "w_bl_num_tokens_generated": outputs_w_bl.shape[1] - inputs.shape[1]
    })
    example.update({
        "no_bl_sec_per_tok": example["no_bl_gen_time"] / example["no_bl_num_tokens_generated"],
        "no_bl_tok_per_sec": example["no_bl_num_tokens_generated"] / example["no_bl_gen_time"],
        "w_bl_sec_per_tok": example["w_bl_gen_time"] / example["w_bl_num_tokens_generated"],
        "w_bl_tok_per_sec": example["w_bl_num_tokens_generated"] / example["w_bl_gen_time"],
    })

    
    
    
    

    return example


def compute_bl_metrics(example: dict,
                       idx: int,
                       hf_model_name: str = None,
                       tokenizer: Tokenizer = None,
                       initial_seed: int = None,
                       dynamic_seed: str = None,
                       bl_proportion: float = None,
                       use_cuda: bool = None,
                       record_hits: bool = False,
                       limit_output_tokens: int = 0):
    
    
    baseline_before = example["baseline_completion"]
    example["baseline_completion"] = baseline_before.replace(example["truncated_input"][:-1], "")
    if example["baseline_completion"] != baseline_before:
        print("baseline input replacement bug occurred!")

    no_bl_before = example["no_bl_output"]
    example["no_bl_output"] = no_bl_before.replace(example["truncated_input"][:-1], "")
    if example["no_bl_output"] != no_bl_before:
        print("no_bl_output input replacement bug occurred!")

    w_bl_before = example["w_bl_output"]
    example["w_bl_output"] = w_bl_before.replace(example["truncated_input"][:-1], "")
    if example["w_bl_output"] != w_bl_before:
        print("w_bl_output input replacement bug occurred!")

    if ("w_bl_output_attacked" in example):
        w_bl_attacked_before = example["w_bl_output_attacked"]
        example["w_bl_output_attacked"] = w_bl_attacked_before.replace(
            example["truncated_input"][:-1], "")
        if example["w_bl_output_attacked"] != w_bl_attacked_before:
            print("w_bl_output_attacked input replacement bug occurred!")

    

    
    inputs = tokenize_and_truncate({"text": example["truncated_input"]},
                                   completion_length=0,
                                   hf_model_name=hf_model_name,
                                   tokenizer=tokenizer)["inputs"]

    baseline_outputs = tokenize_and_truncate({"text": example["baseline_completion"]},
                                             completion_length=0,
                                             hf_model_name=hf_model_name,
                                             tokenizer=tokenizer)["inputs"][:, 1:]

    no_bl_outputs = tokenize_and_truncate({"text": example["no_bl_output"]},
                                          completion_length=0,
                                          hf_model_name=hf_model_name,
                                          tokenizer=tokenizer)["inputs"][:, 1:]

    w_bl_outputs = tokenize_and_truncate({"text": example["w_bl_output"]},
                                         completion_length=0,
                                         hf_model_name=hf_model_name,
                                         tokenizer=tokenizer)["inputs"][:, 1:]
    if "w_bl_output_attacked" in example:
        w_bl_attacked_outputs = tokenize_and_truncate({"text": example["w_bl_output_attacked"]},
                                                      completion_length=0,
                                                      hf_model_name=hf_model_name,
                                                      tokenizer=tokenizer)["inputs"][:, 1:]
    else:
        w_bl_attacked_outputs = None

    if limit_output_tokens > 0:
        example["orig_baseline_completion"] = example["baseline_completion"]
        example["orig_real_completion_length"] = example["real_completion_length"]
        baseline_outputs = baseline_outputs[:, :limit_output_tokens]
        example["real_completion_length"] = baseline_outputs.shape[1]
        example["baseline_completion"] = \
        tokenizer.batch_decode(baseline_outputs, skip_special_tokens=True)[0]

        example["orig_no_bl_output"] = example["no_bl_output"]
        example["orig_no_bl_num_tokens_generated"] = example["no_bl_num_tokens_generated"]
        no_bl_outputs = no_bl_outputs[:, :limit_output_tokens]
        example["no_bl_num_tokens_generated"] = no_bl_outputs.shape[1]
        example["no_bl_output"] = tokenizer.batch_decode(no_bl_outputs, skip_special_tokens=True)[0]

        example["orig_w_bl_output"] = example["w_bl_output"]
        example["orig_w_bl_num_tokens_generated"] = example["w_bl_num_tokens_generated"]
        w_bl_outputs = w_bl_outputs[:, :limit_output_tokens]
        example["w_bl_num_tokens_generated"] = w_bl_outputs.shape[1]
        example["w_bl_output"] = tokenizer.batch_decode(w_bl_outputs, skip_special_tokens=True)[0]

        example["orig_spike_entropies"] = example["spike_entropies"]
        example["spike_entropies"] = [example["spike_entropies"][0][:limit_output_tokens]]

        if "w_bl_output_attacked" in example:
            
            example["orig_w_bl_output_attacked"] = example["w_bl_output_attacked"]
            
            w_bl_attacked_outputs = w_bl_attacked_outputs[:, :limit_output_tokens]
            example["w_bl_attacked_num_tokens_generated"] = w_bl_attacked_outputs.shape[1]
            example["w_bl_output_attacked"] = \
            tokenizer.batch_decode(w_bl_attacked_outputs, skip_special_tokens=True)[0]

    
    result = score_sequence(inputs=inputs,
                            outputs=baseline_outputs,  
                            initial_seed=initial_seed,
                            dynamic_seed=dynamic_seed,
                            bl_proportion=bl_proportion,
                            tokenizer=tokenizer,
                            use_cuda=use_cuda,
                            record_hits=record_hits,
                            debug=False)
    if record_hits:
        bl_hits, num_toks_gend, hit_list = result
    else:
        bl_hits, num_toks_gend = result
    example.update({"baseline_num_toks_gend_eq_0": (num_toks_gend == 0)})
    
    

    if num_toks_gend == 0:
        
        wl_frac = -1
        bl_frac = -1
    else:
        wl_frac = (num_toks_gend - bl_hits) / num_toks_gend
        bl_frac = bl_hits / num_toks_gend
    baseline_stats = {
        "baseline_whitelist_fraction": wl_frac,
        "baseline_blacklist_fraction": bl_frac
    }
    example.update(baseline_stats)
    if record_hits: example.update({"baseline_hit_list": hit_list})

    result = score_sequence(inputs=inputs,
                            outputs=no_bl_outputs,  
                            initial_seed=initial_seed,
                            dynamic_seed=dynamic_seed,
                            bl_proportion=bl_proportion,
                            tokenizer=tokenizer,
                            record_hits=record_hits,
                            debug=False)
    if record_hits:
        bl_hits, num_toks_gend, hit_list = result
    else:
        bl_hits, num_toks_gend = result
    example.update({"no_bl_num_toks_gend_eq_0": (num_toks_gend == 0)})
    
    

    if num_toks_gend == 0:
        
        wl_frac = -1
        bl_frac = -1
    else:
        wl_frac = (num_toks_gend - bl_hits) / num_toks_gend
        bl_frac = bl_hits / num_toks_gend
    no_bl_stats = {
        "no_bl_whitelist_fraction": wl_frac,
        "no_bl_blacklist_fraction": bl_frac
    }
    example.update(no_bl_stats)
    if record_hits: example.update({"no_bl_hit_list": hit_list})

    result = score_sequence(inputs=inputs,
                            outputs=w_bl_outputs,  
                            initial_seed=initial_seed,
                            dynamic_seed=dynamic_seed,
                            bl_proportion=bl_proportion,
                            tokenizer=tokenizer,
                            record_hits=record_hits,
                            
                            debug=False)
    if record_hits:
        bl_hits, num_toks_gend, hit_list = result
    else:
        bl_hits, num_toks_gend = result
    example.update({"w_bl_num_toks_gend_eq_0": (num_toks_gend == 0)})
    
    

    if num_toks_gend == 0:
        
        wl_frac = -1
        bl_frac = -1
    else:
        wl_frac = (num_toks_gend - bl_hits) / num_toks_gend
        bl_frac = bl_hits / num_toks_gend
    w_bl_stats = {
        "w_bl_whitelist_fraction": wl_frac,
        "w_bl_blacklist_fraction": bl_frac
    }
    example.update(w_bl_stats)
    if record_hits: example.update({"w_bl_hit_list": hit_list})

    if w_bl_attacked_outputs is not None:
        result = score_sequence(inputs=inputs,
                                outputs=w_bl_attacked_outputs,
                                
                                initial_seed=initial_seed,
                                dynamic_seed=dynamic_seed,
                                bl_proportion=bl_proportion,
                                tokenizer=tokenizer,
                                record_hits=record_hits,
                                
                                debug=False)
        if record_hits:
            bl_hits, num_toks_gend, hit_list = result
        else:
            bl_hits, num_toks_gend = result
        example.update({"w_bl_attacked_num_toks_gend_eq_0": (num_toks_gend == 0)})
        

        if num_toks_gend == 0:
            
            wl_frac = -1
            bl_frac = -1
        else:
            wl_frac = (num_toks_gend - bl_hits) / num_toks_gend
            bl_frac = bl_hits / num_toks_gend
        w_bl_attacked_stats = {
            "w_bl_attacked_num_tokens_generated": num_toks_gend,
            "w_bl_attacked_whitelist_fraction": wl_frac,
            "w_bl_attacked_blacklist_fraction": bl_frac
        }
        example.update(w_bl_attacked_stats)
        if record_hits: example.update({"w_bl_attacked_hit_list": hit_list})

    return example


def aggregate_bl_stats(example: dict, idx: int, stat_table: dict):
    stat_table["baseline_stats"]["whitelist_fraction"] += example["baseline_stats"][
        "whitelist_fraction"]
    stat_table["baseline_stats"]["blacklist_fraction"] += example["baseline_stats"][
        "blacklist_fraction"]

    stat_table["w_bl_stats"]["whitelist_fraction"] += example["w_bl_stats"]["whitelist_fraction"]
    stat_table["w_bl_stats"]["blacklist_fraction"] += example["w_bl_stats"]["blacklist_fraction"]

    stat_table["no_bl_stats"]["whitelist_fraction"] += example["no_bl_stats"]["whitelist_fraction"]
    stat_table["no_bl_stats"]["blacklist_fraction"] += example["no_bl_stats"]["blacklist_fraction"]

    stat_table["num_examples"] += 1

    return example


def compute_ppl_single(prefix_and_output_text=None,
                       output_text=None,
                       oracle_model_name=None,
                       oracle_model=None,
                       oracle_tokenizer=None):
    with torch.no_grad():
        tokd_prefix = tokenize_and_truncate({"text": prefix_and_output_text}, completion_length=0,
                                            hf_model_name=oracle_model_name,
                                            tokenizer=oracle_tokenizer,
                                            model_max_seq_len=oracle_model.config.max_position_embeddings)[
            "inputs"]
        tokd_inputs = tokd_prefix
        
        tokd_suffix = tokenize_and_truncate({"text": output_text}, completion_length=0,
                                            hf_model_name=oracle_model_name,
                                            tokenizer=oracle_tokenizer,
                                            model_max_seq_len=oracle_model.config.max_position_embeddings)[
            "inputs"]

        tokd_inputs = tokd_inputs.to(oracle_model.device)
        
        tokd_labels = tokd_inputs.clone().detach()
        tokd_labels[:, :tokd_labels.shape[1] - tokd_suffix.shape[1] + 1] = -100

        outputs = oracle_model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss  
        ppl = torch.tensor(math.exp(loss))

    return loss.item(), ppl.item()


def evaluate_generation_fluency(example: dict,
                                idx: int,
                                oracle_model_name=None,
                                oracle_model=None,
                                oracle_tokenizer=None):
    
    inputs_plus_baseline_output = f"{example['truncated_input']}{example['baseline_completion']}"
    baseline_output = f"{example['baseline_completion']}"

    inputs_plus_no_bl_output = f"{example['truncated_input']}{example['no_bl_output']}"
    no_bl_output = f"{example['no_bl_output']}"

    inputs_plus_w_bl_output = f"{example['truncated_input']}{example['w_bl_output']}"
    w_bl_output = f"{example['w_bl_output']}"

    
    loss, ppl = compute_ppl_single(inputs_plus_baseline_output, baseline_output, oracle_model_name,
                                   oracle_model, oracle_tokenizer)
    example["baseline_loss"] = loss
    example["baseline_ppl"] = ppl
    loss, ppl = compute_ppl_single(inputs_plus_no_bl_output, no_bl_output, oracle_model_name,
                                   oracle_model, oracle_tokenizer)
    example["no_bl_loss"] = loss
    example["no_bl_ppl"] = ppl
    loss, ppl = compute_ppl_single(inputs_plus_w_bl_output, w_bl_output, oracle_model_name,
                                   oracle_model, oracle_tokenizer)
    example["w_bl_loss"] = loss
    example["w_bl_ppl"] = ppl

    
    return example


def add_idx(example, idx):
    example.update({"idx": idx})
    return example


def check_input_lengths(example, idx, min_sample_len=0, min_prompt_len=0, min_completion_len=0):
    orig_sample_length = example["orig_sample_length"]
    prompt_length = example["prompt_length"]
    real_completion_length = example["real_completion_length"]

    

    conds = all([
        orig_sample_length >= min_sample_len,
        prompt_length >= min_prompt_len,
        real_completion_length >= min_completion_len,
    ])
    return conds


def check_output_lengths(example, min_output_len=0):
    no_bl_output_len = example["no_bl_num_tokens_generated"]
    w_bl_output_len = example["w_bl_num_tokens_generated"]
    conds = all([
        no_bl_output_len >= min_output_len,
        w_bl_output_len >= min_output_len,
    ])
    return conds


