import json
import os
import torch
import sys
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, MinLengthLogitsProcessor
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from watermarking.watermark_processor import RepetitionPenaltyLogitsProcessor
from src.utils.text_tools import truncate
from src.utils.load_local import load_local_model_or_tokenizer
from datasets import load_dataset, load_from_disk
from watermarking.CredID.message_models.lm_message_model import LMMessageModel
from watermarking.CredID.message_model_processor import WmProcessorMessageModel
from src.utils.watermark import compute_ppl_single
from src.embedding.load import load_config, prepare_input_from_prompts, load_model_and_tokenizer

def main():
    # Define the path of the configuration file
    # config_path = os.path.join(project_root, 'config', 'CTWL.json')
    config_path = os.path.join(project_root, 'config', 'CredID_simple.json')
    # Load the configuration
    config = load_config(config_path)
    
    # Configure logging
    logging.basicConfig(filename=config['log_file'], level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
    torch.set_printoptions(threshold=np.inf)
    
    #Load the model and tokenizer using the new function
    tokenizer, model = load_model_and_tokenizer(config) 

    tokenized_input = prepare_input_from_prompts(model.device,tokenizer, config, project_root)

    min_length_processor = MinLengthLogitsProcessor(min_length=config['min_length'], eos_token_id=tokenizer.eos_token_id)
    repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=config['penalty'])

    # Read the parameters of lm_message_model from the configuration file
    lm_message_model_params = config['lm_message_model_params']
    lm_message_model = LMMessageModel(
        tokenizer=tokenizer, 
        lm_model=model, 
        lm_tokenizer=tokenizer,
        delta=lm_message_model_params['delta'], 
        lm_prefix_len=lm_message_model_params['lm_prefix_len'], 
        lm_topk=lm_message_model_params['lm_topk'], 
        message_code_len=lm_message_model_params['message_code_len'], 
        random_permutation_num=lm_message_model_params['random_permutation_num']
    )

    # Read the parameters of wm_processor_message_model from the configuration file
    wm_processor_message_model_params = config['wm_processor_message_model_params']
    wm_precessor_message_model = WmProcessorMessageModel(
        message_model=lm_message_model, 
        tokenizer=tokenizer,
        encode_ratio=wm_processor_message_model_params['encode_ratio'], 
        max_confidence_lbd=wm_processor_message_model_params['max_confidence_lbd'], 
        strategy=wm_processor_message_model_params['strategy'], 
        message=wm_processor_message_model_params['message']
    )
    
    start_length = tokenized_input['input_ids'].shape[-1]
    wm_precessor_message_model.start_length = start_length
    output_tokens = model.generate(
        **tokenized_input, 
        max_new_tokens=config['max_new_tokens'], 
        num_beams=config['num_beams'],
        logits_processor=LogitsProcessorList([min_length_processor, repetition_processor, wm_precessor_message_model])
    )
    print("watermarked_text:",output_tokens)
    output_text = tokenizer.decode(output_tokens[0][start_length:], skip_special_tokens=True)
    logging.info(output_text)
    prefix_and_output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    log_probs = wm_precessor_message_model.decode(output_text)
    print("my watermark confidence:", log_probs[1][1][0][2])
    print("decoded message:", log_probs[0])
    
    
    oracle_tokenizer = AutoTokenizer.from_pretrained(config['oracle_model_name'])
    oracle_model = AutoModelForCausalLM.from_pretrained(config['oracle_model_name'],device_map="auto")#,torch_dtype=torch.float16)
    

    loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text,
                                oracle_model_name='huggyllama/llama-13b',
                                output_text=output_text,
                                oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
    print("loss,ppl:",loss, ppl)
    
    

if __name__ == '__main__':
    main()
