import json
import os
import torch
import sys
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, MinLengthLogitsProcessor
from importlib import reload

# Set project path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# Reload modules
import watermarking.watermark_processor
watermarking.watermark_processor = reload(watermarking.watermark_processor)
from watermarking.watermark_processor import RepetitionPenaltyLogitsProcessor
from src.utils.text_tools import truncate
from src.utils.load_local import load_local_model_or_tokenizer
from datasets import load_dataset, load_from_disk
import watermarking.CredID.message_models.lm_message_model
watermarking.CredID.message_models.lm_message_model = reload(watermarking.CredID.message_models.lm_message_model)
import watermarking.CredID.message_model_processor
watermarking.CredID.message_model_processor = reload(watermarking.CredID.message_model_processor)
from watermarking.CredID.message_models.lm_message_model import LMMessageModel
from watermarking.CredID.message_model_processor import WmProcessorMessageModel

def main():
    # Configure logging
    logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
    torch.set_printoptions(threshold=np.inf)

    # Model and data paths
    model_name = "huggyllama/llama-7b"
    path = "./my_watermark_result/lm_new_7_10/tiiuae-falcon-7b_1.5_1.5_300_50_250_42_42_10_4_4_1.0_10_-1_200_max_confidence_updated_0.5_huggyllama/llama-7b_1000.json"
    
    # Load result data
    with open(path, 'r') as f:
        results = json.load(f)

    texts = results['text']
    prefix_and_output_texts = results['prefix_and_output_text']
    output_texts = results['output_text']
    analysis_results = results.get('analysis', {})

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Set LogitsProcessor
    min_length_processor = MinLengthLogitsProcessor(min_length=10000, eos_token_id=tokenizer.eos_token_id)
    repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=1.5)

    # Initialize message model and processor
    lm_message_model_update = LMMessageModel(
        tokenizer=tokenizer,
        lm_model=model,
        lm_tokenizer=tokenizer,
        delta=1.5,
        lm_prefix_len=10,
        lm_topk=-1,
        message_code_len=10,
        random_permutation_num=200,
        shifts=[21, 24, 3, 8, 14, 2, 4, 28, 31, 3, 8, 14, 2, 4, 28]
    )
    wm_precessor_message_model_update = WmProcessorMessageModel(
        message_model=lm_message_model_update,
        tokenizer=tokenizer,
        encode_ratio=10,
        max_confidence_lbd=0.5,
        strategy='max_confidence_updated',
        message=[42]
    )

    confidences = []
    decoded_messages = []
    accs = []
    decoded_message_confidences = []

    # Process output texts
    for output_text in tqdm(output_texts):
        decoded_message, other_information = wm_precessor_message_model_update.decode(
            output_text[:len(output_text) * 2 // 3],
            disable_tqdm=True
        )
        confidence = other_information[1][0][2]
        available_message_num = 110 // (int(10 * 10))
        acc = decoded_message[:available_message_num] == [42]

        confidences.append(confidence)
        decoded_messages.append(decoded_message)
        decoded_message_confidences.append({'decoded_message': decoded_message, 'confidence': confidence})
        accs.append(acc)

    analysis_results['message_confidence'] = decoded_message_confidences
    results['analysis'] = analysis_results

    # Save results
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)

    # Read and analyze data
    with open(path, 'r') as f:
        data = json.load(f)

    confidence1 = [item[0][2] for item in data['confidence']]
    con = [item['confidence'] for item in data['analysis_att_ori']['message_confidence']]
    confidence2 = [item['confidence'] for item in data['analysis_att_gemma']['message_confidence']]
    confidence3 = [item['confidence'] for item in data['analysis_att_llama']['message_confidence']]
    message = [item['decoded_message'] for item in data['analysis_att_ori']['message_confidence']]
    
    count = 0
    for i in range(len(message)):
        if message[i] != [42]:
            print(f"  Decoded message: {message[i]}")
            print(f"  ori confidence: {con[i]}")
            print(f"  gemma confidence: {confidence2[i]}")
            print(f"  llama confidence: {confidence3[i]}")
            count += 1
    print(count)

if __name__ == '__main__':
    main()
