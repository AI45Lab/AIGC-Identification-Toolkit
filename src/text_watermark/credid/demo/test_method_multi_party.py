import json
import os
import torch
import sys
import logging
import numpy as np
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
from src.embedding.load import load_config, prepare_input_from_prompts, load_model_and_tokenizer

def main():
    # Configure logging
    logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
    torch.set_printoptions(threshold=np.inf)

    # Load models and tokenizers
    tokenizer_1 = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
    model_1 = AutoModelForCausalLM.from_pretrained("gpt2", trust_remote_code=True)
    model_1 = model_1.to('cuda:1')

    tokenizer_2 = AutoTokenizer.from_pretrained("facebook/opt-1.3b", trust_remote_code=True)
    model_2 = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", trust_remote_code=True)
    model_2 = model_2.to('cuda:2')

    tokenizer_3 = AutoTokenizer.from_pretrained("facebook/opt-2.7b", trust_remote_code=True)
    model_3 = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", trust_remote_code=True)
    model_3 = model_3.to('cuda:3')

    # Load dataset
    c4_sliced_and_filted = load_from_disk('./datasets/c4-train.00000-of-00512_sliced')
    c4_sliced_and_filted = c4_sliced_and_filted['train'].shuffle(seed=42).select(range(100))

    sample_idx = 68
    input_text = c4_sliced_and_filted[sample_idx]['text']
    tokenized_input = tokenizer_1(input_text, return_tensors='pt').to(model_1.device)
    tokenized_input = truncate(tokenized_input, max_length=300)

    # Set LogitsProcessor
    min_length_processor = MinLengthLogitsProcessor(min_length=1000, eos_token_id=tokenizer_1.eos_token_id)
    repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=1.5)

    # Initialize message models and processors
    lm_message_model_1 = LMMessageModel(
        tokenizer=tokenizer_1,
        lm_model=model_1,
        lm_tokenizer=tokenizer_1,
        delta=5.5,
        lm_prefix_len=10,
        lm_topk=-1,
        message_code_len=10,
        random_permutation_num=50,
        shifts=[4, 2, 21, 28, 14, 4, 8, 8, 3, 14, 24, 31, 3, 28, 2]
    )
    wm_precessor_message_model_1 = WmProcessorMessageModel(
        message_model=lm_message_model_1,
        tokenizer=tokenizer_1,
        encode_ratio=5,
        max_confidence_lbd=0.5,
        strategy='max_confidence_updated',
        message=[11, 22, 333]
    )

    lm_message_model_2 = LMMessageModel(
        tokenizer=tokenizer_2,
        lm_model=model_2,
        lm_tokenizer=tokenizer_2,
        delta=5.5,
        lm_prefix_len=10,
        lm_topk=-1,
        message_code_len=10,
        random_permutation_num=100,
        shifts=[21, 24, 5, 8, 14, 2, 4, 28, 31, 3, 8, 14, 2, 4, 28]
    )
    wm_precessor_message_model_2 = WmProcessorMessageModel(
        message_model=lm_message_model_2,
        tokenizer=tokenizer_2,
        encode_ratio=5,
        max_confidence_lbd=0.5,
        strategy='max_confidence_updated',
        message=[44, 55, 666]
    )

    lm_message_model_3 = LMMessageModel(
        tokenizer=tokenizer_3,
        lm_model=model_3,
        lm_tokenizer=tokenizer_3,
        delta=5.5,
        lm_prefix_len=10,
        lm_topk=-1,
        message_code_len=10,
        random_permutation_num=100,
        shifts=[3, 8, 24, 14, 2, 4, 8, 14, 2, 28, 3, 31, 4, 28, 21]
    )
    wm_precessor_message_model_3 = WmProcessorMessageModel(
        message_model=lm_message_model_3,
        tokenizer=tokenizer_3,
        encode_ratio=5,
        max_confidence_lbd=0.5,
        strategy='max_confidence_updated',
        message=[77, 88, 999]
    )

    # Watermark embedding
    start_length = tokenized_input['input_ids'].shape[-1]

    # company1
    wm_precessor_message_model_1.start_length = start_length
    output_tokens = model_1.generate(
        **tokenized_input,
        max_new_tokens=160,
        num_beams=4,
        logits_processor=LogitsProcessorList([min_length_processor, repetition_processor, wm_precessor_message_model_1])
    )
    output_text_1 = tokenizer_1.decode(output_tokens[0][start_length:], skip_special_tokens=True)
    logging.info(output_text_1)
    prefix_and_output_text_1 = tokenizer_1.decode(output_tokens[0], skip_special_tokens=True)
    logging.warning(prefix_and_output_text_1)

    # company2
    tokenized_input_2 = {key: value.to('cuda:2') for key, value in tokenized_input.items()}
    wm_precessor_message_model_2.start_length = start_length
    output_tokens_2 = model_2.generate(
        **tokenized_input_2,
        max_new_tokens=160,
        num_beams=4,
        logits_processor=LogitsProcessorList([min_length_processor, repetition_processor, wm_precessor_message_model_2])
    )
    output_text_2 = tokenizer_2.decode(output_tokens_2[0][start_length:], skip_special_tokens=True)
    logging.info(output_text_2)
    prefix_and_output_text_2 = tokenizer_2.decode(output_tokens_2[0], skip_special_tokens=True)

    # company3
    tokenized_input_3 = {key: value.to('cuda:3') for key, value in tokenized_input.items()}
    wm_precessor_message_model_3.start_length = start_length
    output_tokens_3 = model_3.generate(
        **tokenized_input_3,
        max_new_tokens=160,
        num_beams=4,
        logits_processor=LogitsProcessorList([min_length_processor, repetition_processor, wm_precessor_message_model_3])
    )
    output_text_3 = tokenizer_3.decode(output_tokens_3[0][start_length:], skip_special_tokens=True)
    logging.info(output_text_3)
    prefix_and_output_text_3 = tokenizer_3.decode(output_tokens_3[0], skip_special_tokens=True)

    # Watermark detection
    def detect_watermark(output_text, wm_precessor_message_models, company_names):
        max_log_prob = -1
        best_company = None
        for wm_precessor_message_model, company_name in zip(wm_precessor_message_models, company_names):
            log_probs = wm_precessor_message_model.decode(output_text)
            confidence = log_probs[1][1][0][2]
            decoded_message = log_probs[0]
            print(f"{company_name} - Confidence: {confidence}")
            print(f"{company_name} - Decoded message: {decoded_message}")
            logging.info(f"{company_name} - Confidence: {confidence}")
            logging.info(f"{company_name} - Decoded message: {decoded_message}")
            if decoded_message > 0.5 and decoded_message > max_log_prob:
                max_log_prob = decoded_message
                best_company = company_name
        if best_company:
            print(f"Best company: {best_company}, log_probs[0]: {max_log_prob}")
            logging.info(f"Best company: {best_company}, log_probs[0]: {max_log_prob}")
        else:
            print("No company has log_probs[0] greater than 0.5")
            logging.info("No company has log_probs[0] greater than 0.5")
        logging.info("{:=^50s}".format("Split Line"))

    # Output 1
    wm_precessor_message_models = [wm_precessor_message_model_1, wm_precessor_message_model_2, wm_precessor_message_model_3]
    company_names = ["Company1", "Company2", "Company3"]
    detect_watermark(output_text_1, wm_precessor_message_models, company_names)

    # Output 2
    detect_watermark(output_text_2, wm_precessor_message_models, company_names)

    # Output 3
    detect_watermark(output_text_3, wm_precessor_message_models, company_names)

if __name__ == '__main__':
    main()
