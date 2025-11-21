from importlib import reload
import watermarking.watermark_processor
import torch
watermarking.watermark_processor = reload(watermarking.watermark_processor)
from watermarking.watermark_processor import RepetitionPenaltyLogitsProcessor
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, LogitsProcessor
from src.utils.text_tools import truncate
from src.utils.load_local import load_local_model_or_tokenizer
from datasets import load_dataset, load_from_disk
from watermarking.CredID.random_message_model_processor import WmProcessorRandomMessageModel
from watermarking.CredID.message_models.random_message_model import RandomMessageModel
import numpy as np

import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
torch.set_printoptions(threshold=np.inf)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

from importlib import reload
import watermarking.CredID.message_models.lm_message_model
watermarking.CredID.message_models.lm_message_model = reload(watermarking.CredID.message_models.lm_message_model)
import watermarking.CredID.message_model_processor
watermarking.CredID.message_model_processor = reload(watermarking.CredID.message_model_processor)
from watermarking.CredID.message_models.lm_message_model import LMMessageModel
from watermarking.CredID.message_model_processor import WmProcessorMessageModel
import json
import os
from tqdm import tqdm
import random
from dataclasses import dataclass
from scipy.signal import argrelextrema
from scipy.ndimage import uniform_filter1d


oracle_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b")
oracle_model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b",device_map="auto")#,torch_dtype=torch.float16)


from src.utils.watermark import compute_ppl_single

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
# model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b",device_map="auto")
# min_length_processor = MinLengthLogitsProcessor(min_length=10000,
#                                                 eos_token_id=tokenizer.eos_token_id)
# repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=1.5)

@dataclass
class RunControlArgs:
    cover_old: bool = True
    select_num: int = -1


class MessageFilter:
    def __init__(self, device: str):
        self.device = device

    def get_messages_and_confidences(self, log_probs, window_size, avg_pool_size):
        if not isinstance(log_probs, torch.Tensor):
            log_probs = torch.tensor(log_probs)
        log_probs = (log_probs > 0).float()
        log_probs = log_probs.transpose(0, 1)
        log_probs = log_probs.unsqueeze(1)
        log_probs = log_probs.to(self.device)
        # print(window_size)
        kernel = torch.ones((1, 1, window_size), device=log_probs.device)
        avg_kernel = torch.ones((1, 1, avg_pool_size), device=log_probs.device)
        sum_result = F.conv1d(log_probs, avg_kernel, stride=avg_pool_size)
        # print(sum_result.shape)
        sum_result = F.conv1d(sum_result, kernel)
        confidences, messages = sum_result.softmax(0).max(0)
        # confidences, messages = sum_result.max(0)
        log_probs = log_probs.cpu()
        return messages, confidences

    def print_messages_and_confidences(self, log_probs, window_size):
        messages, confidences = self.get_messages_and_confidences(log_probs, window_size)
        for message, confidence in zip(messages, confidences):
            print(f'message: {message}, confidence: {confidence}')

    def filter_message(self, log_probs, window_size, smooth_window_size, threshold, order=5,
                       gap=20, avg_pool_size=10):
        gap = gap // avg_pool_size
        window_size = window_size // avg_pool_size
        # threshold = threshold / avg_pool_size
        filtered_messages = []
        filtered_confidences = []
        messages, confidences = self.get_messages_and_confidences(log_probs, window_size,
                                                                  avg_pool_size)
        messages = messages[0].cpu().numpy()
        confidences = confidences[0].cpu().numpy()
        if order > 0:
            zeros = np.zeros(order)
            smoothed_data = np.hstack((zeros, confidences, zeros))
        else:
            smoothed_data = confidences
        if smooth_window_size > 1:
            smoothed_data = uniform_filter1d(smoothed_data, size=smooth_window_size)
        if order == 0:
            local_maximas = np.arange(len(smoothed_data))
        else:
            local_maximas = argrelextrema(smoothed_data, np.greater, order=order)[0]
        local_maximas = local_maximas[
            (local_maximas < len(confidences) + order) & (local_maximas >= order)]
        local_maximas = local_maximas - order
        before_local = -100
        before_max = -100
        for i, local_maxima in enumerate(local_maximas):
            if confidences[local_maxima] > threshold:
                if local_maxima - before_local < gap and messages[local_maxima] == messages[
                    before_local]:
                    if confidences[local_maxima] > before_max:
                        before_max = confidences[local_maxima]
                    continue
                filtered_confidences.append(float(before_max))
                # print(f'confidence updated: {before_max}')
                before_local = local_maxima
                before_max = confidences[local_maxima]
                confidence = confidences[local_maxima]
                message = messages[local_maxima]
                # print(f'message: {message}, confidence: {confidence}')
                filtered_messages.append(int(message))
        # print(f'confidence updated: {before_max}')
        filtered_confidences.append(float(before_max))
        return filtered_messages, filtered_confidences[1:]


def insert_watermark(x_watermarked, x_human_written):
    # Split the text into a list of words
    words = x_human_written.split()

    # Generate a random index
    index = random.randint(0, len(words))

    # Insert the watermark at the random index
    words.insert(index, x_watermarked)

    # Join the list of words back into a single string
    x_human_written = ' '.join(words)

    return x_human_written


def insert_watermark_for_sentences(x_watermarked, x_human_written, seed=42):
    randomer = random.Random()
    randomer.seed(seed)

    x_human_written = randomer.sample(x_human_written, k=len(x_watermarked))
    print(len(x_watermarked[0]))
    x_cped = [insert_watermark(x_wm_single, x_human_single) for x_wm_single, x_human_single in
              zip(x_watermarked, x_human_written)]
    return x_cped

path ="./my_watermark_result/lm_new_7_10_my/huggyllama-llama-7b_1.5_1.5_300_200_100_42_42_20_10.0_4_1.0_10_-1_300_max_confidence_updated_0.5_huggyllama/llama-7b_1000.json"


with open(path, 'r') as f:
        results = json.load(f)

texts = results['text']
prefix_and_output_texts = results['prefix_and_output_text']#[:10]
output_texts = results['output_text']
# output_texts = results['sub-analysis-0.3-3.0-1.0-roberta-large']['ori_texts']
ppls = []

# random.seed(42)
# message_filter = MessageFilter(device="cuda:7")
# human_written_data_for_cp = load_from_disk('./c4-train.00102-of-00512_sliced_for_cp/')
# x_wms = results['output_text']
# # print(len(x_wms))
# x_humans = human_written_data_for_cp['train']['truncated_text']
# x_cpeds = insert_watermark_for_sentences(x_wms, x_humans, seed=42)
# for output_text, prefix_and_output_text in zip(x_cpeds, prefix_and_output_texts):
#   loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text,
#                                oracle_model_name='huggyllama/llama-13b',
#                                output_text=output_text,
#                                oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
#   ppls.append(ppl)

# if len(ppls) > 0:
#     mean_value = sum(ppls) / len(ppls)
# else:
#     mean_value = 0
# print(ppls)
# print(mean_value)


for output_text, prefix_and_output_text in zip(output_texts, prefix_and_output_texts):
  loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text,
                               oracle_model_name='huggyllama/llama-13b',
                               output_text=output_text[:2*len(output_text)//3],
                               oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
  ppls.append(ppl)

if len(ppls) > 0:
    mean_value = sum(ppls) / len(ppls)
else:
    mean_value = 0
print(ppls)
print(mean_value)

# ppls = []
# c4_sliced_and_filted = load_from_disk(
#         os.path.join('./c4-train.00000-of-00512_sliced'))
# c4_sliced_and_filted = c4_sliced_and_filted['train'].shuffle(seed=42).select(
#         range(200))
# for text in tqdm(c4_sliced_and_filted['text']):
#     tokenized_input = tokenizer(text, return_tensors='pt').to(model.device)
#     tokenized_input = truncate(tokenized_input, max_length=300)
#     output_tokens = model.generate(**tokenized_input,
#                                     temperature=1.0,
#                                     max_new_tokens=200,
#                                     num_beams=4,
#                                     logits_processor=LogitsProcessorList(
#                                 [min_length_processor, repetition_processor]))

#     output_text = \
#         tokenizer.batch_decode(
#             output_tokens[:, tokenized_input["input_ids"].shape[-1]:],
#             skip_special_tokens=True)[0]

#     prefix_and_output_text = tokenizer.batch_decode(output_tokens,
#                                                             skip_special_tokens=True)[0]
    
#     loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text,
#                                 oracle_model_name='huggyllama/llama-13b',
#                                 output_text=output_text,
#                                 oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
#     ppls.append(ppl)
# if len(ppls) > 0:
#     mean_value = sum(ppls) / len(ppls)
# else:
#     mean_value = 0
# print(ppls)
# print(mean_value)


# input_text = """"The merge listing the most important changes to Linux 3.8’s sound subsystem includes some other changes to audio drivers. The kernel now includes a driver for human interface devices (HIDs) that use I2C (1, 2 and others), using the "HID over I2C" protocol designed by Microsoft and implemented in WindowsÂ"""
# tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
# tokenized_input = truncate(tokenized_input, max_length=300)

# no_wm_output_tokens = model.generate(**tokenized_input, max_new_tokens=60, num_beams=4,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
#                                      logits_processor=LogitsProcessorList(
#                                        [min_length_processor, repetition_processor]))
# no_wm_output_text = tokenizer.decode(no_wm_output_tokens[0][tokenized_input['input_ids'].shape[-1]:],
#                                      skip_special_tokens=True)
# no_wm_prefix_and_output_text = tokenizer.decode(no_wm_output_tokens[0], skip_special_tokens=True)
# logging.warning(no_wm_output_text)
# loss, ppl = compute_ppl_single(prefix_and_output_text=no_wm_prefix_and_output_text,
#                                oracle_model_name='huggyllama/llama-13b',
#                                output_text=no_wm_output_text,
#                                oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
# print("无水印",loss, ppl)

# no_wm_output_text = "7 and Windows Vista. The drivers can read out data from HIDs and set the appropriate commands to them. An example of such a device is a BT-USB adapter. The sound subsystem now supports two new, high-quality audio codecs (1, 2):"
# no_wm_prefix_and_output_text = """"The merge listing the most important changes to Linux 3.8’s sound subsystem includes some other changes to audio drivers. The kernel now includes a driver for human interface devices (HIDs) that use I2C (1, 2 and others), using the "HID over I2C" protocol designed by Microsoft and implemented in WindowsÂ7 and Windows Vista. The drivers can read out data from HIDs and set the appropriate commands to them. An example of such a device is a BT-USB adapter. The sound subsystem now supports two new, high-quality audio codecs (1, 2):"""

# loss, ppl = compute_ppl_single(prefix_and_output_text=no_wm_prefix_and_output_text,
#                                oracle_model_name='huggyllama/llama-13b',
#                                output_text=no_wm_output_text,
#                                oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
# print("A:",loss, ppl)
