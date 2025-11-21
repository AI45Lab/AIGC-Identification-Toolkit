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
import json
from tqdm import tqdm

import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
torch.set_printoptions(threshold=np.inf)
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "huggyllama/llama-7b"
path ="./my_watermark_result/lm_new_7_10/tiiuae-falcon-7b_1.5_1.5_300_50_250_42_42_10_4_4_1.0_10_-1_200_max_confidence_updated_0.5_huggyllama/llama-7b_1000.json"
with open(path, 'r') as f:
        results = json.load(f)

texts = results['text']
prefix_and_output_texts = results['prefix_and_output_text']
output_texts = results['output_text']

analysis_results = results.get('analysis', {})

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")#,torch_dtype=torch.float16)



min_length_processor = MinLengthLogitsProcessor(min_length=10000,
                                                eos_token_id=tokenizer.eos_token_id)
repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=1.5)

from importlib import reload
import watermarking.CredID.message_models.lm_message_model
watermarking.CredID.message_models.lm_message_model = reload(watermarking.CredID.message_models.lm_message_model)
import watermarking.CredID.message_model_processor
watermarking.CredID.message_model_processor = reload(watermarking.CredID.message_model_processor)
from watermarking.CredID.message_models.lm_message_model import LMMessageModel
from watermarking.CredID.message_model_processor import WmProcessorMessageModel



confidences = []
decoded_messages = []
accs = []
decoded_message_confidences = []


lm_message_model_update = LMMessageModel(tokenizer=tokenizer,lm_model=model,lm_tokenizer=tokenizer,
    delta = 1.5, lm_prefix_len=10, lm_topk=-1, message_code_len = 10,random_permutation_num=200,shifts=[21, 24, 3, 8, 14, 2, 4, 28, 31, 3, 8, 14, 2, 4, 28])#llama: [21, 24, 3, 8, 14, 2, 4, 28, 31, 3, 8, 14, 2, 4, 28]  falcon[4, 2, 21, 28, 14, 4, 8, 8, 3, 14, 24, 31, 3, 28, 2]  gemma[3, 8, 24, 14, 2, 4, 8, 14, 2, 28, 3, 31, 4, 28, 21]
wm_precessor_message_model_update = WmProcessorMessageModel(message_model=lm_message_model_update,tokenizer=tokenizer,
    encode_ratio=10,max_confidence_lbd=0.5,strategy='max_confidence_updated', message=[42])


for output_text in tqdm(output_texts):
        decoded_message, other_information = wm_precessor_message_model_update.decode(output_text[:len(output_text)*2//3],
                                                                    disable_tqdm=True)
        confidence = other_information[1][0][2]
        available_message_num = 110 // (
                    int(10 * 10))
        acc = decoded_message[:available_message_num] == [42]

        confidences.append(confidence)

        decoded_messages.append(decoded_message)
        decoded_message_confidences.append({'decoded_message': decoded_message, 'confidence': confidence})
        accs.append(acc)
analysis_results['message_confidence'] = decoded_message_confidences
# analysis_results['confidence'] = confidences
# analysis_results['decoded_message'] = decoded_messages
# analysis_results['acc'] = accs
results['analysis'] = analysis_results
with open(path, 'w') as f:
        json.dump(results, f, indent=4)



with open(path, 'r') as f:
        data = json.load(f)
        
confidence1 = [item[0][2] for item in data['confidence']]
con = [item['confidence'] for item in data['analysis_att_ori']['message_confidence']]
confidence2 =  [item['confidence'] for item in data['analysis_att_gemma']['message_confidence']]
confidence3 =  [item['confidence'] for item in data['analysis_att_llama']['message_confidence']]
# message =   [item[0] for item in data['decoded_message']]
message = [item['decoded_message'] for item in data['analysis_att_ori']['message_confidence']]
count = 0
for i in range(len(message)):
    if (message[i] != [42]) :#& (con[i] > confidence2[i]) & (confidence1[i]>confidence3[i]):
        print(f"  Decoded message: {message[i]}")
        print(f"  ori confidence: {con[i]}")
        print(f"  gemma confidence: {confidence2[i]}")
        print(f"  llama confidence: {confidence3[i]}")
        count += 1
print(count)