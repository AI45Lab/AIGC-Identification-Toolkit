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

import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
torch.set_printoptions(threshold=np.inf)
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b",device_map="auto")



c4_sliced_and_filted = load_from_disk('./c4-train.00000-of-00512_sliced')
c4_sliced_and_filted = c4_sliced_and_filted['train'].shuffle(seed=42).select(
    range(100))


sample_idx = 77
input_text = c4_sliced_and_filted[sample_idx]['text']


tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
tokenized_input = truncate(tokenized_input, max_length=300)

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
from watermarking.CredID.message_models.lm_message_model import LMMessageModel_update
from watermarking.CredID.message_model_processor import WmProcessorMessageModel_update


lm_message_model = LMMessageModel(tokenizer=tokenizer,lm_model=model,lm_tokenizer=tokenizer,
    delta = 1.5, lm_prefix_len=10, lm_topk=-1, message_code_len = 10,random_permutation_num=30)
wm_precessor_message_model = WmProcessorMessageModel(message_model=lm_message_model,tokenizer=tokenizer,
    encode_ratio=5,max_confidence_lbd=0.5,strategy='max_confidence_updated', message=[241])

start_length = tokenized_input['input_ids'].shape[-1]
wm_precessor_message_model.start_length = start_length
output_tokens = model.generate(**tokenized_input, max_new_tokens=60, num_beams=4,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
                               logits_processor=LogitsProcessorList(
                                   [min_length_processor, repetition_processor,
                                    wm_precessor_message_model]))

output_text = tokenizer.decode(output_tokens[0][start_length:],
                               skip_special_tokens=True)
logging.info(output_text)
prefix_and_output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
log_probs = wm_precessor_message_model.decode(output_text)
print("Balanced Modification:",log_probs[1][1][0][2]) 
print("Balanced Modification:",log_probs[0])




rm_lm_message_model = RandomMessageModel(tokenizer=tokenizer,
                                          lm_tokenizer=tokenizer,
                                          delta=1.5,
                                          message_code_len=10,
                                          device=model.device,
                                          )

rm_watermark_processor = WmProcessorRandomMessageModel(message_model=rm_lm_message_model,
                                                        tokenizer=tokenizer,
                                                        encode_ratio=5,
                                                        message=[102],
                                                        top_k=1000,
                                                        )


start_length = tokenized_input['input_ids'].shape[-1]
rm_watermark_processor.start_length = start_length
output_tokens_rm = model.generate(**tokenized_input, max_new_tokens=50, num_beams=4, temperature=1.0,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
                               logits_processor=LogitsProcessorList(
                                   [min_length_processor, repetition_processor,
                                    rm_watermark_processor]))

output_text_rm = tokenizer.decode(output_tokens_rm[0][start_length:],
                               skip_special_tokens=True)
logging.info(output_text_rm)
prefix_and_output_text_rm = tokenizer.decode(output_tokens_rm[0], skip_special_tokens=True)
log_probs_rm = rm_watermark_processor.decode(output_text_rm)
print("Direct Modification:",log_probs_rm[1][1][0][2]) 
print("Direct Modification:",log_probs_rm[0])




lm_message_model_update = LMMessageModel_update(tokenizer=tokenizer,lm_model=model,lm_tokenizer=tokenizer,
    delta = 1.5, lm_prefix_len=10, lm_topk=-1, message_code_len = 20,random_permutation_num=200)
wm_precessor_message_model_update = WmProcessorMessageModel_update(message_model=lm_message_model_update,tokenizer=tokenizer,
    encode_ratio=4,max_confidence_lbd=0.5,strategy='max_confidence_updated', message=[618])

start_length = tokenized_input['input_ids'].shape[-1]
wm_precessor_message_model_update.start_length = start_length
output_tokens_update = model.generate(**tokenized_input, max_new_tokens=90, num_beams=4,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
                               logits_processor=LogitsProcessorList(
                                   [min_length_processor, repetition_processor,
                                    wm_precessor_message_model_update]))

output_text_update = tokenizer.decode(output_tokens_update[0][start_length:],
                               skip_special_tokens=True)

prefix_and_output_text_update = tokenizer.decode(output_tokens_update[0], skip_special_tokens=True)
log_probs_update = wm_precessor_message_model_update.decode(output_text_update[:len(output_text_update)*2//3])
print(log_probs_update[1][1][0][2]) 
print(log_probs_update[0])



oracle_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b")
oracle_model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b",device_map="auto")#,torch_dtype=torch.float16)


from src.utils.watermark import compute_ppl_single

loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text_update,
                               oracle_model_name='huggyllama/llama-13b',
                               output_text=output_text_update,
                               oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
print("Mine:",loss, ppl)

pre = "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: How can we reduce air pollution?\n\n### Assistant:Here are some ways to reduce air pollution:\n\\begin{itemize}\n\\item Encouraging the use of public transportation, cycling, and walking instead of driving personal cars. This will reduce the number of cars on the road, which in turn will lower emissions from vehicles.\n\\item Planning cities and communities in a more sustainable way, with the aim of reducing the need for long commutes and promoting the use of green spaces and public transportation.\n\\item Encouraging the use of renewable energy sources like wind and solar power instead of fossil fuels. This will reduce the emissions from power plants and help the transition to a greener future.\n\\item Encouraging recycling and composting of waste to reduce the amount that ends up in landfills, where it releases methane, a potent greenhouse gas.\n\\item Encouraging the use of green building practices like energy-efficient windows, insulation, and solar panels to reduce the energy consumption of buildings.\n\\end{itemize}\n\nThese are some of the ways we can reduce air pollution. It is important to note that it takes time and effort"
out = "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: How can we reduce air pollution?\n\n### Assistant:"
loss, ppl = compute_ppl_single(prefix_and_output_text=pre,
                               oracle_model_name='huggyllama/llama-13b',
                               output_text=out,
                               oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
print("Test:",loss, ppl)


loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text,
                               oracle_model_name='huggyllama/llama-13b',
                               output_text=output_text,
                               oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
print("Balanced:",loss, ppl)

loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text_rm,
                               oracle_model_name='huggyllama/llama-13b',
                               output_text=output_text_rm,
                               oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
print("Random:",loss, ppl)




no_wm_output_tokens = model.generate(**tokenized_input, max_new_tokens=400, num_beams=4,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
                                     logits_processor=LogitsProcessorList(
                                       [min_length_processor, repetition_processor]))
no_wm_output_text = tokenizer.decode(no_wm_output_tokens[0][tokenized_input['input_ids'].shape[-1]:],
                                     skip_special_tokens=True)
no_wm_prefix_and_output_text = tokenizer.decode(no_wm_output_tokens[0], skip_special_tokens=True)
# print(no_wm_output_text )
# logging.warning(no_wm_output_text)
loss, ppl = compute_ppl_single(prefix_and_output_text=no_wm_prefix_and_output_text,
                               oracle_model_name='huggyllama/llama-13b',
                               output_text=no_wm_output_text,
                               oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
print("No Watermark:",loss, ppl)


output_text_update_no = tokenizer.decode(no_wm_output_tokens[0][start_length:],
                               skip_special_tokens=True)
output_text_update_no = "We are pleased to announce the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25), which will be held in Philadelphia, Pennsylvania at the Pennsylvania Convention Center from February 25 to March 4, 2025"
# logging.info(output_text_update)
for i in range(100):
    output_text_update_no = c4_sliced_and_filted[i]['text'][:60]

    log_probs_update = wm_precessor_message_model_update.decode(output_text_update_no)
    print("Type II Error:",log_probs_update[1][1][0][2]) 
    print("Type II Error:",log_probs_update[0])
