import json

from tqdm import tqdm

import watermarking.watermark_processor
from src.utils.load_local import load_local_model_or_tokenizer
from datasets import load_dataset, load_from_disk
from watermarking.CredID.message_models.lm_message_model import LMMessageModel
from watermarking.CredID.message_model_processor import WmProcessorMessageModel
from watermarking.arg_classes.wm_arg_class import WmLMArgs
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

from attacks.substition_attack import SubstitutionAttacker

MODEL_NAME = 'roberta-large'
REPLACEMENT_PROPORTION = 0.5
ATTEMPT_RATE = 3.0
LOGIT_THRESHOLD = 1.0

config = {
    'model_name': MODEL_NAME,
    'replacement_proportion': REPLACEMENT_PROPORTION,
    'attempt_rate': ATTEMPT_RATE,
    'logit_threshold': LOGIT_THRESHOLD,
}


def main(args: WmLMArgs):
    # results = args.load_result()
    path = './my_watermark_result/lm_new_7_10/huggyllama-llama-7b_5.5_1.5_300_170_20_42_42_10_16_4_1.0_10_-1_200_max_confidence_updated_0.5_huggyllama/llama-7b_1000.json'
    import json
    with open(path, 'r') as f:
        results = json.load(f)
    # tokenizer = load_local_model_or_tokenizer(args.model_name, 'tokenizer')
    # lm_tokenizer = load_local_model_or_tokenizer(args.lm_model_name, 'tokenizer')
    # lm_model = load_local_model_or_tokenizer(args.lm_model_name, 'model')
    # lm_model = lm_model.to(args.device)
    lm_model = AutoModelForCausalLM.from_pretrained(args.model_name ,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    lm_message_model = LMMessageModel(tokenizer=tokenizer, lm_model=lm_model,
                                      lm_tokenizer=tokenizer,
                                      delta=5.5, lm_prefix_len=10, lm_topk=-1, message_code_len=10,
                                      random_permutation_num=200)
    wm_precessor_message_model = WmProcessorMessageModel(message_model=lm_message_model,
                                                         tokenizer=tokenizer,
                                                         encode_ratio=16, max_confidence_lbd=0.5,
                                                         strategy='max_confidence_updated',
                                                         message=[42])

    random.seed(args.sample_seed)

    substitution_attacker = SubstitutionAttacker(model_name=MODEL_NAME,
                                                 replacement_proportion=REPLACEMENT_PROPORTION,
                                                 attempt_rate=ATTEMPT_RATE,
                                                 logit_threshold=LOGIT_THRESHOLD,
                                                 device=args.device)

    x_wms = results['output_text']

    decoded_results = []
    ori_texts = []
    substituted_texts = []
    substituted_text_other_informations = []
    corrected = []
    for i in tqdm(range(20)):
        substituted_text, other_information = substitution_attacker.attack(x_wms[i])
        print(x_wms[i])
        print(substituted_text)
        ori_texts.append(x_wms[i])
        substituted_texts.append(substituted_text)
        substituted_text_other_informations.append(other_information)
        y = wm_precessor_message_model.decode(substituted_text, disable_tqdm=True)
        corrected.append(y[0] == wm_precessor_message_model.message.tolist()[:len(y)])
        decoded_results.append([y[0], y[1][1]])
    successful_rate = sum(corrected) / len(corrected)

    if f'sub-analysis-{REPLACEMENT_PROPORTION}-{ATTEMPT_RATE}-{LOGIT_THRESHOLD}-{MODEL_NAME}' not in results:
        analysis_results = {}
        analysis_results['successful_rate'] = successful_rate
        analysis_results['decoded_results'] = decoded_results
        analysis_results['config'] = config
        analysis_results['substituted_texts'] = substituted_texts
        analysis_results['ori_texts'] = ori_texts
        analysis_results[
            'substituted_text_other_informations'] = substituted_text_other_informations
        results[
            f'sub-analysis-{REPLACEMENT_PROPORTION}-{ATTEMPT_RATE}-{LOGIT_THRESHOLD}-{MODEL_NAME}'] = analysis_results

    print(analysis_results)
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
