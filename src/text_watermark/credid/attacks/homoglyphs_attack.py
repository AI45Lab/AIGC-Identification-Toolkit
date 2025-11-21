""" Text-based normalizers, used to mitigate simple attacks against watermarking.

This implementation is unlikely to be a complete list of all possible exploits within the unicode standard,
it represents our best effort at the time of writing.

These normalizers can be used as stand-alone normalizers. They could be made to conform to HF tokenizers standard, but that would
require messing with the limited rust interface of tokenizers.NormalizedString
"""
from collections import defaultdict
from functools import cache

import re
import unicodedata
import homoglyphs as hg
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from homoglyphs_test import generate_attack_variant
from homoglyphs import Homoglyphs


from src.utils.watermark import compute_ppl_single

class HomoglyphCanonizer:
    """Attempts to detect homoglyph attacks and find a consistent canon.

    This function does so on a per-ISO-category level. Language-level would also be possible (see commented code).
    """

    def __init__(self):
        self.homoglyphs = None

    def __call__(self, homoglyphed_str: str) -> str:
        # find canon:
        target_category, all_categories = self._categorize_text(homoglyphed_str)
        homoglyph_table = self._select_canon_category_and_load(target_category, all_categories)
        return self._sanitize_text(target_category, homoglyph_table, homoglyphed_str)

    def _categorize_text(self, text: str) -> dict:
        iso_categories = defaultdict(int)
        # self.iso_languages = defaultdict(int)

        for char in text:
            iso_categories[hg.Categories.detect(char)] += 1
            # for lang in hg.Languages.detect(char):
            #     self.iso_languages[lang] += 1
        target_category = max(iso_categories, key=iso_categories.get)
        all_categories = tuple(iso_categories)
        return target_category, all_categories

    @cache
    def _select_canon_category_and_load(
        self, target_category: str, all_categories: tuple[str]
    ) -> dict:
        homoglyph_table = hg.Homoglyphs(
            categories=(target_category, "COMMON")
        )  # alphabet loaded here from file

        source_alphabet = hg.Categories.get_alphabet(all_categories)
        restricted_table = homoglyph_table.get_restricted_table(
            source_alphabet, homoglyph_table.alphabet
        )  # table loaded here from file
        return restricted_table

    def _sanitize_text(
        self, target_category: str, homoglyph_table: dict, homoglyphed_str: str
    ) -> str:
        sanitized_text = ""
        for char in homoglyphed_str:
            # langs = hg.Languages.detect(char)
            cat = hg.Categories.detect(char)
            if target_category in cat or "COMMON" in cat or len(cat) == 0:
                sanitized_text += char
            else:
                sanitized_text += list(homoglyph_table[char])[0]
        return sanitized_text
    
def generate_homoglyph_attack(original_text):
    """
    Generate a homoglyph attack string based on the original text.

    :param original_text: The original text to be transformed.
    :type original_text: str
    :return: A string containing homoglyphs.
    :rtype: str
    """
    # Initialize the Homoglyphs class
    homoglyphs = Homoglyphs(categories=("LATIN", "COMMON"))

    # Generate homoglyph combinations for the original text
    combinations = homoglyphs.get_combinations(original_text)

    # Choose one of the combinations as the attack string
    # Here, we simply choose the first combination for demonstration
    if combinations:
        attack_text = combinations[0]
    else:
        attack_text = original_text  # Fallback to original text if no combinations found

    return attack_text
    

if __name__ == "__main__":
    # Homoglyph = HomoglyphCanonizer()
    path ="./my_watermark_result/lm_new_7_10/huggyllama-llama-7b_1.5_1.5_300_110_100_42_42_10_10_8_1.0_10_-1_200_max_confidence_updated_0.5_huggyllama/llama-7b_1000.json"
    oracle_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b")
    oracle_model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b",device_map="auto")#,torch_dtype=torch.float16)   

    with open(path, 'r') as f:
            results = json.load(f)

    texts = results['text']
    prefix_and_output_texts = results['prefix_and_output_text']#[:10]
    output_texts = results['output_text']
    # output_texts = results['sub-analysis-0.3-3.0-1.0-roberta-large']['ori_texts']
    ppls = []
    for output_text, prefix_and_output_text in zip(output_texts, prefix_and_output_texts):
        print(output_text)
        att_text = generate_attack_variant(output_text, 0.1)
        print(att_text)
        loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text,
                                    oracle_model_name='huggyllama/llama-13b',
                                    output_text=att_text,
                                    oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
        ppls.append(ppl)

    if len(ppls) > 0:
        mean_value = sum(ppls) / len(ppls)
    else:
        mean_value = 0
    print(ppls)
    print(mean_value)



