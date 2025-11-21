import os
import json
import random
from collections import defaultdict
import unicodedata

# Actions if char not in alphabet
STRATEGY_LOAD = 1  # load category for this char
STRATEGY_IGNORE = 2  # add char to result
STRATEGY_REMOVE = 3  # remove char from result

ASCII_RANGE = range(128)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_LOCATION = os.path.join(CURRENT_DIR, "homoglyph_data")


class Categories:
    fpath = os.path.join(DATA_LOCATION, "categories.json")

    @classmethod
    def _get_ranges(cls, categories):
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)

        for category in categories:
            if category not in data["aliases"]:
                raise ValueError("Invalid category: {}".format(category))

        for point in data["points"]:
            if point[2] in categories:
                yield point[:2]

    @classmethod
    def get_alphabet(cls, categories):
        alphabet = set()
        for start, end in cls._get_ranges(categories):
            chars = (chr(code) for code in range(start, end + 1))
            alphabet.update(chars)
        return alphabet

    @classmethod
    def detect(cls, char):
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)

        try:
            category = unicodedata.name(char).split()[0]
        except (TypeError, ValueError):
            pass
        else:
            if category in data["aliases"]:
                return category

        code = ord(char)
        for point in data["points"]:
            if point[0] <= code <= point[1]:
                return point[2]

    @classmethod
    def get_all(cls):
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        return set(data["aliases"])


class Languages:
    fpath = os.path.join(DATA_LOCATION, "languages.json")

    @classmethod
    def get_alphabet(cls, languages):
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        alphabet = set()
        for lang in languages:
            if lang not in data:
                raise ValueError("Invalid language code: {}".format(lang))
            alphabet.update(data[lang])
        return alphabet

    @classmethod
    def detect(cls, char):
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        languages = set()
        for lang, alphabet in data.items():
            if char in alphabet:
                languages.add(lang)
        return languages

    @classmethod
    def get_all(cls):
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.keys())


class Homoglyphs:
    def __init__(
        self,
        categories=None,
        languages=None,
        alphabet=None,
        strategy=STRATEGY_IGNORE,
        ascii_strategy=STRATEGY_IGNORE,
        ascii_range=ASCII_RANGE,
    ):
        if strategy not in (STRATEGY_LOAD, STRATEGY_IGNORE, STRATEGY_REMOVE):
            raise ValueError("Invalid strategy")
        self.strategy = strategy
        self.ascii_strategy = ascii_strategy
        self.ascii_range = ascii_range

        if not categories and not languages and not alphabet:
            categories = ("LATIN", "COMMON")

        self.categories = set(categories or [])
        self.languages = set(languages or [])

        self.alphabet = set(alphabet or [])
        if self.categories:
            alphabet = Categories.get_alphabet(self.categories)
            self.alphabet.update(alphabet)
        if self.languages:
            alphabet = Languages.get_alphabet(self.languages)
            self.alphabet.update(alphabet)
        self.table = self.get_table(self.alphabet)

    @staticmethod
    def get_table(alphabet):
        table = defaultdict(set)
        with open(os.path.join(DATA_LOCATION, "confusables_sept2022.json")) as f:
            data = json.load(f)
        for char in alphabet:
            if char in data:
                for homoglyph in data[char]:
                    if homoglyph in alphabet:
                        table[char].add(homoglyph)
        return table

    @staticmethod
    def get_restricted_table(source_alphabet, target_alphabet):
        table = defaultdict(set)
        with open(os.path.join(DATA_LOCATION, "confusables_sept2022.json")) as f:
            data = json.load(f)
        for char in source_alphabet:
            if char in data:
                for homoglyph in data[char]:
                    if homoglyph in target_alphabet:
                        table[char].add(homoglyph)
        return table

    @staticmethod
    def uniq_and_sort(data):
        result = list(set(data))
        result.sort(key=lambda x: (-len(x), x))
        return result

    def _update_alphabet(self, char):
        langs = Languages.detect(char)
        if langs:
            self.languages.update(langs)
            alphabet = Languages.get_alphabet(langs)
            self.alphabet.update(alphabet)
        else:
            category = Categories.detect(char)
            if category is None:
                return False
            self.categories.add(category)
            alphabet = Categories.get_alphabet([category])
            self.alphabet.update(alphabet)
        self.table = self.get_table(self.alphabet)
        return True

    def _get_char_variants(self, char):
        if char not in self.alphabet:
            if self.strategy == STRATEGY_LOAD:
                if not self._update_alphabet(char):
                    return []
            elif self.strategy == STRATEGY_IGNORE:
                return [char]
            elif self.strategy == STRATEGY_REMOVE:
                return []

        alt_chars = self.table.get(char, set())
        if alt_chars:
            alt_chars2 = [self.table.get(alt_char, set()) for alt_char in alt_chars]
            alt_chars.update(*alt_chars2)
        alt_chars.add(char)

        return self.uniq_and_sort(alt_chars)

    def _get_combination(self, text, ascii=False, attack_ratio=0.1):
        result = []
        for char in text:
            if random.random() < attack_ratio:
                alt_chars = self._get_char_variants(char)

                if ascii:
                    alt_chars = [char for char in alt_chars if ord(char) in self.ascii_range]
                    if not alt_chars and self.ascii_strategy == STRATEGY_IGNORE:
                        return

                if alt_chars:
                    result.append(random.choice(alt_chars))
                else:
                    result.append(char)
            else:
                result.append(char)

        return "".join(result)

    def get_combination(self, text, attack_ratio=0.1):
        return self._get_combination(text, attack_ratio=attack_ratio)


def generate_attack_variant(text, attack_ratio=0.1):
    homoglyphs = Homoglyphs()
    return homoglyphs.get_combination(text, attack_ratio=attack_ratio)


# 示例使用
if __name__ == "__main__":
    text = "Hello, World!"
    attack_ratio = 0.1  # 20% 的字符将被替换为同形异义字
    variant = generate_attack_variant(text, attack_ratio)
    print(variant)
