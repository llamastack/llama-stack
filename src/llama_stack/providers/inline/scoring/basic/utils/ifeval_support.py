# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import collections
import functools
import random
import re
from collections.abc import Iterable, Sequence
from types import MappingProxyType

import emoji
import langdetect
import nltk

from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="scoring")

from llama_stack.providers.inline.scoring.basic.utils.ifeval_word_list import WORD_LIST

# ISO 639-1 codes to language names.
LANGUAGE_CODES = MappingProxyType(
    {
        "en": "English",
        "es": "Spanish",
        "pt": "Portuguese",
        "ar": "Arabic",
        "hi": "Hindi",
        "fr": "French",
        "ru": "Russian",
        "de": "German",
        "ja": "Japanese",
        "it": "Italian",
        "bn": "Bengali",
        "uk": "Ukrainian",
        "th": "Thai",
        "ur": "Urdu",
        "ta": "Tamil",
        "te": "Telugu",
        "bg": "Bulgarian",
        "ko": "Korean",
        "pl": "Polish",
        "he": "Hebrew",
        "fa": "Persian",
        "vi": "Vietnamese",
        "ne": "Nepali",
        "sw": "Swahili",
        "kn": "Kannada",
        "mr": "Marathi",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "ml": "Malayalam",
        "fi": "Finnish",
    }
)

# Chinese characters
_CHINESE_CHARS_PATTERN = r"[\u4E00-\u9FFF\u3400-\u4DBF]"
# Japanese Hiragana & Katakana
_JAPANESE_CHARS_PATTERN = r"[\u3040-\u309f\u30a0-\u30ff]"
# Korean (Hangul Syllables)
_KOREAN_CHARS_PATTERN = r"[\uAC00-\uD7AF]"
_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
)
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


# Util functions
def split_into_sentences(text):
    """Split the text into sentences.

    Args:
      text: A string that consists of more than or equal to one sentences.

    Returns:
      A list of strings where each string is a sentence.
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(
        _MULTIPLE_DOTS,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text,
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_words(text):
    """Counts the number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    num_words = len(tokens)
    return num_words


def split_chinese_japanese_hindi(lines: str) -> Iterable[str]:
    """
    Split Chinese and Japanese text into sentences.
    From https://stackoverflow.com/questions/27441191/splitting-chinese-document-into-sentences
    Special question/exclamation marks were added upon inspection of our raw data,
    Also supports multiple lines.
    The separator for hindi is '।'
    """
    for line in lines.splitlines():
        yield from re.findall(
            r"[^!?。\.\!\?\！\？\．\n।]+[!?。\.\!\?\！\？\．\n।]?",
            line.strip(),
            flags=re.U,
        )


def count_words_cjk(text: str) -> int:
    """Counts the number of words for Chinese and Japanese and Korean.
    Can be extended to additional languages.
    Source: https://stackoverflow.com/questions/49164507/how-to-count-the-number-of-chinese-korean-and-english-words withadditional modifications
    Example:
        >In: count_words_cjk('こんにちは、ジェイソンさん、Jason? Nice to meet you☺ ❤')
        >Out: 19
    """
    # Non alpha numeric patterns in latin and asian languages.
    non_alphanumeric_patterns = (
        r"[\\.\!\?\．\/_,\{\}<>:;$%^&*(+\"\'+——！，。？、`~@#￥……（）：；《）《》“”()\[\]«»〔〕\-「」]+"
    )
    text = re.sub(non_alphanumeric_patterns, "", text)

    emoji_cnt = emoji.emoji_count(text)  # count emojis
    text = emoji.replace_emoji(text, "")  # remove emojis

    foreign_chars_patterns = "|".join([_CHINESE_CHARS_PATTERN, _JAPANESE_CHARS_PATTERN, _KOREAN_CHARS_PATTERN])
    asian_chars = re.findall(foreign_chars_patterns, text)
    asian_chars_cnt = len(asian_chars)
    non_asian_chars = re.sub(foreign_chars_patterns, " ", text)
    non_asian_words_cnt = len(non_asian_chars.split())

    return non_asian_words_cnt + asian_chars_cnt + emoji_cnt


@functools.cache
def _get_sentence_tokenizer():
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


def count_sentences(text):
    """Count the number of sentences."""
    tokenizer = _get_sentence_tokenizer()
    tokenized_sentences = tokenizer.tokenize(text)
    return len(tokenized_sentences)


def get_langid(text: str, lid_path: str | None = None) -> str:
    """Detect the primary language of a text using per-line language detection.

    Args:
        text: input text to analyze
        lid_path: unused, kept for interface compatibility

    Returns:
        ISO 639-1 language code, defaulting to "en" if detection fails
    """
    line_langs: list[str] = []
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) >= 4]

    for line in lines:
        try:
            line_langs.append(langdetect.detect(line))
        except langdetect.LangDetectException as e:
            logger.info("Unable to detect language for text %s due to %s", line, e)  # refex: disable=pytotw.037

    if len(line_langs) == 0:
        return "en"
    # select the text language to be the most commonly predicted language of the lines.
    return collections.Counter(line_langs).most_common(1)[0][0]


def generate_keywords(num_keywords):
    """Randomly generates a few keywords."""
    return random.sample(WORD_LIST, k=num_keywords)


"""Library of instructions"""
_InstructionArgsDtype = dict[str, int | str | Sequence[str]] | None

_LANGUAGES = LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("less than", "at least")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)

# The options of starter keywords.
_STARTER_OPTIONS = (
    "I would say",
    "My answer is",
    "I believe",
    "In my opinion",
    "I think",
    "I reckon",
    "I feel",
    "From my perspective",
    "As I see it",
    "According to me",
    "As far as I'm concerned",
    "To my understanding",
    "In my view",
    "My take on it is",
    "As per my perception",
)

# The options of ending keywords.
# TODO(jeffreyzhou) add more ending options
_ENDING_OPTIONS = ("Any other questions?", "Is there anything else I can help with?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
_SECTION_SPLITER = ("Section", "SECTION")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500
