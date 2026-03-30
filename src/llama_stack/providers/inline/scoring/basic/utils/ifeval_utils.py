# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.inline.scoring.basic.utils.ifeval_checkers_core import (
    BulletListChecker,
    ConstrainedResponseChecker,
    HighlightSectionChecker,
    KeywordChecker,
    KeywordFrequencyChecker,
    NumberOfSentences,
    NumberOfWords,
    ParagraphChecker,
    PlaceholderChecker,
    PostscriptChecker,
    ResponseLanguageChecker,
    SectionChecker,
)
from llama_stack.providers.inline.scoring.basic.utils.ifeval_checkers_format import (
    CapitalLettersEnglishChecker,
    CapitalWordFrequencyChecker,
    CommaChecker,
    EndChecker,
    ForbiddenWords,
    JsonFormat,
    LetterFrequencyChecker,
    LowercaseLettersEnglishChecker,
    ParagraphFirstWordCheck,
    QuotationChecker,
    RepeatPromptThenAnswer,
    TitleChecker,
    TwoResponsesChecker,
)

_KEYWORD = "keywords:"
_LANGUAGE = "language:"
_LENGTH = "length_constraints:"
_CONTENT = "detectable_content:"
_FORMAT = "detectable_format:"
_MULTITURN = "multi-turn:"
_COMBINATION = "combination:"
_STARTEND = "startend:"
_CHANGE_CASES = "change_case:"
_PUNCTUATION = "punctuation:"

INSTRUCTION_DICT = {
    _KEYWORD + "existence": KeywordChecker,
    _KEYWORD + "frequency": KeywordFrequencyChecker,
    # _KEYWORD + "key_sentences": KeySentenceChecker,
    _KEYWORD + "forbidden_words": ForbiddenWords,
    _KEYWORD + "letter_frequency": LetterFrequencyChecker,
    _LANGUAGE + "response_language": ResponseLanguageChecker,
    _LENGTH + "number_sentences": NumberOfSentences,
    _LENGTH + "number_paragraphs": ParagraphChecker,
    _LENGTH + "number_words": NumberOfWords,
    _LENGTH + "nth_paragraph_first_word": ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": PlaceholderChecker,
    _CONTENT + "postscript": PostscriptChecker,
    _FORMAT + "number_bullet_lists": BulletListChecker,
    # _CONTENT + "rephrase_paragraph": RephraseParagraph,
    _FORMAT + "constrained_response": ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": (HighlightSectionChecker),
    _FORMAT + "multiple_sections": SectionChecker,
    # _FORMAT + "rephrase": RephraseChecker,
    _FORMAT + "json_format": JsonFormat,
    _FORMAT + "title": TitleChecker,
    # _MULTITURN + "constrained_start": ConstrainedStartChecker,
    _COMBINATION + "two_responses": TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": RepeatPromptThenAnswer,
    _STARTEND + "end_checker": EndChecker,
    _CHANGE_CASES + "capital_word_frequency": CapitalWordFrequencyChecker,
    _CHANGE_CASES + "english_capital": CapitalLettersEnglishChecker,
    _CHANGE_CASES + "english_lowercase": LowercaseLettersEnglishChecker,
    _PUNCTUATION + "no_comma": CommaChecker,
    _STARTEND + "quotation": QuotationChecker,
}

INSTRUCTION_LIST = list(INSTRUCTION_DICT.keys()) + [
    _KEYWORD[:-1],
    _LANGUAGE[:-1],
    _LENGTH[:-1],
    _CONTENT[:-1],
    _FORMAT[:-1],
    _MULTITURN[:-1],
    _COMBINATION[:-1],
    _STARTEND[:-1],
    _CHANGE_CASES[:-1],
    _PUNCTUATION[:-1],
]
