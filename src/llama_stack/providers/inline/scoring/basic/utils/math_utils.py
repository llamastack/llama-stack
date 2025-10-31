# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
from collections.abc import Sequence

from llama_stack.providers.utils.scoring.basic_scoring_utils import time_limit

# from minerva
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def try_evaluate_frac(expression: str, fmt: str = "0.2e") -> str:
    if isinstance(expression, float):
        return expression
    new_expression = f"{expression}"
    regex = re.compile(r"\\frac{([^}]+)}{([^}]+)}")
    for match in re.finditer(regex, expression):
        try:
            value = float(match.group(1)) / float(match.group(2))
            new_expression = new_expression.replace(
                match.group(),
                f"{{value:{fmt}}}".format(value=value),
                1,
            )
        except Exception:
            continue
    return new_expression


def try_evaluate_latex(expression: str, fmt: str = ".2e") -> str:
    try:
        with time_limit(seconds=5):
            from sympy.parsing.latex import parse_latex

            value = parse_latex(expression).evalf()  # type: ignore
            return f"{{value:{fmt}}}".format(value=value)
    except Exception:
        return expression


def first_answer(text: str, markers: Sequence[str] = ("Q:", "A:")) -> str:
    for marker in markers:
        text = text.split(marker)[0]
    return text


def extract_result_from_boxed(answer: str) -> str:
    box_start = "\\boxed"
    # format is `\\boxed <value>$` or `\\boxed{<value>}`, with potential white spaces framing `<value>`
    start = answer.rfind(box_start)
    if start < 0:
        return ""
    answer = answer[start + len(box_start) :].strip()
    ends_with_curly = answer.startswith("{")
    i = 0
    open_braces = 0
    while i < len(answer):
        if answer[i] == "{":
            open_braces += 1
        elif answer[i] == "}":
            open_braces -= 1
        if open_braces == 0:
            if ends_with_curly:
                answer = answer[: i + 1].strip()
                break
            elif answer[i] == "$":
                answer = answer[:i].strip()
                break
        i += 1
    else:
        return ""
    # remove extra curly braces
    while True:
        if answer.startswith("{") and answer.endswith("}"):
            answer = answer[1:-1].strip()
        else:
            break
    return answer


# from minerva paper + _normalise_result from xavierm
def normalize_final_answer(final_answer: str, regex_pattern: str, match_first: bool = True) -> str:
    """Extract and normalize a final answer to a quantitative reasoning question."""
    match = re.findall(regex_pattern, final_answer)
    extraction: str
    if len(match) > 0:
        if match_first:
            extraction = match[0]
        else:
            extraction = match[-1]
    else:
        extraction = extract_result_from_boxed(final_answer)

    if len(extraction) == 0:
        return final_answer
    else:
        final_answer = extraction
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    # If the final answer is a single letter in parentheses, remove the parentheses
    # Example: (a) -> a (but not (ab) -> ab)
    if re.match(r"\([a-zA-Z]\)", final_answer):
        final_answer = final_answer[1]
    return _normalise_result(final_answer)


def _normalise_result(string: str) -> str:
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("cfrac", "frac")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\le", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    string = string.split("=")[-1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def _remove_right_units(string: str) -> str:
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    try:
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string
    except AssertionError:
        return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) == 0:
            return string
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0:
                return string
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        ia = int(a)
        ib = int(b)
        assert string == f"{ia}/{ib}"
        new_string = "\\frac{" + str(ia) + "}{" + str(ib) + "}"
        return new_string
    except (ValueError, AssertionError):
        return string
