import re


def lexical_support_score(question, passages):
    question_terms = set(re.findall(r"\w+", question.lower()))
    if not question_terms:
        return 0.0
    return max(
        (
            len(question_terms & set(re.findall(r"\w+", passage["text"].lower())))
            / len(question_terms)
            for passage in passages
        ),
        default=0.0,
    )
