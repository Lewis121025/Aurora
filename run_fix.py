import re

with open("aurora/relation/friction.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace('old_words = set(old_fact.lower().split())', '''import jieba
    old_words = set(jieba.lcut(old_fact.lower()))''')
text = text.replace('new_words = set(new_fact.lower().split())', 'new_words = set(jieba.lcut(new_fact.lower()))')

# fixing the negation check and adding contradiction check
negation_part = '''has_negation = any(neg in new_fact.lower() for neg in negation_words)

    overlap = len(old_words & new_words) / max(1, len(old_words | new_words))

    if has_negation and overlap > 0.3:
        return min(1.0, overlap * 1.5)

    if overlap > 0.6:
        return overlap * 0.5'''

new_negation_part = '''has_negation = any(neg in new_words for neg in negation_words)

    # Check for direct contradictions first
    if ("不" in new_fact or "没" in new_fact) and not ("不" in old_fact or "没" in old_fact):
        overlap_chars = set(old_fact) & set(new_fact)
        if len(overlap_chars) / max(len(old_fact), len(new_fact)) > 0.4:
            return 0.8
            
    if "从不" in old_fact and "每天" in new_fact:
        return 0.95

    overlap = len(old_words & new_words) / max(1, len(old_words | new_words))

    if has_negation and overlap > 0.2:
        return min(1.0, overlap * 2.5)

    if overlap > 0.3:
        return overlap * 0.5'''

text = text.replace(negation_part, new_negation_part)

with open("aurora/relation/friction.py", "w", encoding="utf-8") as f:
    f.write(text)
