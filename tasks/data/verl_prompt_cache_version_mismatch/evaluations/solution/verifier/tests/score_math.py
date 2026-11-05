# Apache-2.0. Answer normalization adapted from pinned veRL math_dapo.py.
import re
SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), ('\\ ', ''), (' ', ''), ('mbox', 'text'), (',\\text{and}', ','), ('\\text{and}', ','), ('\\text{m}', '\\text{}')]
REMOVED_EXPRESSIONS = ['square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples', '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}', '\\text{}', '\\mathrm{th}', '^\\circ', '^{\\circ}', '\\;', ',\\!', '{,}', '"', '\\dots']
ANSWER_PATTERN = re.compile('Answer\\s*:\\s*([^\\n]+)', re.IGNORECASE)

def normalize_final_answer(answer):
    answer = str(answer).split('=')[-1]
    for before, after in SUBSTITUTIONS:
        answer = answer.replace(before, after)
    for expression in REMOVED_EXPRESSIONS:
        answer = answer.replace(expression, '')
    answer = re.sub('(.*?)(\\$)(.*?)(\\$)(.*)', '$\\3$', answer)
    answer = re.sub('(\\\\text\\{)(.*?)(\\})', '\\2', answer)
    answer = re.sub('(\\\\textbf\\{)(.*?)(\\})', '\\2', answer)
    answer = re.sub('(\\\\overline\\{)(.*?)(\\})', '\\2', answer)
    answer = re.sub('(\\\\boxed\\{)(.*)(\\})', '\\2', answer)
    answer = re.sub('(frac)([^{])(.)', 'frac{\\2}{\\3}', answer)
    answer = re.sub('(sqrt)([^{])', 'sqrt{\\2}', answer)
    answer = answer.replace('$', '')
    if answer.replace(',', '').isdigit():
        answer = answer.replace(',', '')
    return answer.strip()

def normalized_prediction(text):
    matches = ANSWER_PATTERN.findall(str(text)[-300:])
    return normalize_final_answer(matches[-1] if matches else '[INVALID]')

def score(text, ground_truth):
    truth = str(ground_truth).strip()
    if not re.fullmatch('[+-]?[0-9]+', truth):
        raise ValueError('DAPO ground_truth must be a plain integer')
    return normalized_prediction(text) == normalize_final_answer(truth)
