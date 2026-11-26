"""Reproducible local IFEval grading; preserve explicit single-character symbols."""
import dataclasses
import hashlib
import random
from instruction_following_eval import evaluation_lib as IF, instructions
import langdetect

PROTOCOL = {'name': 'symbol_corrected', 'seed': 42,
            'seed_scope': 'sha256(42:question_id), reset per sample and strict/loose call',
            'local_correction': 'Preserve explicitly requested single-character symbols in LetterFrequencyChecker'}
_original = instructions.LetterFrequencyChecker.build_description

def _corrected(self, *, letter=None, let_frequency=None, let_relation=None):
    symbol = isinstance(letter, str) and len(letter) == 1 and not ('a' <= letter.lower() <= 'z')
    description = _original(self, letter='a' if symbol else letter,
                            let_frequency=let_frequency, let_relation=let_relation)
    if symbol:
        self._letter = letter
        description = self._description_pattern.format(letter=self._letter,
            let_relation=self._comparison_relation, let_frequency=self._frequency)
    return description

def grade(ref, row):
    instructions.LetterFrequencyChecker.build_description = _corrected
    langdetect.DetectorFactory.seed = 42
    inp = IF.InputExample(**{k: ref[k] for k in ('key', 'instruction_id_list', 'prompt', 'kwargs')})
    seed = int.from_bytes(hashlib.sha256(f'42:{row["id"]}'.encode()).digest()[:8], 'big')
    record = {'id': row['id'], 'sample': row['sample']}
    for mode, fn in [('strict', IF.test_instruction_following_strict), ('loose', IF.test_instruction_following_loose)]:
        random.seed(seed)
        record[mode] = dataclasses.asdict(fn(inp, {inp.prompt: row['response'].rsplit('</think>', 1)[-1].strip()}))
    return record
