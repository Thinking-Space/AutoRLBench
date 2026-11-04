# Closed-loop TACO runner. Execution and comparison match the OpenCompass TACO evaluator.

from __future__ import annotations

import faulthandler
import json
import multiprocessing
import os
import platform
import re
import signal
import sys
import types
from datetime import datetime
from enum import Enum
from io import StringIO
from unittest.mock import mock_open, patch

import numpy as np

try:
    from pyext import RuntimeModule
except ImportError:
    RuntimeModule = None


class _FallbackRuntimeModule:
    @staticmethod
    def from_string(name, _doc, code):
        module = types.ModuleType(name)
        exec(compile(code, name, "exec"), module.__dict__)
        return module


def _runtime_module():
    return RuntimeModule if RuntimeModule is not None else _FallbackRuntimeModule


TIMEOUT = 10
_FAIL_LOG_N = int(os.environ.get("TACO_FAIL_LOG_N", "2"))
_DEBUG_CLIP = 800


def post_process(text):
    if "```" in text:
        blocks = re.findall(r"```(.*?)```", text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split("```")[1]
        else:
            text = blocks[0]
            if not text.startswith("\n"):
                text = text[max(text.find("\n") + 1, 0):]
    return text


def _parse_ground_truth(ground_truth):
    if isinstance(ground_truth, dict):
        return ground_truth
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)
    return json.loads(ground_truth)


def _to_bool_results(raw):
    fixed = []
    for e in raw:
        if isinstance(e, np.ndarray):
            e = e.item(0)
        if isinstance(e, np.bool_):
            e = bool(e)
        fixed.append(e)
    return fixed


def _clip_debug(obj, limit=_DEBUG_CLIP):
    if obj is None:
        return None
    if isinstance(obj, str):
        text = obj
    else:
        try:
            text = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            text = repr(obj)
    if len(text) > limit:
        return text[:limit] + f"...[+{len(text) - limit} chars]"
    return text


def _fill_fail_debug(box, **kwargs):
    if box is None or box:
        return
    for key, value in kwargs.items():
        if key in {"input", "expected", "output", "code", "error", "prompt"}:
            value = _clip_debug(value)
        box[key] = value


def _run_all_tests(solution_str, ground_truth, extra_info=None):
    """Run every test case. Returns (results, fail_debug)."""
    extra_info = extra_info or {}
    code = post_process(solution_str)
    in_outs = _parse_ground_truth(ground_truth)
    sample = {"input_output": json.dumps(in_outs)}
    n_cases = len(in_outs.get("inputs") or [])
    timeout = min(max(TIMEOUT, 4 * max(n_cases, 1) + 1), 120)
    raw, fail_debug = check_correctness(sample, code, timeout=timeout, debug=False)
    results = _to_bool_results(raw)
    n_pass = sum(1 for r in results if r is True)
    if fail_debug is None:
        fail_debug = {}
    if any(r is not True for r in results) and not fail_debug:
        fail_idx = next((i for i, r in enumerate(results) if r is not True), 0)
        fail_debug = {
            "result": results[fail_idx] if results else -1,
            "fail_index": fail_idx,
            "error": "no_per_case_debug",
        }
    if fail_debug:
        fail_debug.setdefault("code", _clip_debug(code, 1200))
        fail_debug.setdefault("n_tests", n_cases)
        fail_debug.setdefault("n_pass", n_pass)
        if isinstance(extra_info, dict):
            fail_debug.setdefault("problem_id", extra_info.get("problem_id"))
            fail_debug.setdefault("code_type", extra_info.get("code_type"))
            fail_debug.setdefault("fn_name", extra_info.get("fn_name") or in_outs.get("fn_name"))
        return results, fail_debug
    return results, None


def _summarize(results):
    n = len(results) if results else 0
    if n == 0:
        return 0.0, 0.0
    n_pass = sum(1 for r in results if r is True)
    pass_rate = n_pass / n
    binary = 1.0 if n_pass == n else 0.0
    return float(pass_rate), float(binary)


def compute_score_pass_rate(solution_str, ground_truth, extra_info=None, **kwargs):
    """Fraction of test cases passed, in [0, 1]."""
    results, _ = _run_all_tests(solution_str, ground_truth, extra_info)
    pass_rate, _ = _summarize(results)
    return pass_rate


def compute_score_binary(solution_str, ground_truth, extra_info=None, **kwargs):
    """1.0 iff every test case passed, else 0.0."""
    results, _ = _run_all_tests(solution_str, ground_truth, extra_info)
    _, binary = _summarize(results)
    return binary


def _is_validate(extra_info):
    if not isinstance(extra_info, dict):
        return False
    if extra_info.get("validate"):
        return True
    return extra_info.get("split") == "test"


def compute_score(solution_str=None, ground_truth=None, extra_info=None, data_source=None, **kwargs):
    """verl entry. Train and val both use strict 0/1."""
    if solution_str is None:
        solution_str = kwargs.get("completion") or kwargs.get("generation") or ""
    results, fail_debug = _run_all_tests(solution_str, ground_truth, extra_info)
    pass_rate, binary = _summarize(results)
    score = binary
    return {
        "score": float(score),
        "acc": float(binary),
        "pass_rate": float(pass_rate),
        "fail_debug": fail_debug,
    }


def _question_for_log(prompt):
    if not prompt:
        return None
    text = str(prompt)
    if "QUESTION:" in text:
        text = "QUESTION:" + text.split("QUESTION:", 1)[1]
    if "\nANSWER:" in text:
        text = text.split("\nANSWER:", 1)[0]
    return _clip_debug(text, 800)


def _code_for_log(code):
    if not code:
        return code
    text = str(code).lstrip()
    if text.lower().startswith("python"):
        text = text[6:].lstrip("\n")
    return _clip_debug(text, 1200)


def log_taco_failures(fail_debug_list, *, prompts=None, tag="train", step=None, n=None):
    """Print up to n failed samples (question / code / expected / output)."""
    limit = _FAIL_LOG_N if n is None else n
    printed = 0
    for i, dbg in enumerate(fail_debug_list or []):
        if not dbg:
            continue
        print("=" * 72, flush=True)
        print(f"[TACO-FAIL] split={tag} step={step} idx={i}", flush=True)
        print(f"  problem_id={dbg.get('problem_id')} code_type={dbg.get('code_type')} fn_name={dbg.get('fn_name')}", flush=True)
        print(f"  n_pass/n_tests={dbg.get('n_pass')}/{dbg.get('n_tests')} fail_index={dbg.get('fail_index')} result={dbg.get('result')}", flush=True)
        if prompts is not None and i < len(prompts) and prompts[i]:
            print(f"[question] {_question_for_log(prompts[i])}", flush=True)
        if dbg.get("error"):
            print(f"[error] {dbg.get('error')}", flush=True)
        print(f"[code] {_code_for_log(dbg.get('code'))}", flush=True)
        print(f"[input] {dbg.get('input')}", flush=True)
        print(f"[expected] {dbg.get('expected')}", flush=True)
        print(f"[output] {dbg.get('output')}", flush=True)
        printed += 1
        if printed >= limit:
            break
    if printed:
        print("=" * 72, flush=True)


def _temp_run(sample, generation, debug, result):
    # Child process body. Always os._exit so reliability_guard cannot hang join().
    try:
        fail_debug = {}
        results = run_test(sample, test=generation, debug=debug, fail_debug=fail_debug)
        # Manager uses pickle; a list subclass defined here is not importable in the parent.
        result.append((list(results), dict(fail_debug) if fail_debug else None))
    except Exception as exc:
        result.append(([-1], {"result": -1, "error": f"child: {exc!r}", "fail_index": 0}))
    finally:
        os._exit(0)


class _ResultList(list):
    """list.append is read-only on 3.12+; subclass to record the first failure."""

    def __init__(self, fail_debug, ctx):
        super().__init__()
        self._fail_debug = fail_debug
        self._ctx = ctx

    def append(self, item):
        super().append(item)
        if item is not True:
            _fill_fail_debug(
                self._fail_debug,
                result=item,
                fail_index=self._ctx["index"],
                input=self._ctx["inputs"],
                expected=self._ctx["expected"],
                output=self._ctx["output"],
                error=self._ctx["error"],
            )


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.

    Returns (results, fail_debug).
    """

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        n = len(in_outs.get("inputs") or [])
        if debug:
            print("global timeout")
        return [-1 for _ in range(n)], {"result": -1, "error": "global_timeout", "fail_index": 0}
    payload = result[0]
    if isinstance(payload, tuple) and len(payload) == 2:
        return payload[0], payload[1]
    return payload, None


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print('alarm went off')

    raise TimeoutException


def _install_sigalrm():
    # Must run in the scoring child process (main thread). Importing this
    # module from a Ray actor thread cannot call signal.signal.
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
    except (AttributeError, ValueError):
        pass


timeout = 4  # seconds


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def run_test(sample, test=None, debug=False, fail_debug=None):
    """If test(generated_code) is not None it'll try to run the code.

    otherwise it'll just return an input and output pair.
    """
    # Disable functionalities that can make destructive changes to the test.
    reliability_guard()
    _install_sigalrm()

    if debug:
        print(f'start = {datetime.now().time()}')

    try:
        in_outs = json.loads(sample['input_output'])
    except ValueError:
        in_outs = None
    if in_outs:
        if in_outs.get('fn_name') is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs['fn_name']

    if debug:
        print(f'loaded input_output = {datetime.now().time()}')

    if test is None:
        return in_outs
    elif test is not None:
        ctx = {"index": None, "inputs": None, "expected": None, "output": None, "error": None}
        results = _ResultList(fail_debug, ctx)
        sol = 'import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n'  # noqa: E501
        if debug:
            print(f'loading test code = {datetime.now().time()}')

        if which_type == CODE_TYPE.call_based:
            sol += test
            if debug:
                print(f'sol = {sol}')
            signal.alarm(timeout)
            try:
                tmp_sol = _runtime_module().from_string('tmp_sol', '', sol)
                if 'class Solution' not in test:
                    tmp = tmp_sol
                else:
                    tmp = tmp_sol.Solution()
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f'type 0 compilation error = {e}')
                ctx["error"] = f"compile: {e!r}"
                results.append(-2)
                return results
            signal.alarm(0)

        elif which_type == CODE_TYPE.standard_input:
            # sol
            tmp_test = test.split('\n')

            new_test = []
            for x in tmp_test:
                if (not x.startswith('from ')) and (
                        not x.startswith('import ')):
                    new_test.append('\t' + x + '\n')
                else:
                    new_test.append(x + '\n')
            tmp_test = new_test

            new_test = ''
            started = False
            for i in tmp_test:
                if i.startswith('\t') and not started:
                    new_test += 'stdin = sys.stdin\nstdout = sys.stdout\n'
                    new_test += 'def code():\n'
                    new_test += i
                    started = True
                elif started and ((i.startswith('from ')) or
                                  (i.startswith('import '))):
                    new_test += '\t' + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            if debug:
                print(f'sol = {sol}')
            method_name = 'code'
            signal.alarm(timeout)
            try:
                tmp_sol = _runtime_module().from_string('tmp_sol', '', sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f'type 1 compilation error = {e}')
                ctx["error"] = f"compile: {e!r}"
                results.append(-2)
                return results
            signal.alarm(0)
        if debug:
            print(f'get method = {datetime.now().time()}')

        try:
            method = getattr(tmp,
                             method_name)  # get_attr second arg must be str
        except:  # noqa: E722
            signal.alarm(0)
            e = sys.exc_info()
            print(f'unable to get function error = {e}')
            ctx["error"] = f"getattr: {e!r}"
            results.append(-2)
            return results

        for index, inputs in enumerate(in_outs['inputs']):
            ctx["index"] = index
            ctx["inputs"] = _clip_debug(inputs)
            ctx["expected"] = _clip_debug(in_outs['outputs'][index])
            ctx["output"] = None
            ctx["error"] = None
            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list) # noqa: E501
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k, v in inputs[0].items()}]
            except:  # noqa: E722
                True
            try:
                if isinstance(in_outs['outputs'][index], dict):
                    in_outs['outputs'][index] = [{
                        int(k): v
                        for k, v in in_outs['outputs'][index].items()
                    }]
            except:  # noqa: E722
                True
            try:
                if isinstance(in_outs['outputs'][index][0], dict):
                    in_outs['outputs'][index] = [{
                        int(k): v
                        for k, v in in_outs['outputs'][index][0].items()
                    }]
            except:  # noqa: E722
                True

            if debug:
                print(
                    f'time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}'  # noqa: E501
                )
            if which_type == CODE_TYPE.call_based:  # Call-based
                signal.alarm(timeout)
                faulthandler.enable()
                try:
                    output = method(*inputs)
                    ctx["output"] = output

                    # ground truth sequences are not tuples
                    if isinstance(output, tuple):
                        output = list(output)

                    tmp_result = output == in_outs['outputs'][index]
                    if isinstance(in_outs['outputs'][index],
                                  list) and in_outs['outputs'][index]:
                        tmp_result = tmp_result or (
                            output == in_outs['outputs'][index][0])

                    # ground truth sequences are not tuples
                    try:
                        if isinstance(output[0], tuple):
                            tmp_result = tmp_result or ([
                                list(x) for x in output
                            ] == in_outs['outputs'][index][0])
                    except:  # noqa: E722
                        True
                    results.append(tmp_result)

                    # reset the alarm
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    faulthandler.disable()
                    if debug:
                        print(
                            f'Standard input runtime error or time limit exceeded error = {e}'  # noqa: E501
                        )
                    ctx["error"] = f"runtime: {e!r}"
                    results.append(-1)
                    continue
                faulthandler.disable()
                signal.alarm(0)
                if debug:
                    print(
                        f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"  # noqa: E501
                    )
            elif which_type == CODE_TYPE.standard_input:  # Standard input
                faulthandler.enable()
                signal.alarm(timeout)
                passed = False

                if isinstance(inputs, list):
                    inputs = '\n'.join(inputs)
                if isinstance(in_outs['outputs'][index], list):
                    in_outs['outputs'][index] = '\n'.join(
                        in_outs['outputs'][index])

                with Capturing() as output:
                    try:
                        call_method(method, inputs)
                        # reset the alarm
                        signal.alarm(0)
                        passed = True
                    except Exception as e:
                        # runtime error or took too long
                        signal.alarm(0)
                        print(
                            f'Call-based runtime error or time limit exceeded error = {repr(e)}{e}'  # noqa: E501
                        )
                        ctx["error"] = f"runtime: {e!r}"
                        results.append(-1)
                    signal.alarm(0)
                ctx["output"] = list(output)

                if not passed:
                    if debug:
                        nl = '\n'
                        if not isinstance(inputs, list):
                            print(
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"  # noqa: E501
                            )
                        else:
                            print(
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"  # noqa: E501
                            )
                    continue

                if passed and debug:
                    print(
                        f"==> output = {output}, test outputs = {in_outs['outputs'][index]}"  # noqa: E501
                    )

                if custom_compare_(output, in_outs['outputs'][index]):
                    tmp_result = True
                    results.append(tmp_result)
                    continue

                # ground truth sequences are expressed as lists not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = False
                try:
                    tmp_result = (output == [in_outs['outputs'][index]])
                    if isinstance(in_outs['outputs'][index], list):
                        tmp_result = tmp_result or (
                            output == in_outs['outputs'][index])
                        if isinstance(output[0], str):
                            tmp_result = tmp_result or ([
                                e.strip() for e in output
                            ] == in_outs['outputs'][index])
                except Exception as e:
                    if debug:
                        print(f'Failed check1 exception = {e}')
                    pass

                if tmp_result == True:  # noqa: E712
                    results.append(tmp_result)
                    continue

                # try one more time without \n
                if isinstance(in_outs['outputs'][index], list):
                    for tmp_index, i in enumerate(in_outs['outputs'][index]):
                        in_outs['outputs'][index][tmp_index] = i.split('\n')
                        in_outs['outputs'][index][tmp_index] = [
                            x.strip()
                            for x in in_outs['outputs'][index][tmp_index] if x
                        ]
                else:
                    in_outs['outputs'][index] = in_outs['outputs'][
                        index].split('\n')
                    in_outs['outputs'][index] = list(
                        filter(len, in_outs['outputs'][index]))
                    in_outs['outputs'][index] = list(
                        map(lambda x: x.strip(), in_outs['outputs'][index]))

                try:
                    tmp_result = (output == [in_outs['outputs'][index]])
                    if isinstance(in_outs['outputs'][index], list):
                        tmp_result = tmp_result or (
                            output == in_outs['outputs'][index])
                except Exception as e:
                    if debug:
                        print(f'Failed check2 exception = {e}')
                    pass

                if tmp_result == True:  # noqa: E712
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    output = list(filter(len, output))

                if debug:
                    nl = '\n'
                    if not isinstance(inputs, list):
                        print(
                            f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"  # noqa: E501
                        )
                    else:
                        print(
                            f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"  # noqa: E501
                        )

                if tmp_result == True:  # noqa: E712
                    results.append(tmp_result)
                    continue

                try:
                    tmp_result = (output == [in_outs['outputs'][index]])
                    if isinstance(in_outs['outputs'][index], list):
                        tmp_result = tmp_result or (
                            output == in_outs['outputs'][index])
                except Exception as e:
                    if debug:
                        print(f'Failed check3 exception = {e}')
                    pass

                try:
                    output_float = [float(e) for e in output]
                    gt_float = [float(e) for e in in_outs['outputs'][index]]
                    tmp_result = tmp_result or (
                        (len(output_float) == len(gt_float))
                        and np.allclose(output_float, gt_float))
                except Exception:
                    pass
                try:
                    if isinstance(output[0], list):
                        output_float = [float(e) for e in output[0]]
                        gt_float = [
                            float(e) for e in in_outs['outputs'][index][0]
                        ]
                        tmp_result = tmp_result or (
                            (len(output_float) == len(gt_float))
                            and np.allclose(output_float, gt_float))
                except Exception:
                    pass

                if tmp_result == True:  # noqa: E712
                    results.append(tmp_result)
                    continue

                # try by converting the stuff into split up list
                if isinstance(in_outs['outputs'][index], list):
                    for tmp_index, i in enumerate(in_outs['outputs'][index]):
                        in_outs['outputs'][index][tmp_index] = set(i.split())
                else:
                    in_outs['outputs'][index] = set(
                        in_outs['outputs'][index].split())

                try:
                    tmp_result = (output == in_outs['outputs'][index])
                except Exception as e:
                    if debug:
                        print(f'Failed check4 exception = {e}')
                    continue

                if tmp_result == True:  # noqa: E712
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = i.split()
                    output = list(filter(len, output))
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = set(i)
                else:
                    output = output.split()
                    output = list(filter(len, output))
                    output = set(output)

                try:
                    tmp_result = (set(frozenset(s) for s in output) == set(
                        frozenset(s) for s in in_outs['outputs'][index]))
                except Exception as e:
                    if debug:
                        print(f'Failed check5 exception = {e}')

                # if they are all numbers, round so that similar numbers are treated as identical # noqa: E501
                try:
                    tmp_result = tmp_result or (
                        set(
                            frozenset(round(float(t), 3) for t in s)
                            for s in output) ==  # noqa: E502, E128
                        set(
                            frozenset(round(float(t), 3) for t in s)
                            for s in in_outs['outputs'][index]))  # noqa: E501
                except Exception as e:
                    if debug:
                        print(f'Failed check6 exception = {e}')

                if tmp_result == True and debug:  # noqa: E712
                    print('PASSED')

                results.append(tmp_result)

                if debug:
                    nl = '\n'
                    if not isinstance(inputs, list):
                        print(
                            f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"  # noqa: E501
                        )
                    else:
                        print(
                            f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"  # noqa: E501
                        )

    return results


def custom_compare_(output, ground_truth):

    if isinstance(output, list):
        output_1 = '\n'.join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = '\n'.join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = '\n'.join(inputs)

    inputs_line_iterator = iter(inputs.split('\n'))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', StringIO(inputs))
    @patch('sys.stdin.readline', lambda *args: next(inputs_line_iterator))
    @patch('sys.stdin.readlines', lambda *args: inputs.split('\n'))
    @patch('sys.stdin.read', lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


def reliability_guard(maximum_memory_bytes=None):
    """This disables various destructive functions and prevents the generated
    code from interfering with the test (e.g. fork bomb, killing other
    processes, removing filesystem files, etc.) WARNING This function is NOT a
    security sandbox.

    Untrusted code, including, model- generated code, should not be blindly
    executed outside of one. See the Codex paper for more information about
    OpenAI's code sandbox, and proceed with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS,
                           (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA,
                           (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK,
                               (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    os.environ['OMP_NUM_THREADS'] = '1'

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__['help'] = None

    import sys

    sys.modules['ipdb'] = None
    sys.modules['joblib'] = None
    sys.modules['resource'] = None
    sys.modules['psutil'] = None
    sys.modules['tkinter'] = None
