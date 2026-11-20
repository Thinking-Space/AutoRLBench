# SPDX-License-Identifier: Apache-2.0
"""Maintainer-only scalar PPO and CP ownership reference; no candidate imports."""
import math


def response_indices(total, length, cp, coordinate):
    if cp == 1:
        return list(range(length))
    width = math.ceil(total / (2 * cp))
    prompt = total - length
    chunks = ((coordinate * width, (coordinate + 1) * width),
              ((2 * cp - coordinate - 1) * width, (2 * cp - coordinate) * width))
    return [i for i in range(length)
            if any(start <= prompt - 1 + i < end for start, end in chunks)]


def reference_step(samples_by_dp, cp, order):
    """Input rows use total_length/response_length/values/loss_mask/reward."""
    dp = len(samples_by_dp)
    if sorted(order) != list(range(cp * dp)):
        raise ValueError('Invalid rank mapping')
    rows = {}
    for d, samples in enumerate(samples_by_dp):
        for c in range(cp):
            rank = order[d * cp + c]
            row = dict(raw=[], returns=[], masks=[], sliced_values=[], advantages=[])
            for sample in samples:
                values = sample['values']
                length = sample['response_length']
                if len(values) != length or len(sample['loss_mask']) != length:
                    raise ValueError('Incorrect scalar fixture lengths')
                decay, carry = (0.5 ** (1.0 / length) if length else 0.0), 0.0
                raw = [0.0] * length
                for t in reversed(range(length)):
                    reward = sample['reward'] if t == length - 1 else 0.0
                    carry = reward + (values[t + 1] if t + 1 < length else 0.0) - values[t] + decay * carry
                    raw[t] = carry
                indices = response_indices(sample['total_length'], length, cp, c)
                row['raw'].append([raw[i] for i in indices])
                row['returns'].append([sample['reward'] for i in indices])
                row['sliced_values'].append([values[i] for i in indices])
                row['masks'].extend(sample['loss_mask'][i] for i in indices)
            row['group'] = sorted(order[i * cp + c] for i in range(dp))
            rows[rank] = row
    for c in range(cp):
        ranks = [order[d * cp + c] for d in range(dp)]
        selected = [(x, w) for rank in ranks
                    for x, w in zip([v for chunk in rows[rank]['raw'] for v in chunk], rows[rank]['masks'])]
        count = math.fsum(w for x, w in selected)
        mean = math.fsum(x * w for x, w in selected) / count if count else None
        variance = math.fsum(w * (x - mean) ** 2 for x, w in selected) / (count - 1 if count >= 2 else count) if count else None
        for rank in ranks:
            rows[rank]['count'] = count
            rows[rank]['advantages'] = (
                [[(x - mean) / math.sqrt(variance + 1e-8) for x in chunk] for chunk in rows[rank]['raw']]
                if count else None)
    return rows


def numeric_equal(actual, expected):
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(numeric_equal(x, y) for x, y in zip(actual, expected)))
    return (type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=5e-4, abs_tol=5e-4))
