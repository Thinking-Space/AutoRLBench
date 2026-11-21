"""Check the published aggregate identities. Uses only the Python standard library."""

import json
from math import isclose
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    data = json.loads(Path(__file__).with_name('aggregate-evidence.json').read_text())
    cohorts = data['frozen_trajectory_diagnostic']
    require(set(cohorts) == {'128k_two_compactions', '256k_zero_compactions'}, 'expected both cohorts')
    total_tokens = 0
    for label, cohort in cohorts.items():
        primary = cohort['primary_pre_group_mask']
        matched = cohort['same_tokens_conditioned_pre_group_mask']
        require(primary['tokens'] == matched['tokens'], 'scoring populations differ')
        require(cohort['support_validation_fraction'] == 1 and cohort['trajectory_count'] == 16,
                'reported support coverage or trajectory count changed')
        total_tokens += primary['tokens']

        # Row-wise, log p(y|S) = log p(y) - log p(S). Check its average against
        # the independently recorded mean log mass of the sampled support.
        shift = primary['signed_train_minus_roll'] - matched['signed_train_minus_roll']
        require(isclose(shift, cohort['mean_log_trainer_support_mass'], rel_tol=0, abs_tol=1e-7),
                'signed mean shift does not match the recorded support mass')
        for metric in ('abs_diff', 'k3'):
            require(primary[metric] > matched[metric] > 0, f'{label}: unexpected {metric} ordering')

        # Under identical unfiltered scores, q(y)=p(y)/p(S), hence p(y)/q(y)=p(S)<=1.
        # An observed nonzero tail above 2 contradicts that single-cause hypothesis.
        before = primary['ratio_above_2_fraction']
        after = matched['ratio_above_2_fraction']
        require(0 < before < after <= 1, f'{label}: unexpected high-ratio fraction')
        print(f"{label} ({primary['tokens']:,} positions): "
              f"abs {primary['abs_diff']:.5f} -> {matched['abs_diff']:.5f}, "
              f"K3 {primary['k3']:.5f} -> {matched['k3']:.5f}, "
              f"ratio > 2: {before:.4%} -> {after:.4%}, mean log support mass {shift:.6f}")
    require(total_tokens == 3_919_116, 'unexpected total population')
    print(f'{total_tokens:,} scoring positions across the two cohorts.')


if __name__ == '__main__':
    main()
