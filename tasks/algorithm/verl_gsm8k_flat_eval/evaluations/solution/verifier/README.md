# Solution verifier notes

The verifier reads the held-out **reward** metric `val-core/openai/gsm8k/reward/mean@1` from the final training step (50). It requires at least `0.74609375`.

Calibration points: observed arm steps 5–50 have maximum `0.734375`; the baseline arm has minimum `0.75390625`; baseline step 0 is `0.73828125`. Thus the threshold is `0.01171875` above the observed maximum, `0.0078125` below the baseline minimum, and `0.0078125` above step 0. It uses one recorded run per arm, so only these margins are reported; no variance estimate is claimed.
