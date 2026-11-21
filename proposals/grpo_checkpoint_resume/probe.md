# Independent AdamW CPU probe

Run `python3 adam_moment_probe.py` with PyTorch installed. This standalone check uses a two-action policy, four fixed batches and a constant learning rate. After three updates, it serializes a checkpoint and compares the uninterrupted fourth update with two separate restores. Both restores reuse the identical weights, Adam step, batch cursor, actions and advantages. One restore zeros only the first and second moments.

On CPU with PyTorch 2.14.0 the maximum parameter difference was **0** for the full restore and **0.005756034771962931** after zeroing the moments. The script prints the local values and checks for exact agreement in the full restore and a positive deviation in the other arm. These numbers belong to this standalone two-logit check. The separately recorded one- and two-rank Megatron checkpoint probe used different data and produced the figures in `historical-evidence.json`.
