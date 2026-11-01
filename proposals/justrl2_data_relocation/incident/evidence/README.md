# Measured reference continuation

These files describe native reference run 1430316, continuing the preserved
step-3 actor/critic state from prefix1430222. They are separate from the
historical eight-update fault demonstrations and from the packaged-task
validation. No LLM agent was evaluated.

- `native-acceptance.json`: fresh acceptance performed by the native validation
  wrapper, with 64 admitted samples and both final checkpoint payloads rehashed.
- `independent-audit.json`: successful fresh re-audit by CPU job1430462 using
  frozen verifier `db21b6dd…`, including all seven model-asset files and both
  output checkpoint payload inventories.
- `provenance.json`: source job, bundle, archive and evidence-file hashes.
- `restore-events.json`: measured native actor/critic restoration, update and
  final-save events. Only `load_root` and `save_root` paths were removed; all
  numeric observations and state digests are preserved. Original event-file
  hashes are recorded in the file.
- `curves.json` and `continuation-curves.{png,svg}`: actual prefix0–3 and its
  accepted reference continuation4–7. The prefix event file is checked against
  the sealed scene inventory. Reward and hard-sample fraction are observations,
  not acceptance thresholds. This figure does not depict an unpatched
  continuation or establish a model-quality benefit.
- `render.py`: matplotlib renderer, requiring the raw scene metadata, resumed
  run evidence and its fresh acceptance receipt.

The independent-audit archive SHA-256 is
`6b6db9ab464a184651d537945014d75bb7eb5a846321d1a91e295117dd89d034`.

The original resumed-evidence archive SHA-256 is
`d76739c254630ac75ffbde4f38d64896af528a2e55854f1da3fa7338786a26ee`.
The scene receipt SHA-256 is
`b48862441e71acd0bbbf6ad3f75c7168e6bd758255ad130e4f73c9ec1262e9ac`.
Large checkpoints and raw generation logs are distributed separately; they are
not included in this Git evidence directory.
