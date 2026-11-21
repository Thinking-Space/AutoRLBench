# Package checks

mechanism_check.json is the newly executed CPU-only socket mechanism probe. preseal_evidence_check.json is the newly executed consistency check before MANIFEST creation; its unsealed field explicitly records that construction stage. Neither is a new GPU/RL experiment.

After sealing, repro/verify_bundle.py is run again against the actual MANIFEST and all shipped content; recipients can repeat that read-only command. ZIP CRC and archived SHA-256 values are also verified when packaging.
