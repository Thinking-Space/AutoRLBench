"""Configure training and validation prompt preprocessing independently."""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from verl.utils.dataset.rl_dataset import RLHFDataset


def _paths(files):
    if isinstance(files, (str, Path)):
        files = [files]
    return tuple(str(Path(path).expanduser().resolve()) for path in files)


class SplitPromptDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        requested = _paths(data_files)
        train, validation = _paths(config.train_files), _paths(config.val_files)
        if train == validation or requested not in (train, validation):
            raise ValueError("Expected distinct, explicitly configured train and validation files")
        self.dapo_split = "train" if requested == train else "val"
        local_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        if self.dapo_split == "val":
            local_config.max_prompt_length = int(config.validation_max_prompt_length)
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=local_config,
                         processor=processor, max_samples=max_samples)


def probe(config_path, output):
    """Exercise the real veRL factory and dataset on duplicate, long-prompt fixtures."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from verl.trainer.main_ppo import create_rl_dataset

    config = OmegaConf.load(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path,
                                              local_files_only=True, trust_remote_code=False)
    fixture = output.parent / "dataset-preflight-fixture"
    fixture.mkdir(exist_ok=True)
    messages = [[{"role": "user", "content": "Remember 17 and 5. " + "context " * count
                 + "Find their sum. Answer:"}] for count in (300, 2500)]
    raw_ids = [tokenizer.encode(tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=False,
        **dict(config.data.apply_chat_template_kwargs)), add_special_tokens=False)
        for prompt in messages]
    if not (int(config.data.max_prompt_length) <= 2048
            and 128 < len(raw_ids[0]) < 2048 < len(raw_ids[1])):
        raise ValueError("Unexpected prompt fixture lengths or configured cap")
    rows = [{"prompt": prompt, "data_source": "dapo-preflight", "ability": "math",
             "reward_model": {"style": "rule", "ground_truth": "22"},
             "extra_info": {"index": index}} for index, prompt in enumerate(messages)
            for _ in range(2)]
    for split in ("train", "val"):
        path = fixture / (split + ".parquet")
        pq.write_table(pa.Table.from_pylist(rows), path)
        config.data[split + "_files"] = str(path)
    original_cap = config.data.max_prompt_length
    measurements = {}
    for split in ("train", "val"):
        dataset = create_rl_dataset(config.data[split + "_files"], config.data, tokenizer,
                                    processor=None, is_train=split == "train", max_samples=-1)
        cap = int(original_cap) if split == "train" else 2048
        samples = [dataset[index] for index in range(len(dataset))]
        valid = (len(dataset) == 4 and dataset.max_prompt_length == cap
                 and config.data.max_prompt_length == original_cap
                 and dataset.dapo_split == split)
        for index, sample in enumerate(samples):
            expected = raw_ids[index // 2][-cap:]
            valid = valid and sample["raw_prompt_ids"] == expected
            valid = valid and sample["input_ids"].numel() == cap
            valid = valid and sample["attention_mask"].sum().item() == len(expected)
        measurements[split] = {"max_prompt_length": dataset.max_prompt_length,
                               "rows": len(dataset), "source_prompt_lengths": list(map(len, raw_ids)),
                               "observed_raw_lengths": [len(row["raw_prompt_ids"]) for row in samples],
                               "duplicate_rows_preserved": len(dataset) == len(rows),
                               "left_truncation_and_padding_verified": bool(valid)}
        if not valid:
            raise ValueError("Actual RLHFDataset behavior disagrees with split configuration")
    result = {"status": "ok", "fixture_only": True, "shared_config_unchanged": True,
              "measurements": measurements}
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.config, args.output)
