"""Load tokenization inputs through the prepared structured-message cache."""
from pathlib import Path
import hashlib
import json
import shutil

import pyarrow.parquet as pq
from verl.utils.dataset.rl_dataset import RLHFDataset

POSITION_COLUMN = 'cache_version_position'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, description):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != description['size_bytes']:
        raise ValueError('Prepared artifact missing or size changed')
    if sha256(path) != description['sha256']:
        raise ValueError('Prepared artifact content changed')


def cache_is_valid(metadata, expected):
    identity = metadata.get('identity', {})
    return (identity.get('source_key') == expected['source_key']
            and identity.get('rows') == expected['rows'])


def paths(files):
    if isinstance(files, (str, Path)):
        files = [files]
    return tuple(str(Path(path).resolve()) for path in files)


class PromptDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        requested = paths(data_files)
        train, val = paths(config.train_files), paths(config.val_files)
        if requested == train and requested != val:
            training = True
        elif requested == val and requested != train:
            training = False
        else:
            raise ValueError('Expected distinct explicit train and validation files')
        self._cached_messages = None
        super().__init__(data_files, tokenizer, config, processor=processor, max_samples=max_samples)
        if training:
            self._initialize_cache(config)

    def _initialize_cache(self, config):
        if len(paths(config.train_files)) != 1:
            raise ValueError('Expected one training source')
        manifest = json.loads(Path(config.cache_manifest_path).read_text())
        source = Path(paths(config.train_files)[0])
        verify_file(source, manifest['files']['train.parquet'])
        total_rows = pq.ParquetFile(source).metadata.num_rows
        if POSITION_COLUMN not in self.dataframe.column_names:
            raise ValueError('Missing prepared row position column')
        expected = {'source_key': 'train.parquet', 'rows': total_rows,
                    'source_sha256': manifest['files']['train.parquet']['sha256'],
                    'preprocessing': manifest['processing_contract']}
        seed = Path(config.cache_seed_path)
        seed_metadata = seed.with_suffix('.json')
        verify_file(seed, manifest['files'][seed.name])
        verify_file(seed_metadata, manifest['files'][seed_metadata.name])
        root = Path(config.cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        active = root/'processed-prompts.parquet'
        metadata_path = active.with_suffix('.json')
        if not active.exists() and not metadata_path.exists():
            shutil.copy2(seed, active)
            shutil.copy2(seed_metadata, metadata_path)
        if not active.is_file() or not metadata_path.is_file():
            raise ValueError('Incomplete active cache; use a fresh output directory')
        metadata = json.loads(metadata_path.read_text())
        verify_file(active, metadata['artifact'])
        if not cache_is_valid(metadata, expected):
            temporary = active.with_suffix('.parquet.tmp')
            pq.write_table(pq.read_table(source, columns=[self.prompt_key]), temporary, compression='zstd')
            temporary.replace(active)
            metadata = {'identity': expected,
                        'artifact': {'sha256': sha256(active), 'size_bytes': active.stat().st_size}}
            temporary_json = metadata_path.with_suffix('.json.tmp')
            temporary_json.write_text(json.dumps(metadata, sort_keys=True, indent=2)+'\n')
            temporary_json.replace(metadata_path)
        cached = pq.read_table(active, columns=[self.prompt_key])
        if cached.num_rows != total_rows:
            raise ValueError('Cache length disagrees with source')
        self._cached_messages = cached[self.prompt_key]

    def _build_messages(self, example):
        if self._cached_messages is not None:
            position = int(example[POSITION_COLUMN])
            if not 0 <= position < len(self._cached_messages):
                raise ValueError('Cache position outside prepared source')
            example[self.prompt_key] = self._cached_messages[position].as_py()
        return super()._build_messages(example)
