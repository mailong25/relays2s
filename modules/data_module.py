import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler, make_worker_init_fn, SimpleCutSampler
from lhotse.dataset.iterable_dataset import IterableDatasetWrapper
from data.dataset import S2SDataset

class S2SDataModule(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
    
    def setup(self, stage):

        self.train_cuts = CutSet.from_shar(
            in_dir=self.cfg.data.train_shar,
            shuffle_shards=True,
            stateful_shuffle=True,
            seed="randomized"
        ).repeat()
        
        self.val_cuts = CutSet.from_shar(in_dir=self.cfg.data.val_shar, shuffle_shards=False)

        if self.cfg.data.get("max_val_samples"):
            self.val_cuts = self.val_cuts.subset(first=self.cfg.data.max_val_samples)
        
        self.train_dataset = S2SDataset(
            tokenizer=self.tokenizer,
            frame_length=self.cfg.model.frame_length,
            sampling_rate=self.cfg.data.sampling_rate,
        )
        self.val_dataset = S2SDataset(
            tokenizer=self.tokenizer,
            frame_length=self.cfg.model.frame_length,
            sampling_rate=self.cfg.data.sampling_rate,
        )

    def train_dataloader(self):
        sampler = DynamicBucketingSampler(
            self.train_cuts,
            max_duration=self.cfg.training.batch_duration,
            shuffle=True,
        )
        return DataLoader(
            IterableDatasetWrapper(self.train_dataset, sampler),
            batch_size=None,
            num_workers=self.cfg.data.n_workers,
            worker_init_fn=make_worker_init_fn(seed=0),
        )
    
    def val_dataloader(self):
        sampler = SimpleCutSampler(
            self.val_cuts,
            max_duration=self.cfg.training.val_batch_duration,
            shuffle=False,
        )
        return DataLoader(
            IterableDatasetWrapper(self.val_dataset, sampler),
            batch_size=None,
            num_workers=1,
        )
