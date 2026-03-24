import os
import sys
import wandb
import torch
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from utils import set_seed, ensure_tokenizer_special_tokens
from models.s2s_model import S2SModel
from modules.s2s_module import S2SLightningModule, AdapterLightningModule
from modules.data_module import S2SDataModule

def main():
    cli_cfg = OmegaConf.from_cli()
    if "config" not in cli_cfg:
        print("Usage: python train.py config=configs/adapter.yaml [overrides...]")
        sys.exit(1)
    stage_cfg = OmegaConf.load(cli_cfg.config)
    cfg = OmegaConf.merge(stage_cfg, cli_cfg)
    set_seed(cfg.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    tokenizer = ensure_tokenizer_special_tokens(tokenizer)
    model = S2SModel(vocab_size=len(tokenizer), pad_id=tokenizer.pad_token_id, sil_id=tokenizer.sil_token_id, llm_name=cfg.model.name)
    
    if "speech_encoder" in cfg and cfg.speech_encoder.get("ckpt_path"):
        model.load_pretrained_speech_encoder(cfg.speech_encoder.ckpt_path)
    
    if "adapter" in cfg and cfg.adapter.get("ckpt_path"):
        model.load_pretrained_adapter(cfg.adapter.ckpt_path)
    
    if cfg.training.get("grad_checkpointing", False):
        model.enable_gradient_checkpointing()
    
    # Train Module
    if cfg['stage'] == 'adapter':
        for param in model.llm.parameters():
            param.requires_grad = False
        model.freeze_speech_encoder(num_layers=cfg.speech_encoder.num_freeze_layers)
        lightning_module = AdapterLightningModule(model, tokenizer, cfg)
    elif cfg['stage'] == 'S2S':
        model.freeze_speech_encoder(num_layers=cfg.speech_encoder.num_freeze_layers)
        lightning_module = S2SLightningModule(model, tokenizer, cfg)
    else:
        raise ValueError(f"Unknown stage: {cfg['stage']}")
    
    if cfg.training.get("load_weights_from_ckpt", None):
        ckpt_path = cfg.training.load_weights_from_ckpt
        print(f">>> Loading weights from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        lightning_module.load_state_dict(ckpt["state_dict"], strict=True)

    # Data Module
    data_module = S2SDataModule(cfg, tokenizer)

    # WandB
    if cfg.get("wandb", {}).get("enabled", True):
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, dir=cfg.wandb.log_dir)
        logger = pl.loggers.WandbLogger(log_model=False)
    else:
        logger = None
    
    run_ckpt_dir = os.path.join(cfg.training.checkpoint_dir, cfg.wandb.run_name)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    tokenizer.save_pretrained(run_ckpt_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_ckpt_dir,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        logger=logger,
        precision=cfg.training.mixed_precision,
        gradient_clip_val=cfg.training.get("grad_clip", 1.0),
        accumulate_grad_batches=cfg.training.get("grad_accum", 2),
        log_every_n_steps=cfg.logging.log_interval,
        check_val_every_n_epoch=None,
        val_check_interval=cfg.training.val_steps,
        default_root_dir=cfg.training.checkpoint_dir,
        use_distributed_sampler=False,
        callbacks=[checkpoint_callback],
    )
    
    # Train
    trainer.fit(lightning_module, datamodule=data_module, ckpt_path=cfg.training.resume_from_ckpt)

if __name__ == "__main__":
    main()