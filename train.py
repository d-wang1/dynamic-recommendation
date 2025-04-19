import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from pathlib import Path

from data.datasets import ML1MDataModule
from data.helper import gen_demographic_table, load_save_neumf_table, gen_datamodule, load_config
from models.hyper_neumf import DCHyperNeuMF


cfg = load_config()
verbose = cfg["verbose"]
dm, n_users, n_movies, demog = gen_datamodule(k=3, test_size=0.2, random_state=42, verbose=verbose)

d_emb = cfg["model_hyperparams"]["d_emb"]

model = DCHyperNeuMF(
    n_movies = n_movies,
    d_emb = d_emb,
    demog_dim = demog.shape[1],
    mlp_layers = cfg["model_hyperparams"]["mlp_layers"],
    lr = cfg["model_hyperparams"]["lr"]
)

if verbose:
    print("=" * 20)
    print("Created model with", n_movies, "movies and", d_emb, "embedding dimension.")
    print("Model hyperparameters:", model.hparams)
    print("=" * 20)

neumf_table_path =(Path(__file__).resolve().parent / cfg["neumf_table"]).resolve()
if not os.path.exists(neumf_table_path):
    if verbose:
        print("Neumf table not found. Generating it...")
    load_save_neumf_table()

if verbose:
    print("Loading pre-trained GMF and MLP tables from", neumf_table_path)
tables = torch.load(neumf_table_path, map_location="cpu")

model.item_gmf.weight.data.copy_(tables["item_gmf"])
model.item_mlp.weight.data.copy_(tables["item_mlp"])

# Freeze the GMF and MLP weights so that they are not updated during training
for p in [*model.item_gmf.parameters(), *model.item_mlp.parameters()]:
    p.requires_grad_(False)


# Set up the logger
api_key = os.getenv("COMET_API_KEY")
if api_key is None:
    if verbose:
        print("COMET_API_KEY environment variable not found. Checking for config.json file instead...")
    api_key = cfg["comet_logger"]["api_key"]
    if api_key is None:
        if verbose:
            print("No API key found. Not logging to Comet.")

if api_key:
    comet_logger = CometLogger(
        api_key = api_key,      # or hard‑code
        project_name="hyper-neumf-movielens",
        # workspace="your‑comet‑workspace",          # optional
        log_code=True,
    )
    comet_logger.log_hyperparams(model.hparams)


train_cfg = cfg["train"]
ckpt_dir = (Path(__file__).resolve().parent / train_cfg["ckpt_dir"]).resolve()

if verbose:
    print("Checkpoint directory:", ckpt_dir)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
    if verbose:
        print("Checkpoint directory does not exist. Creating it...")


ckpt_cb = ModelCheckpoint(
    monitor="val_rmse",
    mode="min",
    dirpath=ckpt_dir,
    filename="epoch{epoch:02d}-rmse{val_rmse:.3f}",
    save_top_k=train_cfg["save_top_k_checkpoints"],
    save_last=True
)
lr_cb = LearningRateMonitor(logging_interval="epoch")


trainer = pl.Trainer(
    max_epochs=train_cfg["max_epochs"],
    accelerator=train_cfg["accelerator"],
    logger=comet_logger,
    callbacks=[ckpt_cb, lr_cb],
    log_every_n_steps=train_cfg["log_every_n_steps"],
    enable_progress_bar=train_cfg["enable_progress_bar"]
)

trainer.fit(model, datamodule=dm)
trainer.validate(model, datamodule=dm, ckpt_path="best")
