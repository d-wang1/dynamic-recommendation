import os
import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.datasets import ML1MDataModule
from data.helper import gen_demographic_table, load_save_neumf_table, gen_datamodule, load_config
from models.hyper_neumf import DCHyperNeuMF

# ─── 1) read config ─────────────────────────────────────────────────────────
cfg = load_config()
hp = cfg["model_hyperparams"]
verbose = cfg.get("verbose", False)

# ─── 2) build data module with your k list and batch_size ─────────────────
#    we only support a *fixed* k per DataModule, so we'll train at max_k
#    (if you want random k each batch, you'd have to modify gen_datamodule)
dm, n_users, n_movies, demog = gen_datamodule(
    k_list=hp["k"],
    max_k = hp["max_k"],
    test_size=0.2,
    random_state=42,
    verbose=cfg["verbose"]
)
# override its batch_size
dm.batch_size = hp["batch_size"]

# ─── 3) instantiate your model ──────────────────────────────────────────────
model = DCHyperNeuMF(
    n_movies   = n_movies,
    d_emb      = hp["d_emb"],
    demog_dim  = demog.shape[1],
    mlp_layers = hp["mlp_layers"],
    lr         = hp["lr"],
    max_k      = hp["max_k"],       # <— pass it in
)

if verbose:
    print(f"Using hyperparams: {model.hparams}")

# ─── 4) load & freeze pretrained NeuMF tables ──────────────────────────────
neumf_tbl = Path(__file__).parent / cfg["neumf_table"]
if not neumf_tbl.exists():
    load_save_neumf_table()
item_gmf, item_mlp = torch.load(neumf_tbl, map_location="cpu").values()
model.item_gmf.weight.data.copy_(item_gmf)
model.item_mlp.weight.data.copy_(item_mlp)
# (optionally leave these frozen or fine‑tune them)
for p in model.item_gmf.parameters(): p.requires_grad_(False)
for p in model.item_mlp.parameters(): p.requires_grad_(False)

# ─── 5) set up logging & callbacks ─────────────────────────────────────────
api_key = os.getenv("COMET_API_KEY") or cfg["comet_logger"].get("api_key")
logger  = CometLogger(api_key=api_key, project_name="hyper-neumf-movielens") \
          if api_key else None

ckpt_cb = ModelCheckpoint(
    monitor = "val_rmse",
    mode    = "min",
    dirpath = cfg["train"]["ckpt_dir"],
    filename= "epoch{epoch:02d}-rmse{val_rmse:.3f}",
    save_top_k = cfg["train"]["save_top_k_checkpoints"],
    save_last  = True,
)
lr_cb = LearningRateMonitor(logging_interval="epoch")

# ─── 6) train & validate ──────────────────────────────────────────────────
trainer = pl.Trainer(
    max_epochs        = cfg["train"]["max_epochs"],
    accelerator       = cfg["train"]["accelerator"],
    logger            = logger,
    callbacks         = [ckpt_cb, lr_cb],
    log_every_n_steps = cfg["train"]["log_every_n_steps"],
    enable_progress_bar = cfg["train"]["enable_progress_bar"],
)
trainer.fit(model, datamodule=dm)
trainer.validate(model, datamodule=dm, ckpt_path="best")