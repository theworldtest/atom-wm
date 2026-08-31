from laq_model import LAQTrainer
from laq_model import LatentActionQuantization

import os
import torch

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

laq = LatentActionQuantization(
    dim=1024,
    quant_dim=32,
    codebook_size=8,
    image_size=256,
    patch_size=32,
    spatial_depth=8,
    temporal_depth=8,
    dim_head=64,
    heads=16,
    code_seq_len=4,
).cuda()

trainer = LAQTrainer(
    laq,
    folder="/mnt/data/wudi/ELUBrain/EgoVerseData",
    offsets=30,

    batch_size=8,          # 每卡8，8卡 global batch=64
    grad_accum_every=1,

    train_on_images=False,
    use_ema=False,

    num_train_steps=100005,

    results_folder="/mnt/data/szeluresearch/wms/egoverse_laq_full",

    lr=1e-4,

    save_model_every=5000,
    save_results_every=5000,
)

trainer.train()
