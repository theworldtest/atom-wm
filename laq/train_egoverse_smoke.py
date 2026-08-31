from laq_model import LAQTrainer
from laq_model import LatentActionQuantization

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
    folder="/mnt/workspace/szeluresearch/wms/egoverse_smoke",
    offsets=30,

    batch_size=1,
    grad_accum_every=1,

    train_on_images=False,
    use_ema=False,

    num_train_steps=20,

    results_folder="/mnt/workspace/szeluresearch/wms/egoverse_laq_smoke_results",

    lr=1e-4,

    save_model_every=10,
    save_results_every=10,
)

trainer.train()
