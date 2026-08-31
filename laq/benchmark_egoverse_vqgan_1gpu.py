#!/usr/bin/env python3
import os
import time
import argparse
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except RuntimeError:
    pass
import tensorflow_datasets as tfds
import albumentations as A
import jax
from latent_pretraining.vqgan import VQGAN


def discover_dataset(data_root, dataset_filter):
    root = Path(data_root)
    matches = []
    for info in sorted(root.glob('*/*/*/dataset_info.json')):
        dataset_dir = info.parent
        key = dataset_dir.relative_to(root).as_posix()
        if dataset_filter in key:
            matches.append((key, dataset_dir))
    if not matches:
        raise RuntimeError(f'No dataset matching {dataset_filter!r} under {data_root}')
    return matches[0]


def preprocess_batch(images, preprocessor):
    processed = np.asarray([
        preprocessor(image=img.astype(np.uint8))['image'] for img in images
    ])
    return (processed / 127.5 - 1.0).astype(np.float32)


def encode(vqgan, batch):
    return jax.device_get(vqgan.encode(batch))[1].astype(np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='/mnt/data/RLDS/EgoVerse_full')
    p.add_argument('--dataset-filter', default='mecka_bimanual_front_1')
    p.add_argument('--vqgan-checkpoint', default='/mnt/data/laixianxin/LAPA/lapa_checkpoints/vqgan')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-images', type=int, default=20000)
    args = p.parse_args()

    print('=' * 80)
    print('JAX devices:', jax.devices())
    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError(f'JAX is not using GPU: {jax.devices()}')

    ckpt = Path(args.vqgan_checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f'Missing VQGAN checkpoint: {ckpt}')

    dataset_key, dataset_dir = discover_dataset(args.data_root, args.dataset_filter)
    print('dataset:', dataset_key)
    print('vqgan:', ckpt)
    print('batch_size:', args.batch_size)
    print('measured_images:', args.max_images)

    preprocessor = A.Compose([
        A.LongestMaxSize(max_size=256),
        A.Resize(256, 256),
    ])

    t0 = time.perf_counter()
    vqgan = VQGAN(str(ckpt), replicate=False)
    print(f'VQGAN loaded in {time.perf_counter() - t0:.2f}s', flush=True)

    builder = tfds.builder_from_directory(str(dataset_dir))
    ds = builder.as_dataset(split='train', shuffle_files=False)

    image_buf = []
    compiled = False
    n_measured = 0
    measured_t0 = None
    sample_tokens = None

    for episode in ds:
        for step in episode['steps']:
            raw = step['observation']['images']['front_1'].numpy()
            image_buf.append(raw)

            if len(image_buf) >= args.batch_size:
                x = preprocess_batch(image_buf, preprocessor)
                ids = encode(vqgan, x)

                if ids.ndim != 3 or ids.shape[1:] != (16, 16):
                    raise RuntimeError(f'Expected [B,16,16], got {ids.shape}')
                if ids.min() < 0 or ids.max() >= 8192:
                    raise RuntimeError(f'Bad token range: {ids.min()}..{ids.max()}')

                if sample_tokens is None:
                    sample_tokens = ids[0].reshape(-1).tolist()

                if not compiled:
                    compiled = True
                    measured_t0 = time.perf_counter()
                    print('WARMUP/JAX COMPILE COMPLETE; starting measurement ...', flush=True)
                else:
                    n_measured += len(image_buf)

                image_buf = []

                if n_measured >= args.max_images:
                    break
        if n_measured >= args.max_images:
            break

    if not compiled or measured_t0 is None:
        raise RuntimeError('No complete batch was encoded')

    elapsed = time.perf_counter() - measured_t0
    fps = n_measured / elapsed if elapsed > 0 else 0.0

    print('=' * 80)
    print('VQGAN 1-GPU SMOKE/BENCHMARK: PASS')
    print('measured_images:', n_measured)
    print('elapsed_sec:', f'{elapsed:.3f}')
    print('end_to_end_images_per_sec:', f'{fps:.2f}')
    print('vision_tokens_per_image:', len(sample_tokens))
    print('sample_token_min:', min(sample_tokens))
    print('sample_token_max:', max(sample_tokens))
    print('sample_first_16_tokens:', sample_tokens[:16])
    if fps > 0:
        one = 116_584_930 / fps / 3600
        print('estimated_full_1gpu_hours:', f'{one:.2f}')
        print('estimated_full_16gpu_ideal_hours:', f'{one / 16:.2f}')
    print('=' * 80)


if __name__ == '__main__':
    main()
