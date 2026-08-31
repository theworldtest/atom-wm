from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

from torchvision import transforms as T

import os
import random

# ===== EgoVerse 新增 =====
import io
import numpy as np
import zarr
# ========================


def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def pair(val):
    return val if isinstance(val, tuple) else (val, val)


# ===== EgoVerse 新增 =====
def unwrap_bytes(x):
    """
    EgoVerse 的 images.front_1[t] 会返回多层 0-D ndarray，
    一直 .item()，直到得到真正的 JPEG bytes。
    """
    while isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()

    if isinstance(x, np.bytes_):
        x = bytes(x)

    return x
# ========================


class ImageVideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        offset=5,
    ):
        super().__init__()

        self.folder = folder
        #self.folder_list = os.listdir(folder)
        #过滤掉没有 zarr.json 的文件夹
        self.folder_list = [
            name
            for name in os.listdir(folder)
            if (
                os.path.isfile(os.path.join(folder, name, "zarr.json"))
                and
                os.path.isfile(
                    os.path.join(folder, name, "images.front_1", "zarr.json")
                )
            )
        ]

        bad_episodes = {
            "2026-04-14-04-38-04-000000",
            "69256d8d0e0793bab697903f",
            "693cdd489d2cd4dd285bd78e",
        }

        self.folder_list = [
            name for name in self.folder_list
            if name not in bad_episodes
        ]

        print(f"EgoVerse valid episodes: {len(self.folder_list)}")
        
        self.image_size = image_size

        self.offset = offset

        # ===== 保持官方原样 =====
        self.transform = T.Compose([
            T.Lambda(
                lambda img:
                img.convert('RGB')
                if img.mode != 'RGB'
                else img
            ),
            T.Resize(image_size),
            T.ToTensor(),
        ])


    def __len__(self):
        return len(self.folder_list)


    def __getitem__(self, index):
        try:
            offset = self.offset

            folder = self.folder_list[index]

            # ==========================================
            # EgoVerse：
            # 原版这里是 os.listdir(folder) 找 JPG
            # 现在改成打开该 episode 的 Zarr
            # ==========================================

            episode_path = os.path.join(
                self.folder,
                folder
            )

            root = zarr.open_group(
                episode_path,
                mode="r"
            )

            rgb = root["images.front_1"]

            # EgoVerse 的 zarr shape 可能比真实有效帧稍长，
            # 优先使用 metadata 中的 total_frames
            total_frames = root.attrs.get(
                "total_frames",
                rgb.shape[0]
            )

            num_frames = min(
                int(total_frames),
                rgb.shape[0]
            )

            # ===== 下面继续保持 LAPA 官方采样逻辑 =====

            first_frame_idx = random.randint(
                0,
                num_frames - 1
            )

            first_frame_idx = min(
                first_frame_idx,
                num_frames - 1
            )

            second_frame_idx = min(
                first_frame_idx + offset,
                num_frames - 1
            )

            # ==========================================
            # 原版：
            #
            # img = Image.open(first_path)
            #
            # EgoVerse：
            # Zarr -> JPEG bytes -> PIL Image
            # ==========================================

            first_bytes = unwrap_bytes(
                rgb[first_frame_idx]
            )

            second_bytes = unwrap_bytes(
                rgb[second_frame_idx]
            )

            img = Image.open(
                io.BytesIO(first_bytes)
            )

            next_img = Image.open(
                io.BytesIO(second_bytes)
            )

            # ===== 以下保持官方原样 =====

            transform_img = self.transform(
                img
            ).unsqueeze(1)

            next_transform_img = self.transform(
                next_img
            ).unsqueeze(1)

            cat_img = torch.cat(
                [
                    transform_img,
                    next_transform_img
                ],
                dim=1
            )

            return cat_img

        except Exception as e:
            print("error", index, e)

            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(
                    random.randint(
                        0,
                        self.__len__() - 1
                    )
                )