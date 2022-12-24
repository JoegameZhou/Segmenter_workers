#  
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Dataset Cityscapes generator."""
import os

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as CType
import mindspore.dataset.vision.c_transforms as c_vision
import numpy as np

from src.data.pipeline.transforms import Resize, RandomCrop, RandomFlip, PhotoMetricDistortion, Normalize, Pad
from src.tools.helpers import to_2tuple
from src.tools.moxing_adapter import sync_data


class CityscapesMS:
    def __init__(self, args, is_train):
        super(CityscapesMS, self).__init__()
        device_num, rank_id = _get_rank_info()
        if args.run_modelarts:
            dataset_path = '/cache/dataset'
            sync_data(args.data_url, dataset_path)
            if "leftImg8bit" not in os.listdir(dataset_path):
                dataset_path = os.path.join(dataset_path, "Cityscapes")
            print(dataset_path)
            args.data_url = dataset_path
            print(f"data_dir: {os.listdir(args.data_url)}")

        if args.encoder in ['vit_large_patch16_384', ]:
            mean = [127.5, 127.5, 127.5]
            std = [127.5, 127.5, 127.5]
        else:
            raise ValueError
        num_parallel_workers = args.num_parallel_workers
        if is_train:
            # create train dataset
            train_dataset = ds.CityscapesDataset(dataset_dir=args.data_url, decode=True, usage="train",
                                                 quality_mode="fine", task="semantic", shuffle=True,
                                                 num_parallel_workers=num_parallel_workers, shard_id=rank_id,
                                                 num_shards=device_num)
            self.train_dataset = create_map_dataset(args=args, dataset=train_dataset, is_train=True, mean=mean, std=std)

        # create val dataset
        val_dataset = ds.CityscapesDataset(dataset_dir=args.data_url, decode=True, usage="val", quality_mode="fine",
                                           task="semantic", shuffle=False, num_parallel_workers=num_parallel_workers)

        self.val_dataset = create_map_dataset(args=args, dataset=val_dataset, is_train=False, mean=mean, std=std)


def create_map_dataset(args, dataset, is_train, mean, std):
    if is_train:
        # BGR2RGB
        dataset = dataset.map(input_columns="task", num_parallel_workers=args.num_parallel_workers,
                              operations=ConvertLabel())
        transform_img_label = [
            Resize(img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
            RandomCrop(crop_size=to_2tuple(args.crop_size), cat_max_ratio=0.75, ignore_label=args.ignore_label),
            RandomFlip(prob=0.5, direction='horizontal'),
            PhotoMetricDistortion(brightness_delta=32, contrast_range=(0.5, 1.5),
                                  saturation_range=(0.5, 1.5), hue_delta=18),
            Normalize(mean=mean, std=std, to_rgb=False),
            Pad(size=to_2tuple(args.crop_size), pad_val=0, seg_pad_val=255),
        ]
        dataset = dataset.map(input_columns=['image', "task"], num_parallel_workers=args.num_parallel_workers,
                              operations=transform_img_label)
        dataset = dataset.map(input_columns='image', num_parallel_workers=args.num_parallel_workers,
                              operations=c_vision.HWC2CHW())

        transform_label = CType.TypeCast(mstype.int32)
        dataset = dataset.map(input_columns="task", num_parallel_workers=args.num_parallel_workers,
                              operations=transform_label)
        dataset = dataset.batch(args.train_batch_size, drop_remainder=True,
                                num_parallel_workers=args.num_parallel_workers)
    else:
        # for val dataset, follow cast bgr to rgb(0-255)
        transform_img = [c_vision.Normalize(mean=mean, std=std), c_vision.HWC2CHW(),
                         SlidingWindow(window_size=args.window_size, window_stride=args.window_stride)]
        transform_label = [ConvertLabel(), CType.TypeCast(mstype.int32)]
        dataset = dataset.map(input_columns="image", num_parallel_workers=args.num_parallel_workers,
                              operations=transform_img)
        dataset = dataset.map(input_columns="task", num_parallel_workers=args.num_parallel_workers,
                              operations=transform_label)
        assert args.eval_batch_size == 1
        dataset = dataset.batch(args.eval_batch_size, drop_remainder=False,
                                num_parallel_workers=args.num_parallel_workers)
    return dataset


class SlidingWindow:
    def __init__(self, window_size, window_stride):
        self.window_size = window_size
        self.window_stride = window_stride

    def __call__(self, image):
        windows = self.sliding_window(image, self.window_size, self.window_stride)
        return windows

    @staticmethod
    def sliding_window(image, window_size, window_stride):
        C, H, W = image.shape
        ws = window_size
        windows = []
        h_anchors = np.arange(0, H, window_stride)
        w_anchors = np.arange(0, W, window_stride)
        h_anchors = [h for h in h_anchors if h < H - ws] + [H - ws]
        w_anchors = [w for w in w_anchors if w < W - ws] + [W - ws]
        for ha in h_anchors:
            for wa in w_anchors:
                window = image[:, int(ha): int(ha + ws), int(wa): int(wa + ws)]
                windows.append(window)
        windows = np.stack(windows, axis=0)
        return windows


class ConvertLabel:
    def __init__(self, ignore_label=255):
        self.ignore_label = ignore_label
        self.label_mapping = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label,
                              10: ignore_label, 11: 2, 12: 3,
                              13: 4, 14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5, 18: ignore_label,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15,
                              29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}

    def __call__(self, label):
        label = np.mean(label, axis=-1)
        temp = label.copy()
        for k, v in self.label_mapping.items():
            label[temp == k] = v
        return label


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
