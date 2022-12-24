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
import os

import numpy as np
from mindspore import nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src import data
from src.engine import TrainOneStepWithGlobalNormClip
from src.models.decoder import MaskTransformer
from src.models.encoder import vit_large_patch16_384
from src.tools.criterion import CrossEntropyWithLogits
from src.tools.moxing_adapter import sync_data


def get_dataset(args, is_train=True):
    """"Get model according to args.set"""
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args, is_train)

    return dataset


def get_encoder(args):
    if args.encoder == "vit_large_patch16_384":
        encoder = vit_large_patch16_384(
            img_size=args.train_image_size,
            drop_path_rate=args.encoder_drop_path_rate,
            drop_rate=args.encoder_dropout
        )
    else:
        raise ValueError
    return encoder


def get_decoder(args):
    if args.decoder == "mask_transformer":
        decoder = MaskTransformer(
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            d_encoder=args.d_encoder,
            n_layers=args.n_layers,
            n_heads=args.d_model // args.head_dim,
            d_model=args.d_model,
            d_ff=args.d_ff,
            drop_path_rate=args.decoder_drop_path_rate,
            dropout=args.decoder_dropout
        )
    else:
        raise ValueError
    return decoder


def get_train_one_step(args, net_with_loss, optimizer):
    """get_train_one_step cell"""
    if args.is_dynamic_loss_scale:
        print(f"=> Using DynamicLossScaleUpdateCell")
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 10, scale_factor=2,
                                                                    scale_window=2000)
    else:
        print(f"=> Using FixedLossScaleUpdateCell, loss_scale_value:{args.loss_scale}")
        scale_sense = nn.wrap.FixedLossScaleUpdateCell(loss_scale_value=args.loss_scale)
    net_with_loss = TrainOneStepWithGlobalNormClip(net_with_loss, optimizer, global_norm=args.clip_global_norm_value,
                                                   scale_sense=scale_sense)

    return net_with_loss


def get_criterion(args):
    """Get loss function with ignore index"""
    assert args.ignore_label == 255
    criterion = CrossEntropyWithLogits(num_classes=args.num_classes, ignore_label=args.ignore_label)
    return criterion


def get_pretrained(args, model):
    """"Load pretrained weights if args.pretrained is given"""
    if args.run_modelarts:
        print('Syncing data.')
        local_data_path = '/cache/weight/model.ckpt'
        sync_data(args.pretrained, local_data_path, threads=128)
        print("=> loading pretrained weights from '{}'".format(local_data_path))
        param_dict = load_checkpoint(local_data_path)
        load_param_into_net(model, param_dict)
    elif os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        param_dict = load_checkpoint(args.pretrained)
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != args.num_classes:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        load_param_into_net(model, param_dict)
    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))


def get_anchors(im_shape, window_size, window_stride):
    C, H, W = im_shape
    ws = window_size

    anchors = []
    h_anchors = np.arange(0, H, window_stride)
    w_anchors = np.arange(0, W, window_stride)
    h_anchors = [h for h in h_anchors if h < H - ws] + [H - ws]
    w_anchors = [w for w in w_anchors if w < W - ws] + [W - ws]
    for ha in h_anchors:
        for wa in w_anchors:
            anchors.append((ha, wa))
    return anchors