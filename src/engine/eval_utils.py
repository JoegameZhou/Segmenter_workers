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
import math

import numpy as np
from mindspore import dtype as mstype
from mindspore import ops, Tensor


def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).transpose(0, 3, 1, 2)
    posemb_grid = ops.ResizeBilinear(size=(gs_h, gs_w))(posemb_grid)
    posemb_grid = posemb_grid.transpose(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = ops.Concat(1)((posemb_tok, posemb_grid))
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    num_extra_tokens = 1 + ("dist_token" in state_dict.keys())
    patch_size = model.patch_size
    image_size = model.patch_embed.image_size
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                None,
                (image_size[0] // patch_size, image_size[1] // patch_size),
                num_extra_tokens,
            )
        out_dict[k] = v
    return out_dict


def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.shape[2], im.shape[3]
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = ops.Pad((0, 0,), (0, 0), (0, pad_w), (0, pad_h))(im_padded)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.shape[2], y.shape[3]
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


def resize(im, smaller_size):
    h, w = im.shape[2:]
    if h < w:
        ratio = w / h
        h_res, w_res = smaller_size, ratio * smaller_size
    else:
        ratio = h / w
        h_res, w_res = ratio * smaller_size, smaller_size
    if min(h, w) < smaller_size:
        im_res = ops.ResizeBilinear(size=(int(h_res), int(w_res)))(im)
    else:
        im_res = im
    return im_res


def sliding_window(im, window_size, window_stride):
    B, C, H, W = im.shape
    ws = window_size

    windows = {"crop": [], "anchors": []}
    h_anchors = np.arange(0, H, window_stride)
    w_anchors = np.arange(0, W, window_stride)
    h_anchors = [h for h in h_anchors if h < H - ws] + [H - ws]
    w_anchors = [w for w in w_anchors if w < W - ws] + [W - ws]
    for ha in h_anchors:
        for wa in w_anchors:
            window = im[:, :, int(ha): int(ha + ws), int(wa): int(wa + ws)]
            windows["crop"].append(window)
            windows["anchors"].append((ha, wa))
    windows["shape"] = (H, W)
    windows['flip'] = False
    return windows


def merge_windows(windows, window_size, ori_shape):
    ws = window_size
    im_windows = windows["seg_maps"]
    anchors = windows["anchors"]
    C = im_windows[0].shape[0]
    H, W = windows["shape"]
    flip = windows["flip"]

    logit = np.zeros((C, H, W))
    count = np.zeros((1, H, W))
    for window, (ha, wa) in zip(im_windows, anchors):
        logit[:, int(ha): int(ha + ws), int(wa): int(wa + ws)] += window.asnumpy()
        count[:, int(ha): int(ha + ws), int(wa): int(wa + ws)] += 1
    logit = logit / count
    logit = Tensor(np.expand_dims(logit, 0), mstype.float32)
    logit = ops.ResizeBilinear(size=ori_shape)(logit)[0]
    if flip:
        logit = np.flip(logit, (2,))
    result = ops.Softmax(axis=0)(logit)
    return result
