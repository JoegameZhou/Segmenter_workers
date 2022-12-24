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
""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2021 Ross Wightman
"""
import time
from functools import partial

import numpy as np
from mindspore import context
from mindspore import dtype as mstype
from mindspore import nn, ops, Parameter, Tensor

from src.models.layers import DropPath1D, Identity
from src.models.layers import trunc_normal_, lecun_normal_, zeros_, ones_, constant_
from src.tools.helpers import to_2tuple


def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )

    gs_old_h, gs_old_w = grid_old_shape
    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).transpose(0, 3, 1, 2)
    posemb_grid = ops.ResizeBilinear(size=(gs_h, gs_w))(posemb_grid)
    posemb_grid = posemb_grid.transpose(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = ops.Concat(axis=1)((posemb_tok, posemb_grid))
    return posemb


class Mlp(nn.Cell):
    """MLP Cell"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Cell):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer else Identity()

    def construct(self, x):
        x = self.proj(x)
        B, C, _, _ = x.shape
        if self.flatten:
            x = x.reshape(B, C, -1).transpose(0, 2, 1)
        x = self.norm(x)
        return x


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # get pair-wise relative position index for each token inside the window
        self.qkv = nn.Dense(in_channels=dim, out_channels=dim * 3, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout(keep_prob=1 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.unstack = ops.Unstack(axis=0)

    def construct(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = self.unstack(qkv)  # make torchscript happy (cannot use tensor as tuple)

        attn = ops.BatchMatMul(transpose_b=True)(q, k) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        attn = ops.Cast()(attn, mstype.float16)
        v = ops.Cast()(v, mstype.float16)
        x = ops.Reshape()(ops.Transpose()(ops.BatchMatMul()(attn, v), (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Cell):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Cell): patch embedding layer
            norm_layer: (nn.Cell): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.distilled = int(distilled)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(Tensor(np.zeros([1, 1, embed_dim]), dtype=mstype.float32))
        self.dist_token = Parameter(Tensor(np.zeros([1, 1, embed_dim]), dtype=mstype.float32)) if distilled else None
        self.pos_embed = Parameter(
            Tensor(np.zeros([1, num_patches + self.num_tokens, embed_dim]), dtype=mstype.float32))
        self.pos_drop = nn.Dropout(keep_prob=1 - drop_rate)

        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.SequentialCell(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer((embed_dim,))

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.SequentialCell([
                nn.Dense(embed_dim, representation_size),
                nn.Tanh()
            ])
        else:
            self.pre_logits = Identity()

        # Classifier head(s)
        # self.head = nn.Dense(self.num_features, num_classes) if num_classes > 0 else Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Dense(self.embed_dim, self.num_classes) if num_classes > 0 else Identity()

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        for name, cell in self.cells_and_names():
            _init_vit_weights(cell, name)

    def construct_features(self, x, return_features):
        B, _, H, W = x.shape
        PS = self.patch_size

        x = self.patch_embed(x)  # forward_features tensor(-0.8333) finished
        cls_tokens = ops.Tile()(self.cls_token, (x.shape[0], 1, 1))
        if self.dist_token is None:
            x = ops.Concat(1)((cls_tokens, x))
        else:
            dist_token = ops.Tile()(self.dist_token, (x.shape[0], 1, 1))
            x = ops.Concat(1)((cls_tokens, dist_token, x))

        pos_embed = self.pos_embed
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                self.num_tokens,
            )
        x = self.pos_drop(x + pos_embed)  # tensor(0.7012) finished
        x = self.blocks(x)
        x = self.norm(x)
        if return_features:
            return x
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        return x[:, 0], x[:, 1]

    def construct(self, x, return_features=False):
        x = self.construct_features(x, return_features)
        if return_features:
            return x
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training:
                # during inference, return the average of both classifier predictions
                return x, x_dist
            return (x + x_dist) / 2
        # x = self.head(x)
        return x


def _init_vit_weights(cell: nn.Cell, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (cell name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(cell, nn.Dense):
        if name.startswith('head'):
            zeros_(cell.weight)
            # trunc_normal_(cell.weight, std=.02)
            constant_(cell.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(cell.weight)
            zeros_(cell.bias)
            trunc_normal_(cell.weight, std=.02)
            if cell.bias is not None:
                zeros_(cell.bias)
    elif isinstance(cell, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(cell.weight)
        if cell.bias is not None:
            zeros_(cell.bias)
    elif isinstance(cell, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        ones_(cell.gamma)
        zeros_(cell.beta)


def vit_large_patch16_384(patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                          img_size=768, drop_path_rate=0., drop_rate=0.):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        in_chans=3,
        img_size=to_2tuple(img_size),
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
    )


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    model = vit_large_patch16_384(img_size=384)
    model.to_float(mstype.float16)
    data = Tensor(np.random.randn(2, 3, 768, 768), dtype=mstype.float32)
    for i in range(1):
        start = time.time()
        print(time.time() - start, i, model(data, return_features=False).shape)
    params = 0.
    num = 0
    for name, param in model.parameters_and_names():
        # params += np.prod(param.shape)
        num += 1
        print(name, param.shape)
    print(params, num)  # 304715752.0 296
