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
import numpy as np
from mindspore import context
from mindspore import dtype as mstype
from mindspore import nn, Parameter, Tensor, ops

from src.models.layers import Identity, DropPath1D
from src.models.layers.attention import Attention
from src.models.layers.ffn import FeedForward
from src.models.layers.weight_init import zeros_, trunc_normal_, ones_


class Block(nn.Cell):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm((dim,))
        self.norm2 = nn.LayerNorm((dim,))
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath1D(drop_path) if drop_path > 0.0 else Identity()

    def construct(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderLinear(nn.Cell):
    def __init__(self, num_classes, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.head = nn.Dense(self.d_encoder, num_classes)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Dense) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            ones_(m.gamma)
            zeros_(m.beta)

    def apply(self, func):
        for _, cell in self.cells_and_names():
            func(cell)

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        B, N, C = x.shape
        x = x.transpose(0, 2, 1).reshape(B, C, GS, -1)

        return x


class MaskTransformer(nn.Cell):
    def __init__(
            self,
            num_classes,
            patch_size,
            d_encoder,
            n_layers,
            n_heads,
            d_model,
            d_ff,
            drop_path_rate,
            dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x for x in np.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.CellList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = Parameter(Tensor(np.random.randn(1, num_classes, d_model), dtype=mstype.float32))
        self.proj_dec = nn.Dense(d_encoder, d_model)

        self.proj_patch = Parameter(Tensor(self.scale * np.random.randn(d_model, d_model), dtype=mstype.float32))
        self.proj_classes = Parameter(Tensor(self.scale * np.random.randn(d_model, d_model), dtype=mstype.float32))

        self.decoder_norm = nn.LayerNorm((d_model,), epsilon=1e-5)
        self.mask_norm = nn.LayerNorm((num_classes,), epsilon=1e-5)
        self.l2norm = ops.L2Normalize(axis=-1)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Dense) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            ones_(m.gamma)
            zeros_(m.beta)

    def apply(self, func):
        for _, cell in self.cells_and_names():
            func(cell)

    def no_weight_decay(self):
        return {"cls_emb"}

    def construct(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.proj_dec(x)
        cls_emb = ops.Tile()(self.cls_emb, (x.shape[0], 1, 1))
        x = ops.Concat(1)((x, cls_emb))
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.num_classes], x[:, -self.num_classes:]

        # step1 cast + reshape + matmul + cast
        proj_patch = self.proj_patch
        patches = ops.Cast()(patches, mstype.float16)
        proj_patch = ops.Cast()(proj_patch, mstype.float16)
        B, N, C = patches.shape
        patches = ops.MatMul()(patches.reshape(-1, C), proj_patch).reshape(B, N, -1)

        # step2 cast + reshape + matmul + cast
        proj_classes = self.proj_classes
        cls_seg_feat = ops.Cast()(cls_seg_feat, mstype.float16)
        proj_classes = ops.Cast()(proj_classes, mstype.float16)
        B, N, C = cls_seg_feat.shape
        cls_seg_feat = ops.MatMul()(cls_seg_feat.reshape(-1, C), proj_classes).reshape(B, N, -1)

        patches = self.l2norm(patches)
        cls_seg_feat = self.l2norm(cls_seg_feat)

        patches = ops.Cast()(patches, mstype.float16)
        cls_seg_feat = ops.Cast()(cls_seg_feat, mstype.float16)
        masks = ops.BatchMatMul(transpose_b=True)(patches, cls_seg_feat)
        masks = self.mask_norm(masks)
        B, N, C = masks.shape
        masks = masks.transpose(0, 2, 1)
        masks = masks.reshape(B, C, GS, -1)
        return masks


if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    model = MaskTransformer(
        num_classes=19,
        patch_size=16,
        d_encoder=1024,
        n_layers=1,
        n_heads=1024 // 64,
        d_model=1024,
        d_ff=4 * 1024,
        drop_path_rate=0.,
        dropout=0.1
    )
    # params = 0.
    # num = 0
    # for name, param in model.parameters_and_names():
    #     params += np.prod(param.shape)
    #     num += 1
    #     print(name, param.shape)
    # print(params, num)
    data1 = Tensor(np.random.randn(8, 2304, 1024), dtype=mstype.float32)
    H = W = 768
    print(model(data1, (H, W)).shape)
    # print(segmenter(data).shape)
