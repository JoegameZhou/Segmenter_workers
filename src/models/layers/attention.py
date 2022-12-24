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
from mindspore import nn, ops


class Attention(nn.Cell):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Dense(dim, dim * 3)
        self.unstack = ops.Unstack(axis=0)
        self.attn_drop = nn.Dropout(keep_prob=1 - dropout)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - dropout)
        self.softmax = nn.Softmax(axis=-1)
        self.batch_matmul = ops.BatchMatMul(transpose_b=True)

    def construct(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).transpose(2, 0, 3, 1, 4)
        q, k, v = self.unstack(qkv)

        attn = self.batch_matmul(q, k) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = ops.BatchMatMul()(attn, v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn