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
from mindspore import nn, ops

from src.engine.eval_utils import padding, unpadding
from src.models.decoder import MaskTransformer
from src.models.encoder import vit_large_patch16_384


class Segmenter(nn.Cell):
    def __init__(
            self,
            encoder,
            decoder,
            num_classes,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def construct(self, im):
        H_ori, W_ori = im.shape[2], im.shape[3]
        im = padding(im, self.patch_size)
        H, W = im.shape[2], im.shape[3]

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))
        masks = ops.ResizeBilinear(size=(H, W))(masks)
        masks = unpadding(masks, (H_ori, W_ori))

        return masks


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    encoder = vit_large_patch16_384(img_size=768)
    decoder = MaskTransformer(
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
    segmenter = Segmenter(encoder=encoder, decoder=decoder, num_classes=19)
    params = 0.
    num = 0
    for name, param in segmenter.parameters_and_names():
        # params += np.prod(param.shape)
        num += 1
        params += np.prod(param.shape)
        print(name, param.shape)
    print(params, num)

    # data = Tensor(np.random.randn(1, 3, 768, 768), dtype=mstype.float32)
    # print(segmenter(data).shape)
