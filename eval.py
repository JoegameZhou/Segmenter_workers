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
"""train"""

from mindspore import Model, context
from mindspore.common import set_seed

from src.args import args
from src.models.segmenter import Segmenter
from src.tools.amp import cast_amp
from src.tools.criterion import NetWithLoss
from src.tools.criterion import WithEvalCell
from src.tools.get_misc import get_anchors
from src.tools.get_misc import get_encoder, get_decoder, get_dataset, get_criterion, get_train_one_step, get_pretrained
from src.tools.miou_metric import MIOU
from src.tools.optim import get_optimizer
from src.tools.prepare_misc import prepare_context


def main(args):
    # prepare context
    set_seed(args.seed)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    args.rank = prepare_context(args)

    # prepare dataset
    data = get_dataset(args)
    batch_num = data.train_dataset.get_dataset_size()

    # prepare loss function
    criterion = get_criterion(args)

    # prepare model
    encoder = get_encoder(args)
    decoder = get_decoder(args)
    segmenter = Segmenter(encoder=encoder, decoder=decoder, num_classes=args.num_classes)

    # precare mix precision
    cast_amp(args=args, net=segmenter)

    segmenter_with_loss = NetWithLoss(segmenter, criterion)

    if args.pretrained:
        get_pretrained(args=args, model=segmenter_with_loss)

    # prepare optimizer
    optimizer = get_optimizer(args=args, model=segmenter, batch_num=batch_num)

    # prepare net_with_loss
    net_with_loss = get_train_one_step(args, segmenter_with_loss, optimizer)

    # if resume
    if args.resume:
        args.pretrained = args.resume
        get_pretrained(args=args, model=net_with_loss)

    # prepare metric
    anchors = get_anchors(im_shape=(3, args.infer_image_size, int(args.infer_image_size * 2)),
                          window_size=args.window_size, window_stride=args.window_stride)
    eval_metrics = {'miou': MIOU(num_classes=args.num_classes, anchors=anchors, ignore_label=args.ignore_label)}
    eval_network = WithEvalCell(segmenter)

    # get Model
    model = Model(net_with_loss, metrics=eval_metrics, eval_network=eval_network)

    # prepare callbacks
    args.ckpt_save_dir = "./ckpt_" + str(args.rank)
    if args.run_modelarts:
        args.ckpt_save_dir = "/cache/ckpt_" + str(args.rank)

    print(f"=> begin eval")
    # start to eval
    results = model.eval(data.val_dataset)
    print(f"=> eval results: {results}")
    print(f"=> eval success")


if __name__ == '__main__':
    main(args=args)
