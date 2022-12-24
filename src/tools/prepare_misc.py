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

from mindspore import context
from mindspore.communication import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.tools.eval_callback import EvaluateCallBack


def prepare_context(args):
    """prepare context"""
    rank = 0
    # set context and device
    device_target = args.device_target
    device_num = int(os.environ.get("DEVICE_NUM", 1))
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if device_target == "Ascend":
        if device_num > 1:
            context.set_context(device_id=int(os.environ["DEVICE_ID"]))
            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id)
    elif device_target == "GPU":
        if device_num > 1:
            init(backend_name='nccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id)
    else:
        raise ValueError("Unsupported platform.")

    return rank


def prepare_callbacks(args, data, model):
    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=data.train_dataset.get_dataset_size(),
                                 keep_checkpoint_max=args.save_every)
    ckpoint_cb = ModelCheckpoint(prefix=f"{args.decoder}_{args.encoder}_{args.rank}", directory=args.ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor()
    callbacks = [time_cb, ckpoint_cb, loss_cb]

    eval_cb = EvaluateCallBack(model=model, eval_dataset=data.val_dataset, src_url=args.ckpt_save_dir,
                               train_url=os.path.join(args.train_url, "ckpt_" + str(args.rank)), args=args,
                               total_epochs=args.epochs - args.start_epoch, save_freq=args.save_every)

    callbacks.append(eval_cb)

    return callbacks
