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
"""global args for Transformer in Transformer(TNT)"""
import argparse
import ast
import os
import sys

import yaml

from src.configs import parser as _parser

args = None


def parse_arguments():
    """parse_arguments"""
    global args
    parser = argparse.ArgumentParser(description="MindSpore Cityscapes Training")

    # general config
    parser.add_argument("--train_batch_size", default=1, type=int, metavar="N",
                        help="mini-batch size (default: 256), this is the total "
                             "batch size of all Devices on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--eval_batch_size", default=1, type=int, metavar="N",
                        help="mini-batch size (default: 256), this is the total "
                             "batch size of all Devices on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--config", help="Config file to use (see configs dir)", default=None)
    parser.add_argument("--epochs", default=126, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--loss_scale", default=1024, type=int, help="loss_scale")
    parser.add_argument("--save_every", default=2, type=int, help="save every ___ epochs(default:2)")
    parser.add_argument("--seed", default=0, type=int, help="seed for initializing training. ")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N",
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("--is_dynamic_loss_scale", default=1, type=int, help="is_dynamic_loss_scale ")
    parser.add_argument("--keep_checkpoint_max", default=20, type=int, help="keep checkpoint max num")
    parser.add_argument("-j", "--num_parallel_workers", default=20, type=int, metavar="N",
                        help="number of data loading workers (default: 20)")

    # model related
    parser.add_argument("--amp_level", default="O2", choices=["O0", "O1", "O2", "O3"], help="AMP Level")
    parser.add_argument("--encoder", default="vit_large_patch16_384", help="encoder architecture")
    parser.add_argument("--decoder", default="masktransformer", help="decoder architecture")
    parser.add_argument("--pretrained", dest="pretrained", default=None, type=str, help="use pre-trained model")
    parser.add_argument("--resume", dest="resume", default=None, type=str, help="use resume model")
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--patch_size", default=16, type=int)

    parser.add_argument("--d_model", default=1024, type=int)
    parser.add_argument("--d_ff", default=4096, type=int)
    parser.add_argument("--encoder_drop_path_rate", default=0., type=float)
    parser.add_argument("--encoder_dropout", default=0.0, type=float)
    parser.add_argument("--dncoder_drop_path_rate", default=0., type=float)
    parser.add_argument("--dncoder_dropout", default=0.0, type=float)
    parser.add_argument("--d_encoder", default=1024, type=int)
    parser.add_argument("--head_dim", default=64, type=int)
    parser.add_argument("--n_layers", default=1, type=int)
    parser.add_argument("--base_size", default=2048, type=int)
    parser.add_argument("--scale_factor", default=16, type=int)

    # optimizer related
    parser.add_argument("--base_lr", default=5e-4, type=float, help="initial lr")
    parser.add_argument("--min_lr", default=1e-5, type=float, help="min lr")
    parser.add_argument("--beta", default=[0.9, 0.999], type=lambda x: [float(a) for a in x.split(",")],
                        help="beta for optimizer")
    parser.add_argument("--lr_scheduler", default="cosine_annealing", help="schedule for the learning rate.")
    parser.add_argument("--lr_adjust", default=30, type=float, help="interval to drop lr")
    parser.add_argument("--lr_gamma", default=0.97, type=float, help="multistep multiplier")
    parser.add_argument("--poly_power", default=0.9, type=float, help="poly power")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--optimizer", help="Which optimizer to use", default="momentum")
    parser.add_argument("--wd", "--weight_decay", default=0.0, type=float, metavar="W",
                        help="weight decay (default: 0.05)", dest="weight_decay")
    parser.add_argument("--warmup_length", default=0, type=int, help="number of warmup iterations")
    parser.add_argument("--warmup_lr", default=5e-7, type=float, help="warm up learning rate")
    parser.add_argument("--eval_while_train", type=ast.literal_eval, default=False, help="whether eval while True")

    # data related
    parser.add_argument("--set", help="name of dataset", type=str, default="Cityscapes")
    parser.add_argument("--num_classes", default=19, type=int)
    parser.add_argument("--infer_image_size", default=1024, help="image Size.", type=int)
    parser.add_argument("--train_image_size", default=768, help="image Size.", type=int)
    parser.add_argument("--ignore_label", default=255, type=int)
    parser.add_argument("--crop_size", default=768, type=int)
    parser.add_argument("--window_size", default=768, type=int)
    parser.add_argument("--window_stride", default=512, type=int)

    # hardware related
    parser.add_argument("--device_id", default=0, type=int, help="device id")
    parser.add_argument("--device_num", default=1, type=int, help="device num")
    parser.add_argument("--device_target", default="Ascend", choices=["GPU", "Ascend"], type=str)

    # modelarts related
    parser.add_argument('--data_url', default="./data", help='location of data.')
    parser.add_argument('--train_url', default="./", help='location of training outputs.')
    parser.add_argument("--run_modelarts", type=ast.literal_eval, default=False, help="whether run on modelarts")

    # export related
    parser.add_argument("--file_format", type=str, choices=["AIR", "MINDIR"], default="MINDIR", help="file format")

    args = parser.parse_args()

    get_config()


def get_config():
    """get_config"""
    global args
    override_args = _parser.argv_to_vars(sys.argv)
    config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', args.config)
    yaml_txt = open(config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)

    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML configs from {args.config}")

    args.__dict__.update(loaded_yaml)
    print(args)
    os.environ["DEVICE_TARGET"] = args.device_target
    if "DEVICE_NUM" not in os.environ.keys():
        os.environ["DEVICE_NUM"] = str(args.device_num)
    if "RANK_SIZE" not in os.environ.keys():
        os.environ["RANK_SIZE"] = str(args.device_num)


def run_args():
    """run and get args"""
    global args
    if args is None:
        parse_arguments()


run_args()
