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
"""callback function"""
from mindspore.train.callback import Callback


class EvaluateCallBack(Callback):
    """EvaluateCallBack"""

    def __init__(self, model, eval_dataset, src_url, train_url, total_epochs, args, save_freq=50):
        super(EvaluateCallBack, self).__init__()
        self.args = args
        self.model = model
        self.eval_dataset = eval_dataset
        self.src_url = src_url
        self.train_url = train_url
        self.total_epochs = total_epochs
        self.save_freq = save_freq

        # eval config
        self.ignore_label = args.ignore_label
        self.num_classes = args.num_classes
        self.window_size = args.window_size
        self.window_stride = args.window_stride

        self.best_miou = 0
        self.best_epoch = 0

    def epoch_end(self, run_context):
        """
            Test when epoch end, save best model with best.ckpt.
        """
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        if (cb_params.cur_epoch_num > self.total_epochs * 0.9 or int(
                cb_params.cur_epoch_num - 1) % self.save_freq == 0 or cb_params.cur_epoch_num < 10) and self.args.eval_while_train:
            scores = self.model.eval(self.eval_dataset)
            miou = scores['miou']
            if miou > self.best_miou:
                self.best_miou = miou
                self.best_epoch = cur_epoch_num
            log_text1 = 'EPOCH: %d, mIoU: %.4f\n' % (cur_epoch_num, miou)
            log_text2 = 'BEST EPOCH: %s, BEST mIoU: %0.4f\n' % (self.best_epoch, self.best_miou)
            print("==================================================\n",
                  log_text1,
                  log_text2,
                  "==================================================",
                  flush=True)
        if self.args.run_modelarts:
            import moxing as mox
            if cur_epoch_num % self.save_freq == 0:
                mox.file.copy_parallel(src_url=self.src_url, dst_url=self.train_url)
