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
from mindspore import nn
from mindspore.nn.metrics import Metric


class MIOU(Metric):
    def __init__(self, num_classes, anchors, ignore_label=255):
        super(MIOU, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.confusion_matrix = nn.ConfusionMatrix(num_classes=num_classes, normalize='no_norm', threshold=0.5)
        self.iou = np.zeros(num_classes)

    def clear(self):
        self.confusion_matrix.clear()
        self.iou = np.zeros(self.num_classes)

    def eval(self):
        confusion_matrix = self.confusion_matrix.eval()
        for index in range(self.num_classes):
            area_intersect = confusion_matrix[index, index]
            area_pred_label = np.sum(confusion_matrix[:, index])
            area_label = np.sum(confusion_matrix[index, :])
            area_union = area_pred_label + area_label - area_intersect
            self.iou[index] = area_intersect / area_union
        miou = np.nanmean(self.iou)
        return miou

    def postprocess(self, im_windows, H, W):
        window_size = im_windows.shape[-1]
        logit = np.zeros((self.num_classes, H, W))
        count = np.zeros((1, H, W))
        for window, (ha, wa) in zip(im_windows, self.anchors):
            logit[:, int(ha): int(ha + window_size), int(wa): int(wa + window_size)] += window
            count[:, int(ha): int(ha + window_size), int(wa): int(wa + window_size)] += 1
        logit = logit / count
        logit = np.argmax(logit, axis=0)
        return logit

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError("For 'ConfusionMatrix.update', it needs 2 inputs (predicted value, true value), "
                             "but got {}.".format(len(inputs)))
        H, W = inputs[1].shape[1:]
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1]).reshape(-1)
        mask = y != self.ignore_label
        y_pred_postprocess = self.postprocess(y_pred, H=H, W=W).reshape(-1)
        y = y[mask].astype(np.int)
        y_pred_postprocess = y_pred_postprocess[mask].astype(np.int)
        self.confusion_matrix.update(y_pred_postprocess, y)
