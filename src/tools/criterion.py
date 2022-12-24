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
"""functions of criterion"""
from mindspore import Tensor, nn, ops
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import operations as P


class NetWithLoss(nn.Cell):
    def __init__(self, model, criterion):
        super(NetWithLoss, self).__init__()
        self.model = model
        self.criterion = criterion

    def construct(self, data, label):
        predict = self.model(data)
        loss = self.criterion(predict, label)
        return loss


class CrossEntropyWithLogits(LossBase):
    """
    Cross-entropy loss function for semantic segmentation,
        and different classes have the same weight.
    """

    def __init__(self, num_classes=19, ignore_label=255):
        super(CrossEntropyWithLogits, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.argmax = P.Argmax(output_type=mstype.int32)
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        """Loss construction."""
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_classes))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_classes, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))

        return loss


class WithEvalCell(nn.Cell):
    def __init__(self, network):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self._network = network

    def construct(self, data, label):
        outputs = self._network(data[0])
        label = ops.Cast()(label, mstype.float32)
        outputs = ops.cast(outputs, mstype.float32)
        return outputs, label