import mxnet as mx
import numpy as np


def get_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss', 'rcnn_cls_prob', 'rcnn_bbox_loss', 'rcnn_label', 'mask_prob', 'mask_target', 'mask_weight']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        label = preds[self.pred.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        label = preds[self.pred.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')
        cls = pred[np.arange(label.shape[0]), label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()
        bbox_weight = labels[self.label.index('rpn_bbox_weight')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        label = preds[self.pred.index('rcnn_label')].asnumpy()

        # calculate num_inst
        keep_inds = np.where(label != 0)[0]
        num_inst = len(keep_inds)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst
        

class MaskLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MaskLogLossMetric, self).__init__('MaskLogLoss')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        # reshape and concat
        label = preds[self.pred.index('rcnn_label')].asnumpy().reshape((-1,)).astype('Int32')
        num_classes = preds[self.pred.index('mask_target')].shape[1]
        mask_target = preds[self.pred.index('mask_target')].asnumpy().reshape((-1, num_classes, 14, 14))
        mask_weight = preds[self.pred.index('mask_weight')].asnumpy().reshape((-1, num_classes, 1,1))
        mask_prob = preds[self.pred.index('mask_prob')].asnumpy()  # (n_rois, c, h, w)

        real_inds   = np.where(label != -1)[0]
        n_rois      = real_inds.shape[0]
        mask_prob   = mask_prob[real_inds, label[real_inds]]
        mask_target = mask_target[real_inds, label[real_inds]]
        mask_weight = mask_weight[real_inds, label[real_inds]]
        l = mask_weight*mask_target * np.log(mask_prob + 1e-14) + mask_weight * (1 - mask_target) * np.log(1 - mask_prob + 1e-14)
        self.sum_metric += -np.sum(l)
        self.num_inst += mask_prob.shape[-1] * mask_prob.shape[-2] * n_rois

class MaskAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MaskAccMetric, self).__init__('MaskACC')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        # reshape and concat
        label = preds[self.pred.index('rcnn_label')].asnumpy().reshape((-1,)).astype('Int32')
        num_classes = preds[self.pred.index('mask_target')].shape[1]
        mask_target = preds[self.pred.index('mask_target')].asnumpy().reshape((-1, num_classes, 14, 14))
        mask_weight = preds[self.pred.index('mask_weight')].asnumpy().reshape((-1, num_classes, 1,1))
        mask_prob = preds[self.pred.index('mask_prob')].asnumpy()  # (n_rois, c, h, w)

        real_inds = np.where(label != -1)[0]
        n_rois = real_inds.shape[0]
        mask_prob   = mask_prob[real_inds, label[real_inds]]
        mask_target = mask_target[real_inds, label[real_inds]]
        mask_weight = mask_weight[real_inds, label[real_inds]]
        idx = np.where(np.logical_and(mask_prob > 0.5, mask_weight == 1))
        mask_pred = np.zeros_like(mask_prob)
        mask_pred[idx] = 1
        self.sum_metric += np.sum(mask_target == mask_pred)
        self.num_inst += mask_prob.shape[-1] * mask_prob.shape[-2] * n_rois
