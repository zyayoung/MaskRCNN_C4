"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
import cv2
import PIL.Image as Image
import math

from symdata.bbox import bbox_overlaps, bbox_transform

def compute_mask_and_label(ex_rois, ex_labels, seg, flipped=False):
    # assert os.path.exists(seg_gt), 'Path does not exist: {}'.format(seg_gt)
    # im = Image.open(seg_gt)
    # pixel = list(im.getdata())
    # pixel = np.array(pixel).reshape([im.size[1], im.size[0]])
    ins_seg = seg[0]
    # print(ins_seg)
    rois = ex_rois[:,1:]
    # print(rois)
    n_rois = ex_rois.shape[0]
    label = ex_labels
    class_id = [0, 24, 25, 26, 27, 28, 31, 32, 33]
    mask_target = np.zeros((n_rois, 28, 28), dtype=np.int8)
    mask_label = np.zeros((n_rois), dtype=np.int8)
    # print(rois)
    for n in range(n_rois):
        target = ins_seg[int(rois[n, 1]): int(rois[n, 3]), int(rois[n, 0]): int(rois[n, 2])]
        # print(target.shape)
        ids = np.unique(target)
        ins_id = 0
        max_count = 0
        for id in ids:
            if math.floor(id / 1000) == class_id[int(label[int(n)])]:
                px = np.where(ins_seg == int(id))
                x_min = np.min(px[1])
                y_min = np.min(px[0])
                x_max = np.max(px[1])
                y_max = np.max(px[0])
                x1 = max(rois[n, 0], x_min)
                y1 = max(rois[n, 1], y_min)
                x2 = min(rois[n, 2], x_max)
                y2 = min(rois[n, 3], y_max)
                iou = (x2 - x1) * (y2 - y1)
                iou = iou / ((rois[n, 2] - rois[n, 0]) * (rois[n, 3] - rois[n, 1])
                             + (x_max - x_min) * (y_max - y_min) - iou)
                             
                # print(math.floor(id / 1000), x_min, y_min, x_max, y_max, iou)
                if iou > max_count:
                    ins_id = id
                    max_count = iou

        if max_count == 0:
            continue
        # print max_count
        mask = np.zeros(target.shape)
        idx = np.where(target == ins_id)
        mask[idx] = 1
        # cv2.imwrite('tmp/mask_train{}_{}_{}.jpg'.format(n, id, np.random.randint(1000)), mask*255)
        mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_LINEAR)

        mask_target[n] = mask
        mask_label[n] = label[int(n)]
    return mask_target, mask_label


def sample_rois(rois, gt_boxes, num_classes, rois_per_image, fg_rois_per_image, fg_overlap, box_stds, seg):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: [n, 5] (batch_index, x1, y1, x2, y2)
    :param gt_boxes: [n, 5] (x1, y1, x2, y2, cls)
    :param num_classes: number of classes
    :param rois_per_image: total roi number
    :param fg_rois_per_image: foreground roi number
    :param fg_overlap: overlap threshold for fg rois
    :param box_stds: std var of bbox reg
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    overlaps = bbox_overlaps(rois[:, 1:], gt_boxes[:, :4])
    gt_assignment = overlaps.argmax(axis=1)
    labels = gt_boxes[gt_assignment, 4]
    max_overlaps = overlaps.max(axis=1)

    # select foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(max_overlaps >= fg_overlap)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_this_image = min(fg_rois_per_image, len(fg_indexes))
    # sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_this_image:
        fg_indexes = np.random.choice(fg_indexes, size=fg_rois_this_image, replace=False)

    # select background RoIs as those within [0, FG_THRESH)
    bg_indexes = np.where(max_overlaps < fg_overlap)[0]
    # compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_this_image = rois_per_image - fg_rois_this_image
    bg_rois_this_image = min(bg_rois_this_image, len(bg_indexes))
    # sample bg rois without replacement
    if len(bg_indexes) > bg_rois_this_image:
        bg_indexes = np.random.choice(bg_indexes, size=bg_rois_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)
    # pad more bg rois to ensure a fixed minibatch size
    while len(keep_indexes) < rois_per_image:
        gap = min(len(bg_indexes), rois_per_image - len(keep_indexes))
        gap_indexes = np.random.choice(range(len(bg_indexes)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, bg_indexes[gap_indexes])

    # sample rois and labels
    rois = rois[keep_indexes]
    labels = labels[keep_indexes]
    # set labels of bg rois to be 0
    labels[fg_rois_this_image:] = 0

    mask_targets = np.zeros((rois_per_image, num_classes, 28, 28), dtype=np.int8)
    mask_weights = np.zeros((rois_per_image, num_classes, 1, 1), dtype=np.int8)

    _mask_targets, _mask_labels = compute_mask_and_label(rois[:fg_rois_this_image], labels[:fg_rois_this_image], seg)
    for i in range(fg_rois_this_image):
        mask_targets[i, _mask_labels[i]] = _mask_targets[i]
        mask_weights[i, _mask_labels[i]] = 1
    # im = np.uint8(seg[0]/1000)
    # cv2.imwrite('tmp/im.jpg', im)
    # print(mask_weights[:,:,0,0], _mask_labels)
    # cv2.rectangle(im, (int(roi[1]), int(roi[2])), (int(roi[3]), int(roi[4])), (255, 0, 0))
    # cv2.imwrite(_mask_targets[])

    # roi_idx = _mask_labels.argmax()
    # sample = mask_targets[roi_idx, _mask_labels[roi_idx]]
    # print(_mask_labels)

    # load or compute bbox_target
    targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4], box_stds=box_stds)
    bbox_targets = np.zeros((rois_per_image, 4 * num_classes), dtype=np.float32)
    bbox_weights = np.zeros((rois_per_image, 4 * num_classes), dtype=np.float32)
    for i in range(fg_rois_this_image):
        cls_ind = int(labels[i])
        bbox_targets[i, cls_ind * 4:(cls_ind + 1) * 4] = targets[i]
        bbox_weights[i, cls_ind * 4:(cls_ind + 1) * 4] = 1

    return rois, labels, bbox_targets, bbox_weights, mask_targets, mask_weights


class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction, fg_overlap, box_stds):
        super(ProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._rois_per_image = int(batch_rois / batch_images)
        self._fg_rois_per_image = int(round(fg_fraction * self._rois_per_image))
        self._fg_overlap = fg_overlap
        self._box_stds = box_stds

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_images == in_data[1].shape[0], 'check batch size of gt_boxes'

        all_rois = in_data[0].asnumpy()
        all_gt_boxes = in_data[1].asnumpy()
        all_segs = in_data[2].asnumpy()

        rois = np.empty((0, 5), dtype=np.float32)
        labels = np.empty((0, ), dtype=np.float32)
        bbox_targets = np.empty((0, 4 * self._num_classes), dtype=np.float32)
        bbox_weights = np.empty((0, 4 * self._num_classes), dtype=np.float32)
        mask_targets = np.empty((0, self._num_classes, 28, 28), dtype=np.int8)
        mask_weights = np.empty((0, self._num_classes, 1, 1), dtype=np.int8)
        for batch_idx in range(self._batch_images):
            b_rois = all_rois[np.where(all_rois[:, 0] == batch_idx)[0]]
            b_gt_boxes = all_gt_boxes[batch_idx]
            b_gt_boxes = b_gt_boxes[np.where(b_gt_boxes[:, -1] > 0)[0]]

            # Include ground-truth boxes in the set of candidate rois
            batch_pad = batch_idx * np.ones((b_gt_boxes.shape[0], 1), dtype=b_gt_boxes.dtype)
            b_rois = np.vstack((b_rois, np.hstack((batch_pad, b_gt_boxes[:, :-1]))))

            b_rois, b_labels, b_bbox_targets, b_bbox_weights, b_mask_targets, b_mask_weights = \
                sample_rois(b_rois, b_gt_boxes, num_classes=self._num_classes, rois_per_image=self._rois_per_image,
                            fg_rois_per_image=self._fg_rois_per_image, fg_overlap=self._fg_overlap, box_stds=self._box_stds, seg=all_segs)

            rois = np.vstack((rois, b_rois))
            labels = np.hstack((labels, b_labels))
            bbox_targets = np.vstack((bbox_targets, b_bbox_targets))
            bbox_weights = np.vstack((bbox_weights, b_bbox_weights))
            mask_targets = np.vstack((mask_targets, b_mask_targets))
            mask_weights = np.vstack((mask_weights, b_mask_weights))

        self.assign(out_data[0], req[0], rois)
        self.assign(out_data[1], req[1], labels)
        self.assign(out_data[2], req[2], bbox_targets)
        self.assign(out_data[3], req[3], bbox_weights)
        self.assign(out_data[4], req[4], mask_targets)
        self.assign(out_data[5], req[5], mask_weights)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register('proposal_target')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes='21', batch_images='1', batch_rois='128', fg_fraction='0.25',
                 fg_overlap='0.5', box_stds='(0.1, 0.1, 0.2, 0.2)'):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._fg_fraction = float(fg_fraction)
        self._fg_overlap = float(fg_overlap)
        self._box_stds = tuple(np.fromstring(box_stds[1:-1], dtype=float, sep=','))

    def list_arguments(self):
        return ['rois', 'gt_boxes', 'seg']

    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight', 'mask_targets', 'mask_weights']

    def infer_shape(self, in_shape):
        assert self._batch_rois % self._batch_images == 0, \
            'BATCHIMAGES {} must devide BATCH_ROIS {}'.format(self._batch_images, self._batch_rois)

        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]
        seg_shape = in_shape[2]

        output_rois_shape = (self._batch_rois, 5)
        label_shape = (self._batch_rois, )
        bbox_target_shape = (self._batch_rois, self._num_classes * 4)
        bbox_weight_shape = (self._batch_rois, self._num_classes * 4)
        mask_target_shape = (self._batch_rois, self._num_classes, 28, 28)
        mask_weight_shape = (self._batch_rois, self._num_classes, 1, 1)

        return [rpn_rois_shape, gt_boxes_shape, seg_shape], \
               [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape, mask_target_shape, mask_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._fg_fraction,
                                      self._fg_overlap, self._box_stds)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
