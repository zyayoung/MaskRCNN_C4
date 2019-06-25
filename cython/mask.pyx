# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float

def compute_mask_and_label_single_cython(
        np.ndarray[np.float32_t, ndim=1] roi,
        int label,
        np.ndarray[np.float32_t, ndim=2] ins_seg
        ):
    cdef list class_id = [0, 24, 25, 26, 27, 28, 31, 32, 33]
    cdef np.ndarray[np.float32_t, ndim=2] target = ins_seg[int(roi[1]): int(roi[3]), int(roi[0]): int(roi[2])]
    cdef np.ndarray[np.int32_t, ndim=1] ids = np.int32(np.unique(target))
    cdef int ins_id=0, _id
    cdef float max_count = 0
    cdef float x_min, y_min, x_max, y_max, x1, y1, x2, y2, iou
    ids = ids[np.floor(ids / 1000) == class_id[int(label)]]
    for _id in ids:
        px = np.where(ins_seg == _id)
        x_min = min(px[1])
        y_min = min(px[0])
        x_max = max(px[1])
        y_max = max(px[0])
        x1 = max(roi[0], x_min)
        y1 = max(roi[1], y_min)
        x2 = min(roi[2], x_max)
        y2 = min(roi[3], y_max)
        iou = (x2 - x1) * (y2 - y1)
        iou = iou / ((roi[2] - roi[0]) * (roi[3] - roi[1])
                        + (x_max - x_min) * (y_max - y_min) - iou)
                        
        if iou > max_count:
            ins_id = _id
            max_count = iou
    return target == ins_id, label