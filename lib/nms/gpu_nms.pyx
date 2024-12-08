# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
cimport numpy as np
from libc.stdint cimport int32_t

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_nms.hpp":
    void _nms(int32_t* keep_out, int* num_out, const float* boxes_host,
              int boxes_num, int boxes_dim, float nms_overlap_thresh, int device_id)

def gpu_nms(np.ndarray[np.float32_t, ndim=2] boxes, float thresh, int device_id=0):
    cdef int boxes_num = boxes.shape[0]
    cdef int boxes_dim = boxes.shape[1]
    cdef np.ndarray[np.int32_t, ndim=1] keep = np.zeros(boxes_num, dtype=np.int32)
    cdef int num_out
    _nms(<int32_t*>keep.data, &num_out, <float*>boxes.data, boxes_num,
         boxes_dim, thresh, device_id)
    return keep[:num_out]
