import mxnet as mx
from . import proposal_target, mask_output, mask_roi

eps=2e-5
use_global_stats=True
workspace=1024


def residual_unit(data, num_filter, stride, dim_match, name, prefix=''):
    # bn1_g = mx.symbol.Variable(name + '_bn1_gamma')
    # bn1_b = mx.symbol.Variable(name + '_bn1_beta')
    # bn2_g = mx.symbol.Variable(name + '_bn2_gamma')
    # bn2_b = mx.symbol.Variable(name + '_bn2_beta')
    # bn3_g = mx.symbol.Variable(name + '_bn3_gamma')
    # bn3_b = mx.symbol.Variable(name + '_bn3_beta')
    if prefix:
        prefix += '_'
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=prefix+name + '_bn1', )
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=prefix+name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True, workspace=workspace , name=prefix+name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=prefix+name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=prefix+name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=True, workspace=workspace, name=prefix+name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=prefix+name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=prefix+name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                               workspace=workspace, name=prefix+name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=prefix+name + '_sc')
    sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=prefix+name + '_plus')
    return sum


def get_resnet_feature(data, units, filter_list):
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i)

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i)

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i)
    return unit


def get_resnet_top_feature(data, units, filter_list, prefix=''):
    unit = residual_unit(data=data, num_filter=filter_list[3], stride=(1, 1), dim_match=False, name='stage4_unit1', prefix=prefix)
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i, prefix=prefix)

    bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    return unit


def get_resnet_train(anchor_scales, anchor_ratios, rpn_feature_stride,
                     rpn_pre_topk, rpn_post_topk, rpn_nms_thresh, rpn_min_size, rpn_batch_rois,
                     num_classes, rcnn_feature_stride, rcnn_pooled_size, rcnn_batch_size,
                     rcnn_batch_rois, rcnn_fg_fraction, rcnn_fg_overlap, rcnn_bbox_stds,
                     units, filter_list):
    num_anchors = len(anchor_scales) * len(anchor_ratios)

    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    seg = mx.symbol.Variable(name="seg")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')
    

    # shared convolutional layers
    conv_feat = get_resnet_feature(data, units=units, filter_list=filter_list)

    # rpn feature
    rpn_conv = mx.symbol.Convolution(
        data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")

    # rpn classification
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
    rpn_cls_act = mx.symbol.softmax(
        data=rpn_cls_score_reshape, axis=1, name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')

    # rpn bbox regression
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
    rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / rpn_batch_rois)

    # rpn proposal
    rois = mx.symbol.contrib.MultiProposal(
        cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        feature_stride=rpn_feature_stride, scales=anchor_scales, ratios=anchor_ratios,
        rpn_pre_nms_top_n=rpn_pre_topk, rpn_post_nms_top_n=rpn_post_topk,
        threshold=rpn_nms_thresh, rpn_min_size=rpn_min_size)

    # rcnn roi proposal target
    group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes, seg=seg, im_info=im_info, op_type='proposal_target',
                             num_classes=num_classes, batch_images=rcnn_batch_size,
                             batch_rois=rcnn_batch_rois, fg_fraction=rcnn_fg_fraction,
                             fg_overlap=rcnn_fg_overlap, box_stds=rcnn_bbox_stds)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]
    mask_target = group[4]
    mask_weight = group[5]

    # rcnn roi pool
    roi_pool = mx.symbol.contrib.ROIAlign(
        name='roi_pool', data=conv_feat, rois=rois, pooled_size=rcnn_pooled_size, spatial_scale=1.0 / rcnn_feature_stride)

    # rcnn top feature
    top_feat = get_resnet_top_feature(roi_pool, units=units, filter_list=filter_list)

    
    mask_deconv1 = mx.symbol.Deconvolution(data=top_feat, kernel=(2, 2), stride=(2, 2), num_filter=256,
                                            name="mask_deconv1")
    mask_relu1 = mx.symbol.Activation(data=mask_deconv1, act_type="relu", name="mask_relu1")
    mask_conv_tmp = mx.symbol.Convolution(data=mask_relu1, kernel=(3, 3), num_filter=256, pad=(1, 1),
                                          name="mask_conv_t1")
    mask_conv_tmp = mx.symbol.Activation(data=mask_conv_tmp, act_type="relu")
    mask_conv_tmp = mx.symbol.Convolution(data=mask_conv_tmp, kernel=(3, 3), num_filter=256, pad=(1, 1),
                                          name="mask_conv_t2")
    mask_conv_tmp = mx.symbol.Activation(data=mask_conv_tmp, act_type="relu")
    mask_conv_tmp = mx.symbol.Convolution(data=mask_conv_tmp, kernel=(3, 3), num_filter=256, pad=(1, 1),
                                          name="mask_conv_t3")
    mask_conv_tmp = mx.symbol.Activation(data=mask_conv_tmp, act_type="relu")
    mask_conv_tmp = mx.symbol.Convolution(data=mask_conv_tmp, kernel=(3, 3), num_filter=256, pad=(1, 1),
                                          name="mask_conv_t4")
    mask_conv_tmp = mx.symbol.Activation(data=mask_conv_tmp, act_type="relu")
    mask_deconv2 = mx.symbol.Deconvolution(data=mask_conv_tmp, kernel=(2, 2), stride=(2, 2), num_filter=256,
                                            name="mask_deconv2")
    mask_relu2 = mx.symbol.Activation(data=mask_deconv2, act_type="relu")
    mask_conv2 = mx.symbol.Convolution(data=mask_relu1, kernel=(1, 1), num_filter=num_classes,
                                          name="mask_conv2")
    mask_prob = mx.symbol.Activation(data=mask_conv2, act_type='sigmoid', name="mask_prob")
    mask_output = mx.symbol.Custom(mask_prob=mask_prob, mask_target=mask_target, mask_weight=mask_weight,
                                   label=label, name="mask_output", op_type='MaskOutput')
    
    pool1 = mx.symbol.Pooling(data=top_feat, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flatten = mx.symbol.Flatten(data=pool1, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=1024)
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    fc7 = mx.symbol.FullyConnected(data=relu6, num_hidden=1024)
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_relu7")

    # rcnn classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=flatten, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')

    # rcnn bbox regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=flatten, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / rcnn_batch_rois)

    # reshape output
    label = mx.symbol.Reshape(data=label, shape=(rcnn_batch_size, -1), name='label_reshape')
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(rcnn_batch_size, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(rcnn_batch_size, -1, 4 * num_classes), name='bbox_loss_reshape')

    # group output
    group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label), mask_output, mask_target, mask_weight])
    return group


def get_resnet_test(anchor_scales, anchor_ratios, rpn_feature_stride,
                    rpn_pre_topk, rpn_post_topk, rpn_nms_thresh, rpn_min_size,
                    num_classes, rcnn_feature_stride, rcnn_pooled_size, rcnn_batch_size,
                    units, filter_list):
    num_anchors = len(anchor_scales) * len(anchor_ratios)

    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_resnet_feature(data, units=units, filter_list=filter_list)

    # rpn feature
    rpn_conv = mx.symbol.Convolution(
        data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")

    # rpn classification
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_act = mx.symbol.softmax(
        data=rpn_cls_score_reshape, axis=1, name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')

    # rpn bbox regression
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # rpn proposal
    rois = mx.symbol.contrib.MultiProposal(
        cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        feature_stride=rpn_feature_stride, scales=anchor_scales, ratios=anchor_ratios,
        rpn_pre_nms_top_n=rpn_pre_topk, rpn_post_nms_top_n=rpn_post_topk,
        threshold=rpn_nms_thresh, rpn_min_size=rpn_min_size)

    # rcnn roi pool
    roi_pool = mx.symbol.contrib.ROIAlign(
        name='roi_pool', data=conv_feat, rois=rois, pooled_size=rcnn_pooled_size, spatial_scale=1.0 / rcnn_feature_stride)

    # rcnn top feature
    top_feat = get_resnet_top_feature(roi_pool, units=units, filter_list=filter_list)
    
    pool1 = mx.symbol.Pooling(data=top_feat, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flatten = mx.symbol.Flatten(data=pool1, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=1024)
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    fc7 = mx.symbol.FullyConnected(data=relu6, num_hidden=1024)
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_relu7")

    # rcnn classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=flatten, num_hidden=num_classes)
    cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)

    # rcnn bbox regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=flatten, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(rcnn_batch_size, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(rcnn_batch_size, -1, 4 * num_classes), name='bbox_pred_reshape')

    mask_rois = mx.symbol.Custom(data=data, label=cls_prob, rois=rois, bbox_deltas=bbox_pred,
                                 op_type='mask_roi',
                                 num_classes=num_classes, name='mask_roi')

    # rcnn roi pool
    mask_roi_pool = mx.symbol.contrib.ROIAlign(
        name='mask_roi_pool', data=conv_feat, rois=mask_rois, pooled_size=rcnn_pooled_size, spatial_scale=1.0 / rcnn_feature_stride)

    # rcnn top feature
    mask_top_feat = get_resnet_top_feature(mask_roi_pool, units=units, filter_list=filter_list, prefix="mask")
    
    mask_deconv1 = mx.symbol.Deconvolution(data=mask_top_feat, kernel=(2, 2), stride=(2, 2), num_filter=256,
                                        name="mask_deconv1")
    mask_relu1 = mx.symbol.Activation(data=mask_deconv1, act_type="relu", name="mask_relu1")
    mask_conv_tmp = mx.symbol.Convolution(data=mask_relu1, kernel=(3, 3), num_filter=256, pad=(1, 1),
                                        name="mask_conv_t1")
    mask_conv_tmp = mx.symbol.Activation(data=mask_conv_tmp, act_type="relu")
    mask_conv_tmp = mx.symbol.Convolution(data=mask_conv_tmp, kernel=(3, 3), num_filter=256, pad=(1, 1),
                                        name="mask_conv_t2")
    mask_conv_tmp = mx.symbol.Activation(data=mask_conv_tmp, act_type="relu")
    mask_conv_tmp = mx.symbol.Convolution(data=mask_conv_tmp, kernel=(3, 3), num_filter=256, pad=(1, 1),
                                        name="mask_conv_t3")
    mask_conv_tmp = mx.symbol.Activation(data=mask_conv_tmp, act_type="relu")
    mask_conv_tmp = mx.symbol.Convolution(data=mask_conv_tmp, kernel=(3, 3), num_filter=256, pad=(1, 1),
                                        name="mask_conv_t4")
    mask_conv_tmp = mx.symbol.Activation(data=mask_conv_tmp, act_type="relu")
    mask_deconv2 = mx.symbol.Deconvolution(data=mask_conv_tmp, kernel=(2, 2), stride=(2, 2), num_filter=256,
                                        name="mask_deconv2")
    mask_relu2 = mx.symbol.Activation(data=mask_deconv2, act_type="relu")
    mask_conv2 = mx.symbol.Convolution(data=mask_relu1, kernel=(1, 1), num_filter=num_classes,
                                          name="mask_conv2")
    mask_prob = mx.symbol.Activation(data=mask_conv2, act_type='sigmoid', name="mask_prob")

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred, mask_prob])
    return group
