import mxnet as mx


def load_param(params, ctx=None):
    """same as mx.model.load_checkpoint, but do not load symnet and will convert context"""
    if ctx is None:
        ctx = mx.cpu()
    save_dict = mx.nd.load(params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            # print(name, v.shape)
            arg_params[name] = v.as_in_context(ctx)
            if name.startswith("stage4_unit"):
                arg_params['mask_'+name] = v.as_in_context(ctx)
        if tp == 'aux':
            # print(name)
            aux_params[name] = v.as_in_context(ctx)
            if name.startswith("stage4_unit"):
                aux_params['mask_'+name] = v.as_in_context(ctx)
    return arg_params, aux_params


def infer_param_shape(symbol, data_shapes):
    arg_shape, _, aux_shape = symbol.infer_shape(**dict(data_shapes))
    arg_shape_dict = dict(zip(symbol.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(symbol.list_auxiliary_states(), aux_shape))
    return arg_shape_dict, aux_shape_dict


def infer_data_shape(symbol, data_shapes):
    _, out_shape, _ = symbol.infer_shape(**dict(data_shapes))
    data_shape_dict = dict(data_shapes)
    out_shape_dict = dict(zip(symbol.list_outputs(), out_shape))
    return data_shape_dict, out_shape_dict


def check_shape(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    data_shape_dict, out_shape_dict = infer_data_shape(symbol, data_shapes)
    for k in symbol.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        if k not in arg_params:
            if k.split('_')[-1] == 'bias':
                print("{} not initialized. Initiaizing with 0.".format(k))
                arg_params[k] = mx.nd.zeros(shape=arg_shape_dict[k])
            else:
                print("{} not initialized. Initiaizing with normal.".format(k))
                arg_params[k] = mx.random.normal(0, 0.001, shape=arg_shape_dict[k])
        if arg_params[k].shape != arg_shape_dict[k]:
            if k.split('_')[-1] == 'bias':
                print("{} shape inconsistent. Initiaizing with 0.".format(k))
                arg_params[k] = mx.nd.zeros(shape=arg_shape_dict[k])
            else:
                print("{} shape inconsistent. Initiaizing with normal.".format(k))
                arg_params[k] = mx.random.normal(0, 0.001, shape=arg_shape_dict[k])
        assert k in arg_params, '%s not initialized' % k
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for %s inferred %s provided %s' % (k, arg_shape_dict[k], arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        if k not in aux_params:
            print("{} not initialized. Initiaizing with 0.".format(k))
            aux_params[k] = mx.nd.zeros(shape=aux_shape_dict[k])
        if aux_params[k].shape != aux_shape_dict[k]:
            print("{} shape inconsistent. Initiaizing with 0.".format(k))
            aux_params[k] = mx.nd.zeros(shape=aux_shape_dict[k])
        assert k in aux_params, '%s not initialized' % k
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for %s inferred %s provided %s' % (k, aux_shape_dict[k], aux_params[k].shape)


def initialize_frcnn(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
    arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
    arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
    arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
    arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_bbox_pred_weight'])
    arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])
    arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
    arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
    arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
    arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])
    arg_params['mask_deconv1_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['mask_deconv1_weight'])
    # arg_params['mask_deconv2_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['mask_deconv2_weight'])
    arg_params['mask_conv2_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['mask_conv2_weight'])
    arg_params['mask_conv2_bias'] = mx.nd.zeros(shape=arg_shape_dict['mask_conv2_bias'])
    # for i in range(1, 5):
    #     arg_params['mask_conv_t{}_weight'.format(i)] = mx.random.normal(0, 0.001, shape=arg_shape_dict['mask_conv_t{}_weight'.format(i)])
    #     arg_params['mask_conv_t{}_bias'.format(i)] = mx.nd.zeros(shape=arg_shape_dict['mask_conv_t{}_bias'.format(i)])
    
    return arg_params, aux_params


def get_fixed_params(symbol, fixed_param_prefix=''):
    fixed_param_names = []
    if fixed_param_prefix:
        for name in symbol.list_arguments():
            for prefix in fixed_param_prefix:
                if prefix in name:
                    fixed_param_names.append(name)
    return fixed_param_names


def get_all_params(symbol):
    fixed_param_names = []
    for name in symbol.list_arguments():
        fixed_param_names.append(name)
    return fixed_param_names
