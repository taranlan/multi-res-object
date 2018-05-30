import mxnet as mx

def deconvolution_module(conv1, conv2, num_filter, upstrid = 2, level = 1, lr_mult=1):
    deconv2 = mx.symbol.Deconvolution(
        data=conv2, kernel=(2, 2), stride=(upstrid, upstrid),num_filter=num_filter, attr={'lr_mult': '%f' % lr_mult}, name="{}de_{}".format(conv2.name, str(level)))
    conv2_1 = mx.symbol.Convolution(
        data=deconv2, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, attr={'lr_mult': '%f' % lr_mult}, name="{}de_conv_{}".format(conv2.name, str(level)))
    BN2 = mx.symbol.BatchNorm(data=conv2_1, name="{}de_conv_bn_{}".format(conv2.name, str(level)))
    conv1_1 = mx.symbol.Convolution(
        data=conv1, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, attr={'lr_mult': '%f' % lr_mult}, name="{}conv_{}".format(conv1.name, str(level)))
    BN1_1 = mx.symbol.BatchNorm(data=conv1_1, name="{}conv_bn_{}".format(conv1.name, str(level)))
    relu1_1 = mx.symbol.Activation(data=BN1_1, act_type="relu", name="{}conv_bn_relu_{}".format(conv1.name, str(level)))
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, attr={'lr_mult': '%f' % lr_mult}, name="{}conv_bn_relu_conv_{}".format(conv1.name, str(level)))
    BN1_2 = mx.symbol.BatchNorm(data=conv1_2, name="{}conv_bn_relu_conv_bn{}".format(conv1.name, str(level)))
    BN2_clip = mx.symbol.Crop(*[BN2, BN1_2])
    element_product1plus2up = mx.symbol.broadcast_mul(BN1_2, BN2_clip)
    relu = mx.symbol.Activation(data=element_product1plus2up, act_type="relu", name="conv{}product{}de".format(str(level),str(level+1)))
    return relu

def construct_multi_res_layer(from_layers, num_filters):
    multi_res_layers = []
    num_layers = len(from_layers)

    concat_conv = deconvolution_module(from_layers[1],from_layers[2],512,2,1)
    from_layers[1] = concat_conv

    concat_conv = deconvolution_module(from_layers[0],concat_conv,256,2,2)
    from_layers[0] = concat_conv

    # last convolutional layer for detection, lowest resolution
    #conv1 = mx.symbol.Convolution(data=anti_layers[0], kernel=(1,1), num_filter=num_filters[0], name="conv1")
    # fuse convolutional layers with multiple resolution
    #print "number of layers: ", num_layers
    #if num_layers >= 2: 
    #    deconv2 = mx.symbol.Deconvolution(data=conv1, kernel=(2, 2), stride=(2, 2), num_filter=num_filters[0], name="de2")  # 2X
        #conv2 = mx.symbol.Convolution(data=anti_layers[1], kernel=(3, 3), pad=(1, 1), num_filter=num_filters[0], name="conv2")
    #    conv2 = mx.symbol.Convolution(data=anti_layers[1], kernel=(1, 1), num_filter=num_filters[0], name="conv2")
        #bn2 = mx.symbol.BatchNorm(data=conv2_1, name="de_conv_bn2}")
        #bn2_clip = mx.symbol.Crop(*[bn2, bn1_2])
    #    deconv2_clip = mx.symbol.Crop(*[deconv2, conv2])
        #score_fused = bn1_2 + bn2_clip
    #    score_fused = deconv2_clip + conv2
    #if num_layers >= 3:
    #    deconv3 = mx.symbol.Deconvolution(data=score_fused, kernel=(2, 2), stride=(2, 2), num_filter=num_filters[0], name="de3")  # 2X
        #conv3 = mx.symbol.Convolution(data=anti_layers[2], kernel=(3, 3), pad=(1, 1), num_filter=num_filters[0], name="conv3")
    #    conv3 = mx.symbol.Convolution(data=anti_layers[2], kernel=(1, 1), num_filter=num_filters[0], name="conv3")
    #    deconv3_clip = mx.symbol.Crop(*[deconv3, conv3])
    #    score_final = deconv3_clip + conv3

    #if num_layers == 1: 
    #    multi_res_layers.append(conv1)
    #elif num_layers == 2: 
    #    multi_res_layers.append(score_fused)
    #elif num_layers == 3:
    #    multi_res_layers.append(score_final)

    #return multi_res_layers
    return from_layers

