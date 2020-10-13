"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: ml-inory
Email: madscientist_yang@foxmail.com
Description:
    MobileNet V1 backbone.
    Paper ref to https://arxiv.org/abs/1704.04861
"""

import oneflow as flow

def _get_regularizer(model_name):
    #all decay
    return flow.regularizers.l2(0.00004)

def _get_initializer(model_name):
    if model_name == "weight":
        return flow.variance_scaling_initializer(2.0, mode="fan_out", distribution="random_normal", data_format="NCHW")
    elif model_name == "bias":
        return flow.zeros_initializer()
    elif model_name == "gamma":
        return flow.ones_initializer()
    elif model_name == "beta":
        return flow.zeros_initializer()
    elif model_name == "dense_weight":
        return flow.random_normal_initializer(0, 0.01)

def conv_bn(x, oup, ksize=3, stride=1, padding='VALID', groups=1, bias=False, data_format='NCHW', prefix=''):
    '''
    Conv -> BatchNorm -> ReLU
    '''
    conv = flow.layers.conv2d(
        x,
        oup,
        kernel_size=ksize,
        strides=stride,
        padding=padding,
        groups=groups,
        data_format=data_format,
        activation=None,
        kernel_initializer=_get_initializer('weight'),
        bias_initializer=_get_initializer('bias'),
        kernel_regularizer=_get_regularizer('weight'),
        bias_regularizer=_get_regularizer('bias'),
        use_bias=bias,
        name=prefix
    )
    bn = flow.layers.batch_normalization_relu(
        conv,
        beta_initializer=_get_initializer('beta'),
        gamma_initializer=_get_initializer('gamma'),
        beta_regularizer=_get_regularizer('beta'),
        gamma_regularizer=_get_regularizer('gamma'),
        name='%s-BatchNorm'%prefix
    )
    return bn

def conv_dw(x, oup, stride, data_format='NCHW', prefix=''):
    '''
    Depthwise Conv -> PointWise Conv
    '''
    inp = x.shape[1] if data_format=='NCHW' else x.shape[-1]
    dw = conv_bn(x, inp, ksize=3, stride=stride, padding='SAME', groups=inp, bias=False, data_format=data_format, prefix=prefix+'-dw')
    pw = conv_bn(dw, oup, ksize=1, stride=1, padding='VALID', bias=False, data_format=data_format, prefix=prefix+'-pw')
    return pw

MNETV1_CONFIGS_MAP = {
    (224,224):{
        'firstconv_param': 
            (32,   2),
        # c, s
        'bottleneck_params_list':[
            (64,   1), 
            (128,  2), 
            (128,  1), 
            (256,  2), 
            (256,  1), 
            (512,  2), 
            (512,  1), 
            (512,  1), 
            (512,  1), 
            (512,  1), 
            (512,  1), 
            (1024, 2)
        ],
        'filter_num_before_gp': 1024, 
    } 
}

class MobileNetV1(object):
    def __init__(self, data_wh, multiplier, **kargs):
        super(MobileNetV1, self).__init__()
        self.data_wh = data_wh
        self.multiplier = multiplier
        if self.data_wh in MNETV1_CONFIGS_MAP:
            self.config_map = MNETV1_CONFIGS_MAP[self.data_wh]
        else:
            print(f'{data_wh} not in MNETV1_CONFIGS_MAP, use default setting(224,224)')
            self.config_map = MNETV1_CONFIGS_MAP[(224, 224)]

    def build_network(self, input_data, data_format='NCHW', class_num=1000, prefix='', **configs):
        self.config_map.update(configs)

        first_c = int(round(self.config_map['firstconv_param'][0]*self.multiplier))
        first_layer = conv_bn(
            input_data, 
            first_c, 
            stride=self.config_map['firstconv_param'][1], 
            data_format=data_format,
            prefix=prefix+'-Conv0'
        )

        last_bottleneck_layer = first_layer
        for i, layer_setting in enumerate(self.config_map['bottleneck_params_list']):
            c, s = layer_setting
            last_bottleneck_layer = conv_dw(last_bottleneck_layer, int(round(c*self.multiplier)), s, data_format=data_format, prefix=prefix+'-Conv%d'%(i+1))
        
        last_fm = conv_dw(last_bottleneck_layer, self.config_map['filter_num_before_gp'], 1, data_format=data_format, prefix=prefix+'-dw_%d'%(len(self.config_map['bottleneck_params_list'])))

        # global average pooling
        pool_size = max(1, int(self.data_wh[0] / 32))
        pool = flow.nn.avg_pool2d(
            last_fm, ksize=pool_size, strides=1, padding="VALID", data_format=data_format, name="pool1",
        ) 
        fc = flow.layers.dense(
            flow.reshape(pool, (pool.shape[0], -1)),
            units=class_num,
            use_bias=False,
            kernel_initializer=_get_initializer("dense_weight"),
            bias_initializer=_get_initializer("bias"),
            kernel_regularizer=_get_regularizer("dense_weight"),
            bias_regularizer=_get_regularizer("bias"),
            name=prefix+'-fc',
        )
        return fc

    def __call__(self, input_data, class_num=1000, prefix='', **configs):
        sym = self.build_network(input_data, class_num=class_num, prefix=prefix, **configs)
        return sym

def Mobilenet_V1(input_data, args, trainable=True, training=True, num_classes=1000, multiplier=1.0, prefix = ""):
    assert   args.channel_last==False, "Mobilenet does not support channel_last mode, set channel_last=False will be right!"
    data_format="NHWC" if args.channel_last else "NCHW"
    mobilenetgen = MobileNetV1((224,224), multiplier=multiplier)
    out = mobilenetgen(input_data, data_format=data_format, class_num=num_classes, prefix = "MobilenetV1")
    print('MobileNetV1 build done.')
    return out