# @Time    : 2022/3/20 14:43
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : dla_model
# @Project Name :code
# @Time    : 2022/3/12 15:13
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : dla_model
# @Project Name :keras-yolo3-master
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Add, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from functools import wraps
from tensorflow.keras.layers import Lambda
from yolo3.utils import compose

@wraps(Conv2D)
def liner_Conv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DBL(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        liner_Conv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))
def bn_leaky(*args):
    return compose(BatchNormalization(),LeakyReLU(alpha=0.1))

def dla_res(x, num_filters0,num_filters1,num_filters2, kernel_size0,kernel_size1,kernel_size2):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x1 = liner_Conv2D(num_filters0, kernel_size0)(x)
    y = compose(
            DBL(num_filters1, kernel_size1),
            DBL(num_filters2, kernel_size2))(x)
    x2 = liner_Conv2D(num_filters0, kernel_size0)(y)
    x = Add()([x1,x2])
    return x

def dla_body(x):
    #stage one
    x1 = DBL(16, (3, 3))(x)
    x2 = dla_res(x1,16, 4, 4,(1, 1), (1, 1), (3, 3))
    x_add = MaxPooling2D(pool_size=2)(x2)
    x3 = bn_leaky()(x_add)
    x4 = dla_res(x3, 32, 8, 8, (1, 1), (1, 1), (3, 3))
    x5 = MaxPooling2D(pool_size=2)(x4)
    x6 = bn_leaky()(x5)
    x7 = dla_res(x6, 128, 32, 32,(1, 1), (1, 1), (3, 3))
    x8 = bn_leaky()(x7)
    x9 = dla_res(x8, 128, 32, 32, (1, 1), (1, 1), (3, 3))
    x10 = Lambda(lambda x: tf.concat(x, axis=-1))([x7,x9])
    #stage two
    x11 = DBL(128, (1, 1))(x10)  # 第一条长连接
    x12 = MaxPooling2D(pool_size=2)(x11)
    x13 = bn_leaky()(x12)
    x14 = dla_res(x13, 256, 64, 64,(1, 1), (1, 1), (3, 3))
    x15 = bn_leaky()(x14)
    x16 = dla_res(x15, 256, 64, 64, (1, 1), (1, 1), (3, 3))
    x17 = Lambda(lambda x: tf.concat(x, axis=-1))([x14,x16])
    x18 = DBL(256, (1, 1))(x17)  # 第二条长连
    x19 = dla_res(x18, 256, 64, 64, (1, 1), (1, 1), (3, 3))  # 第三条长链
    x20 = dla_res(x19, 256, 64, 64, (1, 1), (1, 1), (3, 3))
    x_11 = MaxPooling2D(pool_size=2)(x11)
    x21 = Lambda(lambda x: tf.concat(x, axis=-1))([x_11,x18,x19,x20])# 首次聚合（52，52）,第3
    #stage three
    x22 = DBL(256, (1, 1))(x21)  # 第一条长连接
    x23 =  MaxPooling2D(pool_size=2)(x22)
    x24 = bn_leaky()(x23)
    x25 = dla_res(x24, 512, 128, 128, (1, 1), (1, 1), (3, 3))
    x26 = bn_leaky()(x25)
    x27 = dla_res(x26, 512, 128, 128, (1, 1), (1, 1), (3, 3))
    x28 = Lambda(lambda x: tf.concat(x, axis=-1))([x25,x27])
    x29 = DBL(512, (1, 1))(x28) # 出来两条长线
    x30 = dla_res(x29, 512, 128, 128, (1, 1), (1, 1), (3, 3))  # 待拼接
    x31 = bn_leaky()(x30)
    x32 = dla_res(x31, 512, 128, 128, (1, 1), (1, 1), (3, 3))
    x33 = Lambda(lambda x: tf.concat(x, axis=-1))([x29,x30,x32])  # 此处去掉x29一根线
    x34 = DBL(512, (1, 1))(x33)  # 出来一根长线
    x35 = dla_res(x34, 512, 128, 128, (1, 1), (1, 1), (3, 3))  # 出来一根线
    x36 = dla_res(x35, 512, 128, 128, (1, 1), (1, 1), (3, 3))
    x_22 = MaxPooling2D(pool_size=2)(x22)
    x37 = Lambda(lambda x: tf.concat(x, axis=-1))([x_22, x29, x34, x35, x36])  # 第二次聚合（26，26）第6
    #stage four
    x38 = DBL(512, (1, 1))(x37)  # 出来第一条长线
    x39 = MaxPooling2D(pool_size=2)(x38)
    x40 = bn_leaky()(x39)
    x41 = dla_res(x40, 1024, 256, 256, (1, 1), (1, 1), (3, 3))
    x42 = bn_leaky()(x41)
    x43 = dla_res(x42, 1024, 256, 256, (1, 1), (1, 1), (3, 3))
    x_38 = MaxPooling2D(pool_size=2)(x38)
    x44 = Lambda(lambda x: tf.concat(x, axis=-1))([x_38, x41, x43])  # 第三次聚合（13，13）
    x45 = DBL(1024, (1, 1))(x44)
    return x45
def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DBL(num_filters, (1,1)),
            DBL(num_filters*2, (3,3)),
            DBL(num_filters, (1,1)),
            DBL(num_filters*2, (3,3)),
            DBL(num_filters, (1,1)))(x)
    y = compose(
            DBL(num_filters*2, (3,3)),
            liner_Conv2D(out_filters, (1,1)))(x)
    return x, y
def dla_yolo(image_input,num_anchors, num_classes):#97,171
    dla_net = Model(image_input, dla_body(image_input))#此处进行了修改
    x, y1 = make_last_layers(dla_net.output, 512, num_anchors*(num_classes+5))
    x = compose(
                DBL(256, (1,1)),
                UpSampling2D(2))(x)
    x = Concatenate()([x,dla_net.layers[175].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DBL(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,dla_net.layers[101].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))
    return Model(image_input,[y1,y2,y3])