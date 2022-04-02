# @Time    : 2022/3/20 14:34
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : train
# @Project Name :code

"""
Retrain the YOLO model for your own dataset.
"""
import os
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
logical_devices = tf.config.list_logical_devices("GPU")#
# import tensorflow.keras.backend.tensorflow_backend as KTF
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = '2007_train.txt'
    # annotation_path_hie = '2022_train.txt'#共151张图，1890个实例

    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    model = create_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path='model_data/ep203-loss17.837-val_loss19.428.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=2)#每3步保存一次
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1#得到数据集
    lines_voc,num_train,num_val = get_lines(annotation_path,val_split)
#=======================================================================================================================
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.模型预热
    if True:#先用voc数据集进行训练
        model.compile(optimizer=Adam(lr=1e-4), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 16
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines_voc[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines_voc[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=300,
                initial_epoch=204,
                callbacks=[logging,checkpoint])
        model.save_weights(log_dir + 'trained_weights_FINAL.h5')
#=======================================================================================================================
    # if True:#接着300步采用voc+hie继续训练
    #     for i in range(len(model.layers)):
    #         model.layers[i].trainable = True
    #     model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    #     print('Unfreeze all of the layers.')
    #
    #     batch_size = 12 # note that more GPU memory is required after unfreezing the body
    #     print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    #     model.fit_generator(data_generator_wrapper(lines_voc[:num_train], batch_size, input_shape, anchors, num_classes),
    #         steps_per_epoch=max(1, num_train//batch_size),
    #         validation_data=data_generator_wrapper(lines_voc[num_train:], batch_size, input_shape, anchors, num_classes),
    #         validation_steps=max(1, num_val//batch_size),
    #         epochs=1050,
    #         initial_epoch=550,
    #         callbacks=[logging, checkpoint])
    #     model.save_weights(log_dir + 'trained_weights_final_new.h5')
#=======================================================================================================================
    # Further training if needed.
#=======================================================================================================================

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    #图像变更此处需要修改
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]


    model_body = yolo_body(image_input, num_anchors//3, num_classes,'dla_model')#original_yolo
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        # if freeze_body in [1, 2]:#注释
        #     # Freeze darknet53 body or freeze all but 3 output layers.
        #     num = (185, len(model_body.layers)-3)[freeze_body-1]
        #     for i in range(num): model_body.layers[i].trainable = False
        #     print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)#数据增强处理
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def get_lines(annotation_path,val_split):
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    return lines,num_train,num_val
if __name__ == '__main__':
    _main()
