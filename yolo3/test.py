# @Time    : 2022/3/20 14:44
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : test
# @Project Name :code

import os
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
logical_devices = tf.config.list_logical_devices("GPU")#
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras.models import load_model
from train import get_classes,get_anchors
from tensorflow.keras import backend as K
from yolo3.model import yolo_eval, yolo_body
from tensorflow.keras.layers import Input
from yolo3.utils import *
import numpy as np
import pandas as pd
model_path = '../model_data/ep240-loss16.916-val_loss18.087.h5'
anchors_path = '../model_data/yolo_anchors.txt'
classes_path = '../model_data/voc_classes.txt'
classes_path_voc = '../model_data/voc_classes.txt'
annotation_path = '../2022_test.txt'
score =  0.3
iou = 0.45
model_image_size = (416,416)
iou_panduan = 0.75
print(os.getcwd())

#获取分类名字，与anchors
class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)
class_names_voc = get_classes(classes_path_voc)
#获得模型,yolo_model
model_path = os.path.expanduser(model_path)
assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
num_anchors = len(anchors)
num_classes = len(class_names)
try:
    yolo_model = load_model(model_path, compile=False)
except:
    yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes, 'dla_model')
    yolo_model.load_weights(model_path)  # make sure model, anchors and classes match
else:
    assert yolo_model.layers[-1].output_shape[-1] == \
           num_anchors / len(yolo_model.output) * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes'
print('{} model, anchors, and classes loaded.'.format(model_path))
#定义feed
sess = tf.compat.v1.keras.backend.get_session()
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = yolo_eval(yolo_model.output, anchors,
                                       len(class_names), input_image_shape,
                                       score_threshold=score, iou_threshold=iou)
#读取数据
with open(annotation_path) as f:
    lines = f.readlines()
#需要存放的参数
confidence = []
TP = []
FP = []
num_box_label = []
num_box_pred = []
iou_eval = []
ap_pred_box = 0
ap_bounding_box = 0
#此处需要循环开始
for line in lines[:70]:#读图片，前30张图
    line = line.split()
    box_info = get_label(line)
    num_box_label.append(len(box_info))
    for single_box in box_info:
        if single_box[-1]==14:#统计有多少个真实框，这是基于voc的索引,voc与coco的类别索引均是0开始
            ap_bounding_box+=1
    image = Image.open(line[0])
    image_data = reshape_image(image,model_image_size)
    out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: image_data,
                    input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
    num_box_pred.append(len(out_boxes))
    for i, c in reversed(list(enumerate(out_classes))):#读图片里面框，遍历每一个预测框,可能存在漏检
        if c==14:#检测人的ap，这是基于coco的
            ap_pred_box+=1
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            top, left, bottom, right = get_int_box(box,image)
            pred_box = [top, left, bottom, right]
            ious = cal_iou(pred_box,box_info)#单个预测框与多个检测框进行对比
            max_label_index = np.argmax(ious)
            #打印看一下iou大于1的预测框与标注框
            if np.max(ious)>1:
                print('这是预测框：',pred_box)
                print('这是标注框：',box_info[max_label_index])
            iou_eval.append(np.max(ious))
            label_class = class_names_voc[box_info[max_label_index][-1]]
            if np.max(ious)>=iou_panduan and predicted_class==label_class:#预测正样本的特征是具备高iou以及类别是相同的,box_info是voc的索引，而c为coco的索引,初始标签索引从0开始
                TP.append(1)
                FP.append(0)
            else:
                TP.append(0)
                FP.append(1)
            confidence.append(score)
print('共计{}个检测框架，{}个真实框架'.format(np.sum(num_box_pred),np.sum(num_box_label)))
print('共计{}个行人检测框架，{}个行人真实框架'.format(ap_pred_box,ap_bounding_box))
df= pd.DataFrame(np.transpose([TP,FP,confidence,iou_eval]),columns=['TP','FP','Confi','iou'])
df.to_csv('../ap_eval.csv')



