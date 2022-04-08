# @Time    : 2022/3/25 22:09
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : cal_ap
# @Project Name :keras-yolo3-master
import numpy as np
import pandas as pd

class Cal_average_precision():
    def __init__(self,ap_path,num_pred_box,num_ground_truth,output_path):
        self.ap_path = ap_path
        self.num_pred_box = num_pred_box
        self.num_ground_truth = num_ground_truth
        self.df = pd.read_csv(self.ap_path)
        self.df.sort_values(by='Confi', ascending=False, inplace=True)
        self.output_path = output_path
        acc_tp = []
        tp_ = 0
        for index, value in enumerate(self.df['TP'].values.tolist()):
            if index == 0:
                acc_tp.append(value)
                tp_ += value
            else:
                tp_ += value
                acc_tp.append(tp_)
        self.df['acc_tp'] = acc_tp

        acc_fp = []
        fp_ = 0
        for index, value in enumerate(self.df['FP'].values.tolist()):
            if index == 0:
                acc_fp.append(value)
                fp_ += value
            else:
                fp_ += value
                acc_fp.append(fp_)
        self.df['acc_fp'] = acc_fp
        # 计算precision
        self.df['precision'] = self.df['acc_tp'] / np.arange(1, len(self.df['acc_tp']) + 1)
        self.df['recall'] = self.df['acc_tp'] / self.num_ground_truth
        self.df.to_csv(self.output_path)
    def cal_ap(self):#11points
        re_set = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        recall_ = self.df['recall'].values.tolist()
        precision_ = self.df['precision'].values.tolist()
        max_presicion = []
        for re_ in re_set:
            pr_index = []
            for index,re in enumerate(recall_):
                if re>=re_:
                    pr_index.append(index)
            if len(pr_index)!=0:
                max_presicion.append(np.max([precision_[i] for i in pr_index]))
            else:
                max_presicion.append(0)
        print(np.average(max_presicion))


def main():
    #ap_path,num_pred_box,num_ground_truth,output_path
    cal_ = Cal_average_precision('../ap_eval.csv',433,970,'./output_yolo3coco.csv')#修改这里就好
    cal_.cal_ap()

if __name__ == '__main__':
    main()

