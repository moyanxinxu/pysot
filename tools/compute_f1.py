import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
dir1 = '/root/pysot/output'
dir2 = '/root/pysot/training_dataset/satsot/train'
# 设置全局美化风格

plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.edgecolor'] = '#333F4B'
mpl.rcParams['axes.linewidth'] = 1.2

def calculate_iou(rec_1,rec_2):
    '''
    rec_1:左上角(rec_1[0],rec_1[1])    右下角：(rec_1[2],rec_1[3])
    rec_2:左上角(rec_2[0],rec_2[1])    右下角：(rec_2[2],rec_2[3])

    （rec_1）
    1--------1
    1   1----1------1
    1---1----1      1
        1           1
        1-----------1 （rec_2）
    '''
    rec_1 = [rec_1[0], rec_1[1], rec_1[0] + rec_1[2], rec_1[1] + rec_1[3]]  # 转换为左上角和右下角坐标
    rec_2 = [rec_2[0], rec_2[1], rec_2[0] + rec_2[2], rec_2[1] + rec_2[3]]  # 转换为左上角和右下角坐标
    s_rec1=(rec_1[2]-rec_1[0])*(rec_1[3]-rec_1[1])   #第一个bbox面积 = 长×宽
    s_rec2=(rec_2[2]-rec_2[0])*(rec_2[3]-rec_2[1])   #第二个bbox面积 = 长×宽
    sum_s=s_rec1+s_rec2                              #总面积
    left=max(rec_1[0],rec_2[0])                      #交集左上角顶点横坐标
    right=min(rec_1[2],rec_2[2])                     #交集右下角顶点横坐标
    bottom=max(rec_1[1],rec_2[1])                    #交集左上角顶点纵坐标
    top=min(rec_1[3],rec_2[3])                       #交集右下角顶点纵坐标
    if left >= right or top <= bottom:               #不存在交集的情况
        return 0
    else:
        inter=(right-left)*(top-bottom)              #求交集面积
        iou=(inter/(sum_s-inter))*1.0                #计算IOU
        return iou

# def evaluate_tracking(pred_boxes, gt_boxes, iou_threshold=0.5):
#     assert len(pred_boxes) == len(gt_boxes), "预测框和真实框数量必须相同"

#     # 计数器
#     tp = 0  # True Positives
#     fp = 0  # False Positives
#     fn = 0  # False Negatives

#     # 成功率与误差
#     for pred, gt in zip(pred_boxes, gt_boxes):
#         iou = calculate_iou(pred, gt)
#         if iou >= iou_threshold:
#             tp += 1
#         else:
#             fp += 1
#             fn += 1

#     # Precision, Recall, F1 Score计算
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

#     return precision, recall, f1_score

pred_boxes = []
gt_boxes = []

for sub_dir in tqdm(os.listdir(dir1)):
    dir_path = os.path.join(dir1, sub_dir)
    if os.path.isfile(dir_path):
        df = pd.read_json(dir_path,  orient='records')

        for idx, row in df.iterrows():
            pred_boxes.append([row['x'], row['y'], row['w'], row['h']])

for sub_dir in tqdm(os.listdir(dir2)):
    dir_path = os.path.join(dir2, sub_dir, 'groundtruth.txt')

    if not os.path.exists(dir_path):
        continue

    with open(dir_path, 'r') as f:
        gt_data = f.readlines()
        for line in gt_data:
            if line.startswith('none'):
                line = '0,0,0,0'
            else:
                line = line.strip()
            box = list(map(int, line.split(',')))
            gt_boxes.append(box)

ious = []
center_errors = []

for pred, gt in zip(pred_boxes, gt_boxes):
    iou = calculate_iou(pred, gt)
    ious.append(iou)

    pred_cx = pred[0] + pred[2] / 2
    pred_cy = pred[1] + pred[3] / 2
    gt_cx = gt[0] + gt[2] / 2
    gt_cy = gt[1] + gt[3] / 2
    error = ((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2) ** 0.5
    center_errors.append(error)

# ----------------- Step 2：绘制 Success Plot -----------------
thresholds = np.arange(0, 1.01, 0.01)
success_rates = [np.mean(np.array(ious) >= t) for t in thresholds]

plt.figure(figsize=(9, 7))
plt.plot(thresholds, success_rates, label='Success Plot', color='#007ACC', marker='o', markersize=4, markevery=0.1)
plt.xlabel('IoU', fontsize=16)
plt.ylabel('Success Rate', fontsize=16)
plt.title('Success Plot (IoU)', fontsize=18, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig('success_plot.png', dpi=200)
plt.close()

# ----------------- Step 3：绘制 Precision Plot -----------------
error_thresholds = np.arange(0, 101)  # 像素误差从 0 到 100
precision_rates = [np.mean(np.array(center_errors) <= t) for t in error_thresholds]

plt.figure(figsize=(9, 7))
plt.plot(error_thresholds, precision_rates, label='Precision Plot', color='#E94F37', marker='s', markersize=4, markevery=10)
plt.xlabel('cle (pix)', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.title('Precision Plot (CLE)', fontsize=18, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig('precision_plot.png', dpi=200)
plt.close()

# 分别保存success_rates和precision_rates到CSV文件
success_df = pd.DataFrame({
    'IoU Threshold': thresholds,
    'Success Rate': success_rates
})
success_df.to_csv('success_rates.csv', index=False)
precision_df = pd.DataFrame({
    'Error Threshold (pixels)': error_thresholds,
    'Precision': precision_rates
})
precision_df.to_csv('precision_rates.csv', index=False)