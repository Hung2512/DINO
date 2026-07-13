from thyroid_dataset import ThyroidDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def cal_avg_uptake(img, bbox):
    bbox = [int(x) for x in bbox]
    ROI = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return np.sum(ROI)

def caculate_rsi(img, target):
    thyroid_mean = cal_avg_uptake(img, target['boxes'][0].tolist())
    shoulder_mean = cal_avg_uptake(img, target['boxes'][1].tolist())
    rsi = thyroid_mean/shoulder_mean
    return rsi

def show_plot_box(neg_rsis, pos_rsis):
    data = [list(neg_rsis.values()), list(pos_rsis.values())]
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(data, labels=[0, 1])
    plt.show()
    
def cal_point_biseral(neg_rsis, pos_rsis):
    features = list(neg_rsis.values()) + list(pos_rsis.values())
    features = np.asanyarray(features)
    labels = [0]*len(neg_rsis)+[1]*len(pos_rsis)
    labels = np.asanyarray(labels)

    return stats.pointbiserialr(features, labels)