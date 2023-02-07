import os
os.environ["OMP_NUM_THREADS"] = "8" 
os.environ["OPENBLAS_NUM_THREADS"] = "8" 
os.environ["MKL_NUM_THREADS"] = "8" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" 
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import math
import sys
from types import new_class
import numpy as np
from scipy.linalg import logm
import tensorflow as tf
from scipy import ndimage
import pandas as pd 
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--category", type=str, help="Category: aeroplane-boat-bus-chair-etc")
    #parser.add_argument("--level", type=str, help="Difficulty level: all-easy-nonDiff-nonOccl")
    parser.add_argument("--patience", default=3, type=int, help="Early stopping patience")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--sigma", default=1, type=float, help="LSR sigma value")
    return parser.parse_args()

args = get_arguments()


epsilon =  1e-30
# nclasses = 20000

data_dir = "/scratch/hnkmah001/Datasets/PASCAL3D+_release1.1/"
train_df =  pd.read_csv(data_dir + "/Image_sets/{}/{}_train_cls.csv".format(args.category, args.category), sep=",")

qw = train_df["qw"].astype(float)
qx = train_df["qx"].astype(float)
qy = train_df["qy"].astype(float)
qz = train_df["qz"].astype(float)
q_class = train_df["gt_vp_labels"].astype(int)


quaternion = [[w,x,y,z] for (w,x,y,z) in zip(qw, qx, qy, qz)]
quaternion_dict = dict(zip(q_class, quaternion))
nclasses = len(set(q_class))
vals_qw = tf.constant([quaternion_dict[i][0] for i in range(nclasses)], dtype=tf.float32)
vals_qx = tf.constant([quaternion_dict[i][1] for i in range(nclasses)], dtype=tf.float32)
vals_qy = tf.constant([quaternion_dict[i][2] for i in range(nclasses)], dtype=tf.float32)
vals_qz = tf.constant([quaternion_dict[i][3] for i in range(nclasses)], dtype=tf.float32)
keys_tensor = tf.constant([i for i in range(nclasses)])


hashtable_qw = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qw), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qx), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qy), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qz), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)



def euler_to_quaternion(r):
    # print(r)
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    #print(qx, qy, qz, qw)
    return [qx, qy, qz, qw]

def quaternion_distance(q1, q2):
    #q2 = tf.cast(q2, dtype=q1.dtype)
    prod = tf.math.abs(tf.reduce_sum(q1 * q2, axis=-1))
    prod2 = tf.where(prod>1.0, x=tf.ones_like(prod), y=prod)
    dist = 1.0 - prod2
    return dist

def quaternion_get_weights(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw.lookup(k), hashtable_qx.lookup(k), hashtable_qy.lookup(k), hashtable_qz.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw.lookup(gt), hashtable_qx.lookup(gt), hashtable_qy.lookup(gt), hashtable_qz.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss

def quaternion_angle(q1, q2):
    prod = tf.math.abs(tf.reduce_sum(q1 * q2))
    if prod > 1.0:
        prod = tf.constant(1.0, dtype=tf.float64)
    theta = 2*tf.math.acos(prod)
    theta = 180.0*theta/np.pi
    return theta

# def quaternion_angle(q1, q2):
#     prod = tf.math.abs(tf.reduce_sum(tf.constant(q1) * tf.constant(q2)))
#     if prod > 1.0:
#         prod = tf.constant(1.0, dtype=tf.float64)
#     theta = 2*tf.math.acos(prod)
#     theta = 180.0*theta/np.pi
#     return theta

if __name__ == "__main__":
    weights = quaternion_get_weights(2000)
    import matplotlib.pyplot as plt 
    plt.figure()
    plt.plot(weights.numpy())
    plt.savefig("label_weights2000_sigma1000.png")
    print("figure saved")

# k = tf.constant([i for i in range(nclasses)])
# gt = tf.cast(4000, dtype=tf.int32) * tf.ones_like(k)
# print(hashtable_qw.lookup(gt))
# print(hashtable_qx.lookup(gt))
# print(hashtable_qy.lookup(gt))
# print(hashtable_qz.lookup(gt))

# print(len(quaternion_dict))
