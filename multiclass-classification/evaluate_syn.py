import os
import psutil
pid = psutil.Process(os.getpid())
pid.cpu_affinity([0, 1, 2, 3])

# from logging import raiseExceptions

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import time
import pandas as pd
# from utils_dense_vp_sampling import euler_to_quaternion, quaternion_angle
from matplotlib import pyplot as plt


print("INFO: Processing dataset...")
INPUT_SIZE = (224, 224)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--category", type=str, help="Category: aeroplane-boat-bus-chair-etc")
    #parser.add_argument("--level", type=str, help="Difficulty level: all-easy-nonDiff-nonOccl")
    parser.add_argument("--patience", default=3, type=int, help="Early stopping patience")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=1e-1, type=float, help="Initial learning rate")
    parser.add_argument("--sigma", default=1000.0, type=float, help="LSR sigma value")
    parser.add_argument("--samples", default=50000, type=int, help="Number of sampling points on the viewpoint sphere")
    return parser.parse_args()

args = get_arguments()

def quaternion_angle(q1, q2):
    prod = tf.math.abs(tf.reduce_sum(tf.constant(q1) * tf.constant(q2)))
    if prod > 1.0:
        prod = tf.constant(1.0, dtype=tf.float64)
    theta = 2*tf.math.acos(prod)
    theta = 180.0*theta/np.pi
    return theta


categories = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", 
                "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]
cate_dict =  dict((j,i) for i,j in enumerate(categories))

data_dir = "/scratch/hnkmah001/Datasets/PASCAL3D+_release1.1/"

df_aero = pd.read_csv(data_dir + f"Image_sets/aeroplane/aeroplane_train_realPlusSyn.csv", sep=",")
qw_aero = df_aero["sampled_qw"].astype(float)
qx_aero = df_aero["sampled_qx"].astype(float)
qy_aero = df_aero["sampled_qy"].astype(float)
qz_aero = df_aero["sampled_qz"].astype(float)
q_class_aero = df_aero["sampled_class_label_data"].astype(int)


quaternion_aero = [[w,x,y,z] for (w,x,y,z) in zip(qw_aero, qx_aero, qy_aero, qz_aero)]
quaternion_dict_aero = dict(zip(q_class_aero, quaternion_aero))
nclasses_aero = len(set(q_class_aero))

# print("INFO", nclasses)
vals_qw_aero = tf.constant([quaternion_dict_aero[i][0] for i in range(nclasses_aero)], dtype=tf.float32)
vals_qx_aero = tf.constant([quaternion_dict_aero[i][1] for i in range(nclasses_aero)], dtype=tf.float32)
vals_qy_aero = tf.constant([quaternion_dict_aero[i][2] for i in range(nclasses_aero)], dtype=tf.float32)
vals_qz_aero = tf.constant([quaternion_dict_aero[i][3] for i in range(nclasses_aero)], dtype=tf.float32)
keys_tensor_aero = tf.constant([i for i in range(nclasses_aero)])


hashtable_qw_aero = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_aero, vals_qw_aero), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx_aero = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_aero, vals_qx_aero), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy_aero = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_aero, vals_qy_aero), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz_aero = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_aero, vals_qz_aero), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)


df_bike = pd.read_csv(data_dir + f"Image_sets/bicycle/bicycle_train_realPlusSyn.csv", sep=",")
qw_bike = df_bike["sampled_qw"].astype(float)
qx_bike = df_bike["sampled_qx"].astype(float)
qy_bike = df_bike["sampled_qy"].astype(float)
qz_bike = df_bike["sampled_qz"].astype(float)
q_class_bike = df_bike["sampled_class_label_data"].astype(int)


quaternion_bike = [[w,x,y,z] for (w,x,y,z) in zip(qw_bike, qx_bike, qy_bike, qz_bike)]
quaternion_dict_bike = dict(zip(q_class_bike, quaternion_bike))
nclasses_bike = len(set(q_class_bike))

# print("INFO", nclasses)
vals_qw_bike = tf.constant([quaternion_dict_bike[i][0] for i in range(nclasses_bike)], dtype=tf.float32)
vals_qx_bike = tf.constant([quaternion_dict_bike[i][1] for i in range(nclasses_bike)], dtype=tf.float32)
vals_qy_bike = tf.constant([quaternion_dict_bike[i][2] for i in range(nclasses_bike)], dtype=tf.float32)
vals_qz_bike = tf.constant([quaternion_dict_bike[i][3] for i in range(nclasses_bike)], dtype=tf.float32)
keys_tensor_bike = tf.constant([i for i in range(nclasses_bike)])


hashtable_qw_bike = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_bike, vals_qw_bike), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx_bike = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_bike, vals_qx_bike), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy_bike = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_bike, vals_qy_bike), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz_bike = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_bike, vals_qz_bike), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)


df_boat = pd.read_csv(data_dir + f"Image_sets/boat/boat_train_realPlusSyn.csv", sep=",")
qw_boat = df_boat["sampled_qw"].astype(float)
qx_boat = df_boat["sampled_qx"].astype(float)
qy_boat = df_boat["sampled_qy"].astype(float)
qz_boat = df_boat["sampled_qz"].astype(float)
q_class_boat = df_boat["sampled_class_label_data"].astype(int)


quaternion_boat = [[w,x,y,z] for (w,x,y,z) in zip(qw_boat, qx_boat, qy_boat, qz_boat)]
quaternion_dict_boat = dict(zip(q_class_boat, quaternion_boat))
nclasses_boat = len(set(q_class_boat))

# print("INFO", nclasses)
vals_qw_boat = tf.constant([quaternion_dict_boat[i][0] for i in range(nclasses_boat)], dtype=tf.float32)
vals_qx_boat = tf.constant([quaternion_dict_boat[i][1] for i in range(nclasses_boat)], dtype=tf.float32)
vals_qy_boat = tf.constant([quaternion_dict_boat[i][2] for i in range(nclasses_boat)], dtype=tf.float32)
vals_qz_boat = tf.constant([quaternion_dict_boat[i][3] for i in range(nclasses_boat)], dtype=tf.float32)
keys_tensor_boat = tf.constant([i for i in range(nclasses_boat)])


hashtable_qw_boat = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_boat, vals_qw_boat), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx_boat = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_boat, vals_qx_boat), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy_boat = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_boat, vals_qy_boat), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz_boat = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_boat, vals_qz_boat), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)


df_bottle = pd.read_csv(data_dir + f"Image_sets/bottle/bottle_train_realPlusSyn.csv", sep=",")
qw_bottle = df_bottle["sampled_qw"].astype(float)
qx_bottle = df_bottle["sampled_qx"].astype(float)
qy_bottle = df_bottle["sampled_qy"].astype(float)
qz_bottle = df_bottle["sampled_qz"].astype(float)
q_class_bottle = df_bottle["sampled_class_label_data"].astype(int)


quaternion_bottle = [[w,x,y,z] for (w,x,y,z) in zip(qw_bottle, qx_bottle, qy_bottle, qz_bottle)]
quaternion_dict_bottle = dict(zip(q_class_bottle, quaternion_bottle))
nclasses_bottle = len(set(q_class_bottle))

# print("INFO", nclasses)
vals_qw_bottle = tf.constant([quaternion_dict_bottle[i][0] for i in range(nclasses_bottle)], dtype=tf.float32)
vals_qx_bottle = tf.constant([quaternion_dict_bottle[i][1] for i in range(nclasses_bottle)], dtype=tf.float32)
vals_qy_bottle = tf.constant([quaternion_dict_bottle[i][2] for i in range(nclasses_bottle)], dtype=tf.float32)
vals_qz_bottle = tf.constant([quaternion_dict_bottle[i][3] for i in range(nclasses_bottle)], dtype=tf.float32)
keys_tensor_bottle = tf.constant([i for i in range(nclasses_bottle)])


hashtable_qw_bottle = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_bottle, vals_qw_bottle), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx_bottle = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_bottle, vals_qx_bottle), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy_bottle = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_bottle, vals_qy_bottle), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz_bottle = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_bottle, vals_qz_bottle), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)


df_bus = pd.read_csv(data_dir + f"Image_sets/bus/bus_train_realPlusSyn.csv", sep=",")
qw_bus = df_bus["sampled_qw"].astype(float)
qx_bus = df_bus["sampled_qx"].astype(float)
qy_bus = df_bus["sampled_qy"].astype(float)
qz_bus = df_bus["sampled_qz"].astype(float)
q_class_bus = df_bus["sampled_class_label_data"].astype(int)


quaternion_bus = [[w,x,y,z] for (w,x,y,z) in zip(qw_bus, qx_bus, qy_bus, qz_bus)]
quaternion_dict_bus = dict(zip(q_class_bus, quaternion_bus))
nclasses_bus = len(set(q_class_bus))

# print("INFO", nclasses)
vals_qw_bus = tf.constant([quaternion_dict_bus[i][0] for i in range(nclasses_bus)], dtype=tf.float32)
vals_qx_bus = tf.constant([quaternion_dict_bus[i][1] for i in range(nclasses_bus)], dtype=tf.float32)
vals_qy_bus = tf.constant([quaternion_dict_bus[i][2] for i in range(nclasses_bus)], dtype=tf.float32)
vals_qz_bus = tf.constant([quaternion_dict_bus[i][3] for i in range(nclasses_bus)], dtype=tf.float32)
keys_tensor_bus = tf.constant([i for i in range(nclasses_bus)])


hashtable_qw_bus = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_bus, vals_qw_bus), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx_bus = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_bus, vals_qx_bus), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy_bus = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_bus, vals_qy_bus), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz_bus = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_bus, vals_qz_bus), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)


df_car = pd.read_csv(data_dir + f"Image_sets/car/car_train_realPlusSyn.csv", sep=",")
qw_car = df_car["sampled_qw"].astype(float)
qx_car = df_car["sampled_qx"].astype(float)
qy_car = df_car["sampled_qy"].astype(float)
qz_car = df_car["sampled_qz"].astype(float)
q_class_car = df_car["sampled_class_label_data"].astype(int)


quaternion_car = [[w,x,y,z] for (w,x,y,z) in zip(qw_car, qx_car, qy_car, qz_car)]
quaternion_dict_car = dict(zip(q_class_car, quaternion_car))
nclasses_car = len(set(q_class_car))

# print("INFO", nclasses)
vals_qw_car = tf.constant([quaternion_dict_car[i][0] for i in range(nclasses_car)], dtype=tf.float32)
vals_qx_car = tf.constant([quaternion_dict_car[i][1] for i in range(nclasses_car)], dtype=tf.float32)
vals_qy_car = tf.constant([quaternion_dict_car[i][2] for i in range(nclasses_car)], dtype=tf.float32)
vals_qz_car = tf.constant([quaternion_dict_car[i][3] for i in range(nclasses_car)], dtype=tf.float32)
keys_tensor_car = tf.constant([i for i in range(nclasses_car)])


hashtable_qw_car = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_car, vals_qw_car), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx_car = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_car, vals_qx_car), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy_car = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_car, vals_qy_car), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz_car = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_car, vals_qz_car), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)

df_chair = pd.read_csv(data_dir + f"Image_sets/chair/chair_train_realPlusSyn.csv", sep=",")
qw_chair = df_chair["sampled_qw"].astype(float)
qx_chair = df_chair["sampled_qx"].astype(float)
qy_chair = df_chair["sampled_qy"].astype(float)
qz_chair = df_chair["sampled_qz"].astype(float)
q_class_chair = df_chair["sampled_class_label_data"].astype(int)


quaternion_chair = [[w,x,y,z] for (w,x,y,z) in zip(qw_chair, qx_chair, qy_chair, qz_chair)]
quaternion_dict_chair = dict(zip(q_class_chair, quaternion_chair))
nclasses_chair = len(set(q_class_chair))

# print("INFO", nclasses)
vals_qw_chair = tf.constant([quaternion_dict_chair[i][0] for i in range(nclasses_chair)], dtype=tf.float32)
vals_qx_chair = tf.constant([quaternion_dict_chair[i][1] for i in range(nclasses_chair)], dtype=tf.float32)
vals_qy_chair = tf.constant([quaternion_dict_chair[i][2] for i in range(nclasses_chair)], dtype=tf.float32)
vals_qz_chair = tf.constant([quaternion_dict_chair[i][3] for i in range(nclasses_chair)], dtype=tf.float32)
keys_tensor_chair = tf.constant([i for i in range(nclasses_chair)])


hashtable_qw_chair = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_chair, vals_qw_chair), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx_chair = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_chair, vals_qx_chair), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy_chair = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_chair, vals_qy_chair), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz_chair = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_chair, vals_qz_chair), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)


df_table = pd.read_csv(data_dir + f"Image_sets/diningtable/diningtable_train_realPlusSyn.csv", sep=",")
qw_table = df_table["sampled_qw"].astype(float)
qx_table = df_table["sampled_qx"].astype(float)
qy_table = df_table["sampled_qy"].astype(float)
qz_table = df_table["sampled_qz"].astype(float)
q_class_table = df_table["sampled_class_label_data"].astype(int)


quaternion_table = [[w,x,y,z] for (w,x,y,z) in zip(qw_table, qx_table, qy_table, qz_table)]
quaternion_dict_table = dict(zip(q_class_table, quaternion_table))
nclasses_table = len(set(q_class_table))

# print("INFO", nclasses)
vals_qw_table = tf.constant([quaternion_dict_table[i][0] for i in range(nclasses_table)], dtype=tf.float32)
vals_qx_table = tf.constant([quaternion_dict_table[i][1] for i in range(nclasses_table)], dtype=tf.float32)
vals_qy_table = tf.constant([quaternion_dict_table[i][2] for i in range(nclasses_table)], dtype=tf.float32)
vals_qz_table = tf.constant([quaternion_dict_table[i][3] for i in range(nclasses_table)], dtype=tf.float32)
keys_tensor_table = tf.constant([i for i in range(nclasses_table)])


hashtable_qw_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_table, vals_qw_table), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_table, vals_qx_table), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_table, vals_qy_table), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_table, vals_qz_table), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)


df_mbike = pd.read_csv(data_dir + f"Image_sets/motorbike/motorbike_train_realPlusSyn.csv", sep=",")
qw_mbike = df_mbike["sampled_qw"].astype(float)
qx_mbike = df_mbike["sampled_qx"].astype(float)
qy_mbike = df_mbike["sampled_qy"].astype(float)
qz_mbike = df_mbike["sampled_qz"].astype(float)
q_class_mbike = df_mbike["sampled_class_label_data"].astype(int)


quaternion_mbike = [[w,x,y,z] for (w,x,y,z) in zip(qw_mbike, qx_mbike, qy_mbike, qz_mbike)]
quaternion_dict_mbike = dict(zip(q_class_mbike, quaternion_mbike))
nclasses_mbike = len(set(q_class_mbike))

# print("INFO", nclasses)
vals_qw_mbike = tf.constant([quaternion_dict_mbike[i][0] for i in range(nclasses_mbike)], dtype=tf.float32)
vals_qx_mbike = tf.constant([quaternion_dict_mbike[i][1] for i in range(nclasses_mbike)], dtype=tf.float32)
vals_qy_mbike = tf.constant([quaternion_dict_mbike[i][2] for i in range(nclasses_mbike)], dtype=tf.float32)
vals_qz_mbike = tf.constant([quaternion_dict_mbike[i][3] for i in range(nclasses_mbike)], dtype=tf.float32)
keys_tensor_mbike = tf.constant([i for i in range(nclasses_mbike)])


hashtable_qw_mbike = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_mbike, vals_qw_mbike), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx_mbike = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_mbike, vals_qx_mbike), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy_mbike = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_mbike, vals_qy_mbike), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz_mbike = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_mbike, vals_qz_mbike), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)


df_sofa = pd.read_csv(data_dir + f"Image_sets/sofa/sofa_train_realPlusSyn.csv", sep=",")
qw_sofa = df_sofa["sampled_qw"].astype(float)
qx_sofa = df_sofa["sampled_qx"].astype(float)
qy_sofa = df_sofa["sampled_qy"].astype(float)
qz_sofa = df_sofa["sampled_qz"].astype(float)
q_class_sofa = df_sofa["sampled_class_label_data"].astype(int)


quaternion_sofa = [[w,x,y,z] for (w,x,y,z) in zip(qw_sofa, qx_sofa, qy_sofa, qz_sofa)]
quaternion_dict_sofa = dict(zip(q_class_sofa, quaternion_sofa))
nclasses_sofa = len(set(q_class_sofa))

# print("INFO", nclasses)
vals_qw_sofa = tf.constant([quaternion_dict_sofa[i][0] for i in range(nclasses_sofa)], dtype=tf.float32)
vals_qx_sofa = tf.constant([quaternion_dict_sofa[i][1] for i in range(nclasses_sofa)], dtype=tf.float32)
vals_qy_sofa = tf.constant([quaternion_dict_sofa[i][2] for i in range(nclasses_sofa)], dtype=tf.float32)
vals_qz_sofa = tf.constant([quaternion_dict_sofa[i][3] for i in range(nclasses_sofa)], dtype=tf.float32)
keys_tensor_sofa = tf.constant([i for i in range(nclasses_sofa)])


hashtable_qw_sofa = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_sofa, vals_qw_sofa), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx_sofa = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_sofa, vals_qx_sofa), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy_sofa = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_sofa, vals_qy_sofa), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz_sofa = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_sofa, vals_qz_sofa), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)

df_train = pd.read_csv(data_dir + f"Image_sets/train/train_train_realPlusSyn.csv", sep=",")
qw_train = df_train["sampled_qw"].astype(float)
qx_train = df_train["sampled_qx"].astype(float)
qy_train = df_train["sampled_qy"].astype(float)
qz_train = df_train["sampled_qz"].astype(float)
q_class_train = df_train["sampled_class_label_data"].astype(int)


quaternion_train = [[w,x,y,z] for (w,x,y,z) in zip(qw_train, qx_train, qy_train, qz_train)]
quaternion_dict_train = dict(zip(q_class_train, quaternion_train))
nclasses_train = len(set(q_class_train))

# print("INFO", nclasses)
vals_qw_train = tf.constant([quaternion_dict_train[i][0] for i in range(nclasses_train)], dtype=tf.float32)
vals_qx_train = tf.constant([quaternion_dict_train[i][1] for i in range(nclasses_train)], dtype=tf.float32)
vals_qy_train = tf.constant([quaternion_dict_train[i][2] for i in range(nclasses_train)], dtype=tf.float32)
vals_qz_train = tf.constant([quaternion_dict_train[i][3] for i in range(nclasses_train)], dtype=tf.float32)
keys_tensor_train = tf.constant([i for i in range(nclasses_train)])


hashtable_qw_train = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_train, vals_qw_train), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx_train = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_train, vals_qx_train), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy_train = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_train, vals_qy_train), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz_train = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_train, vals_qz_train), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)


df_tv = pd.read_csv(data_dir + f"Image_sets/tvmonitor/tvmonitor_train_realPlusSyn.csv", sep=",")
qw_tv = df_tv["sampled_qw"].astype(float)
qx_tv = df_tv["sampled_qx"].astype(float)
qy_tv = df_tv["sampled_qy"].astype(float)
qz_tv = df_tv["sampled_qz"].astype(float)
q_class_tv = df_tv["sampled_class_label_data"].astype(int)


quaternion_tv = [[w,x,y,z] for (w,x,y,z) in zip(qw_tv, qx_tv, qy_tv, qz_tv)]
quaternion_dict_tv = dict(zip(q_class_tv, quaternion_tv))
nclasses_tv = len(set(q_class_tv))

# print("INFO", nclasses)
vals_qw_tv = tf.constant([quaternion_dict_tv[i][0] for i in range(nclasses_tv)], dtype=tf.float32)
vals_qx_tv = tf.constant([quaternion_dict_tv[i][1] for i in range(nclasses_tv)], dtype=tf.float32)
vals_qy_tv = tf.constant([quaternion_dict_tv[i][2] for i in range(nclasses_tv)], dtype=tf.float32)
vals_qz_tv = tf.constant([quaternion_dict_tv[i][3] for i in range(nclasses_tv)], dtype=tf.float32)
keys_tensor_tv = tf.constant([i for i in range(nclasses_tv)])


hashtable_qw_tv = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_tv, vals_qw_tv), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx_tv = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_tv, vals_qx_tv), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy_tv = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_tv, vals_qy_tv), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz_tv = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor_tv, vals_qz_tv), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)

# q_class_aero = df_aero["sampled_class_label_data"].astype(int)
# nclasses_aero = len(set(q_class_aero))
# q_class_bike = df_bike["sampled_class_label_data"].astype(int)
# nclasses_bike = len(set(q_class_bike))
# q_class_boat = df_boat["sampled_class_label_data"].astype(int)
# nclasses_boat = len(set(q_class_boat))
# q_class_bottle = df_bottle["sampled_class_label_data"].astype(int)
# nclasses_bottle = len(set(q_class_bottle))
# q_class_bus = df_bus["sampled_class_label_data"].astype(int)
# nclasses_bus = len(set(q_class_bus))
# q_class_car = df_car["sampled_class_label_data"].astype(int)
# nclasses_car = len(set(q_class_car))
# q_class_chair = df_chair["sampled_class_label_data"].astype(int)
# nclasses_chair = len(set(q_class_chair))
# q_class_table = df_table["sampled_class_label_data"].astype(int)
# nclasses_table = len(set(q_class_table))
# q_class_mbike = df_mbike["sampled_class_label_data"].astype(int)
# nclasses_mbike = len(set(q_class_mbike))
# q_class_sofa = df_sofa["sampled_class_label_data"].astype(int)
# nclasses_sofa = len(set(q_class_sofa))
# q_class_train = df_train["sampled_class_label_data"].astype(int)
# nclasses_train = len(set(q_class_train))
# q_class_tv = df_tv["sampled_class_label_data"].astype(int)
# nclasses_tv = len(set(q_class_tv))


def load_and_resize_img(imgpath):
    raw_img = tf.io.read_file(imgpath)
    img = tf.io.decode_jpeg(raw_img, channels=3)
    img = tf.image.resize_with_pad(img, target_height=INPUT_SIZE[0], target_width=INPUT_SIZE[1], method="nearest")
    #img = tf.image.resize(img, size=INPUT_SIZE, method="nearest")
    return img

def one_hot_encoder_aero(label):
    return tf.one_hot(label, depth=nclasses_aero)

def one_hot_encoder_bike(label):
    return tf.one_hot(label, depth=nclasses_bike)

def one_hot_encoder_boat(label):
    return tf.one_hot(label, depth=nclasses_boat)

def one_hot_encoder_bottle(label):
    return tf.one_hot(label, depth=nclasses_bottle)

def one_hot_encoder_bus(label):
    return tf.one_hot(label, depth=nclasses_bus)

def one_hot_encoder_car(label):
    return tf.one_hot(label, depth=nclasses_car)

def one_hot_encoder_chair(label):
    return tf.one_hot(label, depth=nclasses_chair)

def one_hot_encoder_table(label):
    return tf.one_hot(label, depth=nclasses_table)

def one_hot_encoder_mbike(label):
    return tf.one_hot(label, depth=nclasses_mbike)

def one_hot_encoder_sofa(label):
    return tf.one_hot(label, depth=nclasses_sofa)

def one_hot_encoder_train(label):
    return tf.one_hot(label, depth=nclasses_train)

def one_hot_encoder_tv(label):
    return tf.one_hot(label, depth=nclasses_tv)

def preprocess(imgpath, vp_label):
    img = tf.map_fn(load_and_resize_img, imgpath, fn_output_signature=tf.uint8)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.divide(img, tf.constant(255.0, dtype=tf.float32)) 
    # print("INFO", cate_labels)
    # vp_label = tf.case([(tf.reduce_all(cate_labels == tf.constant(0, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_aero, vp_label, fn_output_signature=tf.float32)), 
    
    #                     (tf.reduce_all(cate_labels == tf.constant(1, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_bike, vp_label, fn_output_signature=tf.float32)), 
                        
    #                     (tf.reduce_all(cate_labels == tf.constant(2, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_boat, vp_label, fn_output_signature=tf.float32)),
    #                     (tf.reduce_all(cate_labels == tf.constant(3, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_bottle, vp_label, fn_output_signature=tf.float32)),
    #                     (tf.reduce_all(cate_labels == tf.constant(4, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_bus, vp_label, fn_output_signature=tf.float32)),
    #                     (tf.reduce_all(cate_labels == tf.constant(5, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_car, vp_label, fn_output_signature=tf.float32)),
    #                     (tf.reduce_all(cate_labels == tf.constant(6, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_chair, vp_label, fn_output_signature=tf.float32)),
    #                     (tf.reduce_all(cate_labels == tf.constant(7, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_table, vp_label, fn_output_signature=tf.float32)),
    #                     (tf.reduce_all(cate_labels == tf.constant(8, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_mbike, vp_label, fn_output_signature=tf.float32)),
    #                     (tf.reduce_all(cate_labels == tf.constant(9, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_sofa, vp_label, fn_output_signature=tf.float32)),
    #                     (tf.reduce_all(cate_labels == tf.constant(10, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_train, vp_label, fn_output_signature=tf.float32)),
    #                     (tf.reduce_all(cate_labels == tf.constant(11, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_tv, vp_label, fn_output_signature=tf.float32))
    #                     ])
    return img, vp_label # tf.one_hot(label, depth=nclasses)

def encode_category_labels(category):
    cate_label_dict = dict([cate,index] for index, cate in enumerate(categories))
    return cate_label_dict[category]

# data_dir = "/scratch/hnkmah001/Datasets/PASCAL3D+_release1.1/"
# train_df = pd.read_csv(data_dir + "Image_sets/{}/{}_train_cls_{}_quaternions.csv".format(args.category, args.category, args.samples), sep=",")

# train_qw = train_df["sampled_qw"].astype(float)
# train_qx = train_df["sampled_qx"].astype(float)
# train_qy = train_df["sampled_qy"].astype(float)
# train_qz = train_df["sampled_qz"].astype(float)
# q_class = train_df["sampled_class_label_data"].astype(int)

# quaternion = [[w,x,y,z] for (w,x,y,z) in zip(train_qw, train_qx, train_qy, train_qz)]
# quaternion_dict = dict(zip(q_class, quaternion))
# nclasses = len(set(q_class))
# vals_qw = tf.constant([quaternion_dict[i][0] for i in range(nclasses)], dtype=tf.float32)
# vals_qx = tf.constant([quaternion_dict[i][1] for i in range(nclasses)], dtype=tf.float32)
# vals_qy = tf.constant([quaternion_dict[i][2] for i in range(nclasses)], dtype=tf.float32)
# vals_qz = tf.constant([quaternion_dict[i][3] for i in range(nclasses)], dtype=tf.float32)
# keys_tensor = tf.constant([i for i in range(nclasses)])


# hashtable_qw = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qw), 
#                                             default_value=tf.constant(0.0, dtype=tf.float32), name=None)
# hashtable_qx = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qx), 
#                                             default_value=tf.constant(0.0, dtype=tf.float32), name=None)
# hashtable_qy = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qy), 
#                                             default_value=tf.constant(0.0, dtype=tf.float32), name=None)
# hashtable_qz = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qz), 
#                                             default_value=tf.constant(0.0, dtype=tf.float32), name=None)


# Load dataset
# img_path_bottle = df_bottle["image_path"].apply(lambda imgID: data_dir + imgID)
# categ_labels_bottle = df_bottle["class_label"].apply(encode_category_labels)
# img_path_list_bottle = tf.constant(np.array(img_path_bottle))
# vp_labels_list_bottle = tf.constant(np.array(q_class_bottle))
# cate_labels_list_bottle = tf.constant(np.array(categ_labels_bottle))


test_df = pd.read_csv(data_dir + "Image_sets/{}/{}_test2.csv".format(args.category, args.category), sep=",")
img_path = test_df["image_path"].apply(lambda imgID: data_dir + imgID)
# categ_labels =  test_df["class_label"].apply(encode_category_labels)
qw = test_df["qw"].astype(float)
qx = test_df["qx"].astype(float)
qy = test_df["qy"].astype(float)
qz = test_df["qz"].astype(float)
# = test_df["class"].astype(int)
gt_quaternions = tf.stack([qw, qx, qy, qz], axis=-1)
gt_quaternions = tf.cast(gt_quaternions, dtype=tf.float32)
# cate_labels_list = tf.constant(np.array(categ_labels))
img_path_list = tf.constant(np.array(img_path))
#labels_list = tf.constant(np.array(q_class))
test_data = tf.data.Dataset.from_tensor_slices((img_path_list, gt_quaternions)).batch(1) #, labels_list



# Define model
# baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
#                                               include_top=False, weights="imagenet")
baseModel = tf.keras.applications.vgg16.VGG16(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")
# baseModel.trainable = False
inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs_aero = tf.keras.layers.Dense(nclasses_aero, activation="softmax", name="aero")(x)
outputs_bike = tf.keras.layers.Dense(nclasses_bike, activation="softmax", name="bike")(x)
outputs_boat = tf.keras.layers.Dense(nclasses_boat, activation="softmax", name="boat")(x)
outputs_bottle = tf.keras.layers.Dense(nclasses_bottle, activation="softmax", name="bottle")(x)
outputs_bus = tf.keras.layers.Dense(nclasses_bus, activation="softmax", name="bus")(x)
outputs_car = tf.keras.layers.Dense(nclasses_car, activation="softmax", name="car")(x)
outputs_chair = tf.keras.layers.Dense(nclasses_chair, activation="softmax", name="chair")(x)
outputs_table = tf.keras.layers.Dense(nclasses_table, activation="softmax", name="table")(x)
outputs_mbike = tf.keras.layers.Dense(nclasses_mbike, activation="softmax", name="mbike")(x)
outputs_sofa = tf.keras.layers.Dense(nclasses_sofa, activation="softmax", name="sofa")(x)
outputs_train = tf.keras.layers.Dense(nclasses_train, activation="softmax", name="train")(x)
outputs_tv = tf.keras.layers.Dense(nclasses_tv, activation="softmax", name="tv")(x)
outputs = [outputs_aero, outputs_bike, outputs_boat, outputs_bottle, outputs_bus, outputs_car,
            outputs_chair, outputs_table, outputs_mbike, outputs_sofa, outputs_train, outputs_tv]

model = tf.keras.Model(inputs=inputs, outputs=outputs) 
# model.summary()

# Define cost function, optimizer and metrics
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
# checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/classification_multiclass/{}_samples/checkpoints/".format(args.samples)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/classification_multiclass_syn/{}_samples/checkpoints/".format( args.samples)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=args.patience+1)

checkpoint.restore(manager.checkpoints[-1])
# tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
model.summary()

# Training loop
preds_list = []
gt = []

for images, gt_quat in tqdm(test_data.map(preprocess, 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Testing"):                                  
    predictions = model(images, training=False)
    preds = predictions[cate_dict[args.category]]
    preds_list.append(tf.squeeze(tf.argmax(preds, axis=-1)).numpy())
    gt.append(tf.squeeze(gt_quat).numpy())
 
pred_tensor = tf.constant(preds_list)
#gt = tf.constant(gt)
if args.category == "aeroplane":
    pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw_aero.lookup(pred_tensor), hashtable_qx_aero.lookup(pred_tensor), hashtable_qy_aero.lookup(pred_tensor), hashtable_qz_aero.lookup(pred_tensor)
elif args.category == "bicycle":
    pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw_bike.lookup(pred_tensor), hashtable_qx_bike.lookup(pred_tensor), hashtable_qy_bike.lookup(pred_tensor), hashtable_qz_bike.lookup(pred_tensor)
elif args.category == "boat":
    pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw_boat.lookup(pred_tensor), hashtable_qx_boat.lookup(pred_tensor), hashtable_qy_boat.lookup(pred_tensor), hashtable_qz_boat.lookup(pred_tensor)
elif args.category == "bottle":
    pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw_bottle.lookup(pred_tensor), hashtable_qx_bottle.lookup(pred_tensor), hashtable_qy_bottle.lookup(pred_tensor), hashtable_qz_bottle.lookup(pred_tensor)
elif args.category == "bus":
    pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw_bus.lookup(pred_tensor), hashtable_qx_bus.lookup(pred_tensor), hashtable_qy_bus.lookup(pred_tensor), hashtable_qz_bus.lookup(pred_tensor)
elif args.category == "car":
    pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw_car.lookup(pred_tensor), hashtable_qx_car.lookup(pred_tensor), hashtable_qy_car.lookup(pred_tensor), hashtable_qz_car.lookup(pred_tensor)
elif args.category == "chair":
    pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw_chair.lookup(pred_tensor), hashtable_qx_chair.lookup(pred_tensor), hashtable_qy_chair.lookup(pred_tensor), hashtable_qz_chair.lookup(pred_tensor)
elif args.category == "diningtable":
    pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw_table.lookup(pred_tensor), hashtable_qx_table.lookup(pred_tensor), hashtable_qy_table.lookup(pred_tensor), hashtable_qz_table.lookup(pred_tensor)
elif args.category == "motorbike":
    pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw_mbike.lookup(pred_tensor), hashtable_qx_mbike.lookup(pred_tensor), hashtable_qy_mbike.lookup(pred_tensor), hashtable_qz_mbike.lookup(pred_tensor)
elif args.category == "sofa":
    pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw_sofa.lookup(pred_tensor), hashtable_qx_sofa.lookup(pred_tensor), hashtable_qy_sofa.lookup(pred_tensor), hashtable_qz_sofa.lookup(pred_tensor)
elif args.category == "train":
    pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw_train.lookup(pred_tensor), hashtable_qx_train.lookup(pred_tensor), hashtable_qy_train.lookup(pred_tensor), hashtable_qz_train.lookup(pred_tensor)
elif args.category == "tvmonitor":
    pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw_tv.lookup(pred_tensor), hashtable_qx_tv.lookup(pred_tensor), hashtable_qy_tv.lookup(pred_tensor), hashtable_qz_tv.lookup(pred_tensor)
else:
    raise ValueError(f"Unknown category provided: {args.category}")



pred_quaternions = tf.stack([pred_qw, pred_qx, pred_qy, pred_qz], axis=-1)
# gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw.lookup(gt), hashtable_qx.lookup(gt), hashtable_qy.lookup(gt), hashtable_qz.lookup(gt)
# gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)

errors = [quaternion_angle(gt[i], pred_quaternions[i]).numpy() for i in range(len(gt))]

thresholds = np.array([theta for theta in range(0, 95, 10)])

print("\n\nMedian Error = {:.4f}".format(np.median(np.array(errors))))
print("Error = ", errors)
with open("result_{}_syn.txt".format(args.category), "w") as f:
    print("Error = ", errors, file=f)
    print("Median Error = {:.4f}".format(np.median(np.array(errors))), file=f)

acc_theta = []

for theta in thresholds:
    acc_bool = np.array([errors[i] <= theta  for i in range(len(errors))])
    acc = np.mean(acc_bool)
    acc_theta.append(acc)
    print("Accuracy at theta = {} is: {:.4f}".format(theta, acc))
    with open("result_{}_cls1.txt".format(args.category), "a") as f:
        print("Accuracy at theta = {} is: {:.4f}".format(theta, acc), file=f)

# plt.figure(figsize=[8, 5])
# #plt.title("Accuracy of the CNN")
# #plt.figure()
# plt.ylabel("Accuracy")
# plt.xlabel("Threshold (degrees)")
# plt.xticks(ticks=[i for i in range(0, 95, 10)])
# plt.yticks(ticks=[i/10 for i in range(21)])
# plt.plot(thresholds, acc_theta)

# # plt.legend(loc="lower right")
# plt.grid(True)
# plt.savefig("accuracy_train1_test1.png")
