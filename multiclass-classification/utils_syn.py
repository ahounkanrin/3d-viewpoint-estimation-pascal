import psutil
import os
pid = psutil.Process(os.getpid())
pid.cpu_affinity([12, 13, 14, 15, 16, 17, 18, 19])

import tensorflow as tf
import math
import sys
from types import new_class
import numpy as np
from scipy.linalg import logm
# import tensorflow as tf
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
    parser.add_argument("--sigma", default=1000, type=float, help="LSR sigma value")
    parser.add_argument("--samples", default=50000, type=int, help="Number of sampling points on the viewpoint sphere")
    return parser.parse_args()

args = get_arguments()


epsilon =  1e-30
# nclasses = 20000

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

# hashtable_qw = [hashtable_qw_aero, hashtable_qw_bike, hashtable_qw_boat, hashtable_qw_bottle, hashtable_qw_bus, hashtable_qw_car, 
#                 hashtable_qw_chair, hashtable_qw_table, hashtable_qw_bike, hashtable_qw_sofa, hashtable_qw_train, hashtable_qw_tv]

# hashtable_qx = [hashtable_qx_aero, hashtable_qx_bike, hashtable_qx_boat, hashtable_qx_bottle, hashtable_qx_bus, hashtable_qx_car, 
#                 hashtable_qx_chair, hashtable_qx_table, hashtable_qx_bike, hashtable_qx_sofa, hashtable_qx_train, hashtable_qx_tv]

# hashtable_qy = [hashtable_qy_aero, hashtable_qy_bike, hashtable_qy_boat, hashtable_qy_bottle, hashtable_qy_bus, hashtable_qy_car, 
#                 hashtable_qy_chair, hashtable_qy_table, hashtable_qy_bike, hashtable_qy_sofa, hashtable_qy_train, hashtable_qy_tv]  

# hashtable_qz = [hashtable_qz_aero, hashtable_qz_bike, hashtable_qz_boat, hashtable_qz_bottle, hashtable_qz_bus, hashtable_qz_car, 
#                 hashtable_qz_chair, hashtable_qz_table, hashtable_qz_bike, hashtable_qz_sofa, hashtable_qz_train, hashtable_qz_tv]  
                           
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
def quaternion_angle(q1, q2):
    prod = tf.math.abs(tf.reduce_sum(tf.constant(q1) * tf.constant(q2)))
    if prod > 1.0:
        prod = tf.constant(1.0, dtype=tf.float64)
    theta = 2*tf.math.acos(prod)
    theta = 180.0*theta/np.pi
    return theta

def quaternion_get_weights_aero(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses_aero)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw_aero.lookup(k), hashtable_qx_aero.lookup(k), hashtable_qy_aero.lookup(k), hashtable_qz_aero.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw_aero.lookup(gt), hashtable_qx_aero.lookup(gt), hashtable_qy_aero.lookup(gt), hashtable_qz_aero.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    # weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy_aero(predictions, vp_labels):
    gt_classes = tf.argmax(vp_labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights_aero(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss

def quaternion_get_weights_bike(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses_bike)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw_bike.lookup(k), hashtable_qx_bike.lookup(k), hashtable_qy_bike.lookup(k), hashtable_qz_bike.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw_bike.lookup(gt), hashtable_qx_bike.lookup(gt), hashtable_qy_bike.lookup(gt), hashtable_qz_bike.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    # weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy_bike(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights_bike(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss

def quaternion_get_weights_boat(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses_boat)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw_boat.lookup(k), hashtable_qx_boat.lookup(k), hashtable_qy_boat.lookup(k), hashtable_qz_boat.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw_boat.lookup(gt), hashtable_qx_boat.lookup(gt), hashtable_qy_boat.lookup(gt), hashtable_qz_boat.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    # weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy_boat(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights_boat(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss

def quaternion_get_weights_bottle(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses_bottle)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw_bottle.lookup(k), hashtable_qx_bottle.lookup(k), hashtable_qy_bottle.lookup(k), hashtable_qz_bottle.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw_bottle.lookup(gt), hashtable_qx_bottle.lookup(gt), hashtable_qy_bottle.lookup(gt), hashtable_qz_bottle.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    # weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy_bottle(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights_bottle(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss

def quaternion_get_weights_bus(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses_bus)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw_bus.lookup(k), hashtable_qx_bus.lookup(k), hashtable_qy_bus.lookup(k), hashtable_qz_bus.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw_bus.lookup(gt), hashtable_qx_bus.lookup(gt), hashtable_qy_bus.lookup(gt), hashtable_qz_bus.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    # weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy_bus(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights_bus(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss


def quaternion_get_weights_car(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses_car)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw_car.lookup(k), hashtable_qx_car.lookup(k), hashtable_qy_car.lookup(k), hashtable_qz_car.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw_car.lookup(gt), hashtable_qx_car.lookup(gt), hashtable_qy_car.lookup(gt), hashtable_qz_car.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    # weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy_car(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights_car(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss

def quaternion_get_weights_chair(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses_chair)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw_chair.lookup(k), hashtable_qx_chair.lookup(k), hashtable_qy_chair.lookup(k), hashtable_qz_chair.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw_chair.lookup(gt), hashtable_qx_chair.lookup(gt), hashtable_qy_chair.lookup(gt), hashtable_qz_chair.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    # weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy_chair(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights_chair(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss

def quaternion_get_weights_table(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses_table)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw_table.lookup(k), hashtable_qx_table.lookup(k), hashtable_qy_table.lookup(k), hashtable_qz_table.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw_table.lookup(gt), hashtable_qx_table.lookup(gt), hashtable_qy_table.lookup(gt), hashtable_qz_table.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    # weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy_table(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights_table(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss

def quaternion_get_weights_mbike(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses_mbike)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw_mbike.lookup(k), hashtable_qx_mbike.lookup(k), hashtable_qy_mbike.lookup(k), hashtable_qz_mbike.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw_mbike.lookup(gt), hashtable_qx_mbike.lookup(gt), hashtable_qy_mbike.lookup(gt), hashtable_qz_mbike.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    # weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy_mbike(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights_mbike(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss

def quaternion_get_weights_sofa(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses_sofa)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw_sofa.lookup(k), hashtable_qx_sofa.lookup(k), hashtable_qy_sofa.lookup(k), hashtable_qz_sofa.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw_sofa.lookup(gt), hashtable_qx_sofa.lookup(gt), hashtable_qy_sofa.lookup(gt), hashtable_qz_sofa.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    # weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy_sofa(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights_sofa(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss

def quaternion_get_weights_train(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses_train)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw_train.lookup(k), hashtable_qx_train.lookup(k), hashtable_qy_train.lookup(k), hashtable_qz_train.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw_train.lookup(gt), hashtable_qx_train.lookup(gt), hashtable_qy_train.lookup(gt), hashtable_qz_train.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    # weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy_train(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights_train(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss

def quaternion_get_weights_tv(gt_class, sigma=args.sigma): 
    k = tf.constant([i for i in range(nclasses_tv)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw_tv.lookup(k), hashtable_qx_tv.lookup(k), hashtable_qy_tv.lookup(k), hashtable_qz_tv.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw_tv.lookup(gt), hashtable_qx_tv.lookup(gt), hashtable_qy_tv.lookup(gt), hashtable_qz_tv.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    # weights = tf.nn.softmax(weights, axis=-1)
    return weights

def quaternion_cross_entropy_tv(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights_tv(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(tf.math.maximum(predictions, tf.constant(epsilon)))
    loss = - weights * pred_log
    return loss

  
# if __name__ == "__main__":

#     weights = quaternion_get_weights(500)
#     import matplotlib.pyplot as plt 
#     plt.figure()
#     plt.scatter(np.arange(len(weights.numpy())), weights.numpy())
#     #plt.plot(weights.numpy())
#     plt.savefig("label_weight500_sigma100_lsr2.png")
    # print("figure saved")
    # plt.grid("on")
    # plt.show()
# k = tf.constant([i for i in range(nclasses)])
# gt = tf.cast(4000, dtype=tf.int32) * tf.ones_like(k)
# print(hashtable_qw.lookup(gt))
# print(hashtable_qx.lookup(gt))
# print(hashtable_qy.lookup(gt))
# print(hashtable_qz.lookup(gt))

# print(len(quaternion_dict))
