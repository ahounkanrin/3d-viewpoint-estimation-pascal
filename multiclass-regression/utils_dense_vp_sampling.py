import psutil
import os
pid = psutil.Process(os.getpid())
pid.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7])

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

def quaternionLoss(predictions, labels):
    assert predictions.shape == labels.shape
    labels = tf.cast(labels, dtype=tf.float32)
    loss_batch = 1.0 - tf.math.square(tf.reduce_sum(predictions * labels, axis=-1))
    #loss = tf.math.reduce_mean(loss_batch)
    return loss_batch #, predictions

def quaternionLoss_old(predictions, labels):
    assert predictions.shape == labels.shape
    predNorm = tf.broadcast_to(tf.norm(predictions, axis=-1, keepdims=True), shape=predictions.shape)
    predictions = tf.divide(predictions, predNorm)
    labels = tf.cast(labels, dtype=tf.float32)
    loss_batch = 1 - tf.math.square(tf.reduce_sum(predictions * labels, axis=-1))
    #loss = tf.math.reduce_mean(loss_batch)
    return loss_batch #, predictions
  
