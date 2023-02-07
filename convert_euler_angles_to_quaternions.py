import pandas as pd
import nibabel as nib
import numpy as np
import time
import cv2 as cv
from scipy.interpolate import interpn
from scipy.fft import fftn, fftshift, ifft2
from multiprocessing import Pool
import math
from tqdm import tqdm

np.random.seed(0)

# def quaternion_to_euler(q):
#     (x, y, z, w) = (q[0], q[1], q[2], q[3])
#     t0 = +2.0 * (w * x + y * z)
#     t1 = +1.0 - 2.0 * (x * x + y * y)
#     roll = math.atan2(t0, t1)
#     t2 = +2.0 * (w * y - z * x)
#     t2 = +1.0 if t2 > +1.0 else t2
#     t2 = -1.0 if t2 < -1.0 else t2
#     pitch = math.asin(t2)
#     t3 = +2.0 * (w * z + x * y)
#     t4 = +1.0 - 2.0 * (y * y + z * z)
#     yaw = math.atan2(t3, t4)
#     return [pitch, yaw, roll]
    
def Rx(theta):
    x = theta #* np.pi / 180.
    r11, r12, r13 = 1., 0. , 0.
    r21, r22, r23 = 0., np.cos(x), -np.sin(x)
    r31, r32, r33 = 0., np.sin(x), np.cos(x)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def Ry(theta):
    x = theta #* np.pi / 180.
    r11, r12, r13 = np.cos(x), 0., np.sin(x)
    r21, r22, r23 = 0., 1., 0.
    r31, r32, r33 = -np.sin(x), 0, np.cos(x)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def Rz(theta):
    x = theta #* np.pi / 180.
    r11, r12, r13 = np.cos(x), -np.sin(x), 0.
    r21, r22, r23 = np.sin(x), np.cos(x), 0.
    r31, r32, r33 = 0., 0., 1.
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def rotate_plane(plane, rotationMatrix):
	return np.matmul(rotationMatrix, plane)

def normalize(img):
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    img = 255 * img 
    return np.uint8(img)

def quaternion_to_euler(q):
    (x, y, z, w) = (q[0], q[1], q[2], q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [pitch, yaw, roll]

def euler_to_quaternion(r):
    # print(r)
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    #print(qx, qy, qz, qw)
    return [qx, qy, qz, qw]

if __name__ == "__main__":
    DATA_DIR = "/scratch/hnkmah001/Datasets/PASCAL3D+_release1.1/"
    classes = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car",
         "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]

    synsetID = ['02691156', '02834778', '02858304', '02876657', '02924116', '02958343', 
                '03001627', '04379243', '03790512', '04256520',  '04468005', '03211117']


    cate2synsetID = dict((x,y) for x,y in zip(classes, synsetID))
    synsetID2cat = dict((y,x) for (x,y) in zip(classes, synsetID))

    categories = ["train"] #, "test"
    #diffLevel = ["all", "easy", "nonDiff", "nonOccl"]
    for class_ in tqdm(classes):
        #print(class_)
        #for level in diffLevel:
        for category in categories:
            df = pd.read_csv(DATA_DIR + "Image_sets/{}/{}_{}_syn.csv".format(class_, class_, category), sep=",")
            azimuth = df["azimuth"]
            elevation = df["elevation"]
            theta = df["theta"]
            roll = np.pi/180.0 * np.array(azimuth)
            pitch = np.pi/180.0 * np.array(elevation)
            yaw = np.pi/180.0 * np.array(theta)
            qw= []
            qx = []
            qy = []
            qz = []
            for i in range(len(roll)):
                w, x, y, z = euler_to_quaternion([yaw[i], pitch[i], roll[i]])
                qw.append(w)
                qx.append(x)
                qy.append(y)
                qz.append(z)
            
            image_name = []

            df["qw"] = qw
            df["qx"] = qx
            df["qy"] = qy
            df["qz"] = qz
            df.to_csv(DATA_DIR + "Image_sets/{}/{}_{}_syn.csv".format(class_, class_, category), sep=",", index=False)
