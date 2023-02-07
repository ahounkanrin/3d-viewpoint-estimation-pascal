import os
import psutil
pid = psutil.Process(os.getpid())
pid.cpu_affinity([0, 1, 2, 3])

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
    parser.add_argument("--sigma", default=1, type=float, help="LSR sigma value")
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


def load_and_resize_img(imgpath):
    raw_img = tf.io.read_file(imgpath)
    img = tf.io.decode_jpeg(raw_img, channels=3)
    img = tf.image.resize_with_pad(img, target_height=INPUT_SIZE[0], target_width=INPUT_SIZE[1], method="nearest")
    #img = tf.image.resize(img, size=INPUT_SIZE, method="nearest")
    return img

def preprocess(imgpath, vp_label):
    img = tf.map_fn(load_and_resize_img, imgpath, fn_output_signature=tf.uint8)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.divide(img, tf.constant(255.0, dtype=tf.float32)) 
    return img, vp_label 

def encode_category_labels(category):
    cate_label_dict = dict([cate,index] for index, cate in enumerate(categories))
    return cate_label_dict[category]


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
outputs_aero = tf.keras.layers.Dense(4, activation=None, name="aero")(x)
outputs_bike = tf.keras.layers.Dense(4, activation=None, name="bike")(x)
outputs_boat = tf.keras.layers.Dense(4, activation=None, name="boat")(x)
outputs_bottle = tf.keras.layers.Dense(4, activation=None, name="bottle")(x)
outputs_bus = tf.keras.layers.Dense(4, activation=None, name="bus")(x)
outputs_car = tf.keras.layers.Dense(4, activation=None, name="car")(x)
outputs_chair = tf.keras.layers.Dense(4, activation=None, name="chair")(x)
outputs_table = tf.keras.layers.Dense(4, activation=None, name="table")(x)
outputs_mbike = tf.keras.layers.Dense(4, activation=None, name="mbike")(x)
outputs_sofa = tf.keras.layers.Dense(4, activation=None, name="sofa")(x)
outputs_train = tf.keras.layers.Dense(4, activation=None, name="train")(x)
outputs_tv = tf.keras.layers.Dense(4, activation=None, name="tv")(x)

outputs_aero = tf.math.l2_normalize(outputs_aero, axis=-1)
outputs_bike = tf.math.l2_normalize(outputs_bike, axis=-1)
outputs_boat = tf.math.l2_normalize(outputs_boat, axis=-1)
outputs_bottle = tf.math.l2_normalize(outputs_bottle, axis=-1)
outputs_bus = tf.math.l2_normalize(outputs_bus, axis=-1)
outputs_car = tf.math.l2_normalize(outputs_car, axis=-1)
outputs_chair = tf.math.l2_normalize(outputs_chair, axis=-1)
outputs_table = tf.math.l2_normalize(outputs_table, axis=-1)
outputs_mbike = tf.math.l2_normalize(outputs_mbike, axis=-1)
outputs_sofa = tf.math.l2_normalize(outputs_sofa, axis=-1)
outputs_train = tf.math.l2_normalize(outputs_train, axis=-1)
outputs_tv = tf.math.l2_normalize(outputs_tv, axis=-1)
outputs = [outputs_aero, outputs_bike, outputs_boat, outputs_bottle, outputs_bus, outputs_car,
            outputs_chair, outputs_table, outputs_mbike, outputs_sofa, outputs_train, outputs_tv]

model = tf.keras.Model(inputs=inputs, outputs=outputs) 
model.summary()

# Define cost function, optimizer and metrics
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
# checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/classification_multiclass/{}_samples/checkpoints/".format(args.samples)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/classification_multiclass_reg/checkpoints/"
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=args.patience+1)

checkpoint.restore(manager.checkpoints[0])
# tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
model.summary()

# Training loop
preds_list = []
gt = []

for images, gt_quat in tqdm(test_data.map(preprocess, 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Testing"):                                  
    predictions = model(images, training=False)
    preds = predictions[cate_dict[args.category]]
    preds_list.append(tf.squeeze(preds).numpy())
    gt.append(tf.squeeze(gt_quat).numpy())
 
errors = [quaternion_angle(gt[i], preds_list[i]) for i in range(len(gt))]

thresholds = np.array([theta for theta in range(0, 95, 10)])

print("\n\nMedian Error = {:.4f}".format(np.median(np.array(errors))))
with open("result_{}_cls1.txt".format(args.category), "w") as f:
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
