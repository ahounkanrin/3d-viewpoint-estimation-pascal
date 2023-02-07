import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import time
import pandas as pd
from utils import quaternion_angle
from matplotlib import pyplot as plt

print("INFO: Processing dataset...")
INPUT_SIZE = (224, 224)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs")
    parser.add_argument("--category", type=str, help="Category: aeroplane-boat-bus-chair-etc")
    #parser.add_argument("--level", type=str, help="Difficulty level: all-easy-nonDiff-nonOccl")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate")
    return parser.parse_args()

args = get_arguments()

def load_and_crop_img(imgpath_bbox):
    imgpath, bbox = imgpath_bbox
    xmin, ymin, xmax, ymax = tf.unstack(tf.cast(bbox, dtype=tf.int32))
    raw_img = tf.io.read_file(imgpath)
    img = tf.io.decode_jpeg(raw_img, channels=3)
    img = tf.image.crop_to_bounding_box(img, offset_height=ymin, offset_width=xmin, target_height=ymax-ymin, target_width=xmax-xmin)
    img = tf.image.resize_with_pad(img, target_height=INPUT_SIZE[0], target_width=INPUT_SIZE[1], method="nearest")
    return img

def load_and_resize_img(imgpath):
    raw_img = tf.io.read_file(imgpath)
    img = tf.io.decode_jpeg(raw_img, channels=3)
    img = tf.image.resize_with_pad(img, target_height=INPUT_SIZE[0], target_width=INPUT_SIZE[1], method="nearest")
    return img

def preprocess(imgpath, label):
    img = tf.map_fn(load_and_resize_img, imgpath, fn_output_signature=tf.uint8)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.divide(img, tf.constant(255.0, dtype=tf.float32))
    return img, label

# def test_step(images, labels):
#     #with tf.device("/gpu:1"):
#     raw_preds = model(images, training=False)
#     predNorm = tf.broadcast_to(tf.norm(raw_preds, axis=-1, keepdims=True), shape=raw_preds.shape)
#     predictions = tf.divide(raw_preds, predNorm)
#     loss = quaternionLoss(raw_preds, labels)
#     loss = tf.reduce_sum(loss)/images.shape[0]
#     return loss, predictions

# Load dataset
data_dir = "/scratch/hnkmah001/Datasets/PASCAL3D+_release1.1/"
test_df = pd.read_csv(data_dir +"Image_sets/{}/{}_test2.csv".format(args.category, args.category), sep=",")
img_path = test_df["image_path"].apply(lambda imgID: data_dir + imgID)
qw = test_df["qw"].astype(float)
qx = test_df["qx"].astype(float)
qy = test_df["qy"].astype(float)
qz = test_df["qz"].astype(float)
# xmin = df["xmin"].astype(int)
# ymin = df["ymin"].astype(int)
# xmax = df["xmax"].astype(int)
# ymax = df["ymax"].astype(int)
img_paths = np.array(img_path)
labels_quaternions =  tf.stack([qw, qx, qy, qz], axis=-1)
# labels_bbox = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
test_data = tf.data.Dataset.from_tensor_slices((img_paths, labels_quaternions)).batch(1)
test_data = test_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)


# Define model
# baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
#                                               include_top=False, weights="imagenet")

baseModel = tf.keras.applications.vgg16.VGG16(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")

inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dense(4, activation=None)(x)
outputs = tf.math.l2_normalize(x, axis=-1) # normalize output to get unit quaternions
model = tf.keras.Model(inputs=inputs, outputs=outputs) 

# Define cost function, optimizer and metrics
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/{}/checkpoints/".format(args.category)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

checkpoint.restore(manager.checkpoints[-1])
model.summary()

# Training loop
preds_list = []
gt = []

#test_logdir = "./logs/{}/bbox".format(args.category)
#test_summary_writer = tf.summary.create_file_writer(test_logdir)
step = 0
for test_images, test_labels in tqdm(test_data, desc="Testing"):                                  
    #loss, preds = test_step(test_images, test_labels)
    preds = model(test_images, training=False)
    preds_list.append(tf.squeeze(preds).numpy())
    gt.append(tf.squeeze(test_labels).numpy())
    step += 1
    #with test_summary_writer.as_default():
    #    tf.summary.image("image", test_images, step=step, max_outputs=1) 

errors = [quaternion_angle(gt[i], preds_list[i]).numpy() for i in range(len(gt))]

thresholds = np.array([theta for theta in range(0, 95, 10)])

print("Error = ", errors)
print("\n\nMedian Error = {:.4f}".format(np.median(np.array(errors))))
with open("result_{}_reg.txt".format(args.category), "w") as f:
    print("Error = ", errors, file=f)
    print("Median Error = {:.4f}".format(np.median(np.array(errors))), file=f)

acc_theta = []

for theta in thresholds:
    acc_bool = np.array([errors[i] <= theta  for i in range(len(errors))])
    acc = np.mean(acc_bool)
    acc_theta.append(acc)
    print("Accuracy at theta = {} is: {:.4f}".format(theta, acc))
    with open("result_{}.txt".format(args.category), "a") as f:
        print("Accuracy at theta = {} is: {:.4f}".format(theta, acc), file=f)

# plt.figure(figsize=[8, 5])
# plt.ylabel("Accuracy")
# plt.xlabel("Threshold (degrees)")
# plt.xticks(ticks=[i for i in range(0, 95, 10)])
# plt.yticks(ticks=[i/10 for i in range(21)])
# plt.plot(thresholds, acc_theta)

# # plt.legend(loc="lower right")
# plt.grid(True)
# plt.savefig("accuracy_reg_{}.png".format(args.category))
