import os
from traceback import print_tb
from unicodedata import category
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import time
import pandas as pd
from utils import quaternionLoss

INPUT_SIZE = (224, 224)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--category", type=str, help="Category: aeroplane-boat-bus-chair-etc")
    #parser.add_argument("--level", type=str, help="Difficulty level: all-easy-nonDiff-nonOccl")
    parser.add_argument("--patience", default=3, type=int, help="Early stopping patience")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Initial learning rate")
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

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        rawpreds = model(images, training=True)
        loss = quaternionLoss(rawpreds, labels)
        loss = tf.reduce_sum(loss) / images.shape[0]    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def test_step(images, labels):
    rawpreds = model(images, training=False)
    loss = quaternionLoss(rawpreds, labels)
    loss = tf.reduce_sum(loss) / images.shape[0]
    return loss

# Load dataset


data_dir = "/scratch/hnkmah001/Datasets/PASCAL3D+_release1.1/"
train_val_df = pd.read_csv("./Image_sets/{}/{}_train.csv".format(args.category, args.category), sep=",")
img_path = train_val_df["image_path"].apply(lambda imgID: data_dir + imgID)
qw = train_val_df["qw"].astype(float)
qx = train_val_df["qx"].astype(float)
qy = train_val_df["qy"].astype(float)
qz = train_val_df["qz"].astype(float)
# xmin = df["xmin"].astype(int)
# ymin = df["ymin"].astype(int)
# xmax = df["xmax"].astype(int)
# ymax = df["ymax"].astype(int)
img_paths = np.array(img_path)
labels_quaternions =  tf.stack([qw, qx, qy, qz], axis=-1)
# labels_bbox = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
data = tf.data.Dataset.from_tensor_slices((img_paths, labels_quaternions)).shuffle(len(img_paths))
val_data =  data.take(int(0.2 * len(img_path))).batch(args.batch_size)
train_data =  data.skip(int(0.2 * len(img_path))).batch(args.batch_size)
val_data = val_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_data = train_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Define model
# baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
#                                               include_top=False, weights="imagenet")

baseModel = tf.keras.applications.vgg16.VGG16(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")
#baseModel.trainable = False
inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dense(4, activation=None)(x)
outputs = tf.math.l2_normalize(x, axis=-1)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# Define cost function, optimizer and metrics
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/{}/checkpoints/".format(args.category)
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=args.patience+1)

# Save logs with TensorBoard Summary
train_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/{}/logs/train".format(args.category)
val_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/{}/logs/val".format(args.category)
train_summary_writer = tf.summary.create_file_writer(train_logdir)
val_summary_writer = tf.summary.create_file_writer(val_logdir)

# Training loop
step = 0
Lmin = tf.constant(np.inf)
for epoch in range(args.epochs):
    for images, labels in train_data:
        tic = time.time()
        train_loss = train_step(images, labels)
        
        if step % 100 == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss, step=step)
                tf.summary.image("image", images, step=step, max_outputs=1) 
        step += 1
        toc = time.time()
        print("Step {}: \t loss = {:.8f}  \t({:.2f} seconds/step)".format(step, train_loss, toc-tic))

    test_it = 0
    test_loss = 0.
    for test_images, test_labels in tqdm(val_data, desc="Validation"):
        test_loss += test_step(test_images, test_labels)
        test_it += 1
        
    test_loss = test_loss / tf.constant(test_it, dtype=tf.float32)
    with val_summary_writer.as_default():
        tf.summary.scalar("val_loss", test_loss, step=epoch)
        tf.summary.image("val_images", test_images, step=epoch, max_outputs=1)

    ckpt_path = manager.save()
    template = "Epoch {}, Validation Loss: {:.8f}, ckpt {}\n\n"
    print(template.format(epoch+1, test_loss, ckpt_path))
    

    # Early stopping
    if test_loss < Lmin:
        Lmin = test_loss
        Epoch_min = epoch
    if epoch >= Epoch_min + args.patience:
        print("Stopping after {} epochs...\n".format(epoch))
        break
    
    
