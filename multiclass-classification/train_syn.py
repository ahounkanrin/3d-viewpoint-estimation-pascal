import psutil
import os
pid = psutil.Process(os.getpid())
pid.cpu_affinity([12, 13, 14, 15, 16, 17, 18, 19])
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import time
import pandas as pd
# from utils_dense_vp_sampling import *
from utils_syn import quaternion_cross_entropy_aero
from utils_syn import quaternion_cross_entropy_bike
from utils_syn import quaternion_cross_entropy_boat
from utils_syn import quaternion_cross_entropy_bottle
from utils_syn import quaternion_cross_entropy_bus
from utils_syn import quaternion_cross_entropy_car
from utils_syn import quaternion_cross_entropy_chair
from utils_syn import quaternion_cross_entropy_table
from utils_syn import quaternion_cross_entropy_mbike
from utils_syn import quaternion_cross_entropy_sofa
from utils_syn import quaternion_cross_entropy_train
from utils_syn import quaternion_cross_entropy_tv

INPUT_SIZE = (224, 224)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
parser.add_argument("--category", type=str, help="Category: aeroplane-boat-bus-chair-etc")
parser.add_argument("--samples", type=int, default=50000, help="Number of viewpoints samples")
parser.add_argument("--sigma", default=1000, type=float, help="LSR sigma value")
parser.add_argument("--patience", default=3, type=int, help="Early stopping patience")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Initial learning rate")
args = parser.parse_args()

categories = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", 
                "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]

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

def preprocess(imgpath, cate_labels, vp_label):
    img = tf.map_fn(load_and_resize_img, imgpath, fn_output_signature=tf.uint8)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.divide(img, tf.constant(255.0, dtype=tf.float32)) 
    # print("INFO", cate_labels)
    vp_label = tf.case([(tf.reduce_all(cate_labels == tf.constant(0, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_aero, vp_label, fn_output_signature=tf.float32)), 
                        (tf.reduce_all(cate_labels == tf.constant(1, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_bike, vp_label, fn_output_signature=tf.float32)), 
                        (tf.reduce_all(cate_labels == tf.constant(2, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_boat, vp_label, fn_output_signature=tf.float32)),
                        (tf.reduce_all(cate_labels == tf.constant(3, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_bottle, vp_label, fn_output_signature=tf.float32)),
                        (tf.reduce_all(cate_labels == tf.constant(4, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_bus, vp_label, fn_output_signature=tf.float32)),
                        (tf.reduce_all(cate_labels == tf.constant(5, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_car, vp_label, fn_output_signature=tf.float32)),
                        (tf.reduce_all(cate_labels == tf.constant(6, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_chair, vp_label, fn_output_signature=tf.float32)),
                        (tf.reduce_all(cate_labels == tf.constant(7, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_table, vp_label, fn_output_signature=tf.float32)),
                        (tf.reduce_all(cate_labels == tf.constant(8, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_mbike, vp_label, fn_output_signature=tf.float32)),
                        (tf.reduce_all(cate_labels == tf.constant(9, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_sofa, vp_label, fn_output_signature=tf.float32)),
                        (tf.reduce_all(cate_labels == tf.constant(10, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_train, vp_label, fn_output_signature=tf.float32)),
                        (tf.reduce_all(cate_labels == tf.constant(11, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: tf.map_fn(one_hot_encoder_tv, vp_label, fn_output_signature=tf.float32))
                        ])
    return img, cate_labels, vp_label # tf.one_hot(label, depth=nclasses)

@tf.function
def train_step(images, cate_labels, vp_labels): 
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        (pred_aero, pred_bike, pred_boat, pred_bottle, pred_bus, pred_car,
        pred_chair, pred_table, pred_mbike, pred_sofa, pred_train, pred_tv) = predictions

        preds = tf.case([(tf.reduce_all(cate_labels == tf.constant(0, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_aero), 
                        (tf.reduce_all(cate_labels == tf.constant(1, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_bike), 
                        (tf.reduce_all(cate_labels == tf.constant(2, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_boat),
                        (tf.reduce_all(cate_labels == tf.constant(3, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_bottle),
                        (tf.reduce_all(cate_labels == tf.constant(4, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_bus),
                        (tf.reduce_all(cate_labels == tf.constant(5, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_car),
                        (tf.reduce_all(cate_labels == tf.constant(6, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_chair),
                        (tf.reduce_all(cate_labels == tf.constant(7, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_table),
                        (tf.reduce_all(cate_labels == tf.constant(8, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_mbike),
                        (tf.reduce_all(cate_labels == tf.constant(9, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_sofa),
                        (tf.reduce_all(cate_labels == tf.constant(10, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_train),
                        (tf.reduce_all(cate_labels == tf.constant(11, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_tv)
                        ])
        loss = tf.case([(tf.reduce_all(cate_labels == tf.constant(0, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_aero(pred_aero, vp_labels)), 
                        (tf.reduce_all(cate_labels == tf.constant(1, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_bike(pred_bike, vp_labels)), 
                        (tf.reduce_all(cate_labels == tf.constant(2, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_boat(pred_boat, vp_labels)),
                        (tf.reduce_all(cate_labels == tf.constant(3, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_bottle(pred_bottle, vp_labels)),
                        (tf.reduce_all(cate_labels == tf.constant(4, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_bus(pred_bus, vp_labels)),
                        (tf.reduce_all(cate_labels == tf.constant(5, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_car(pred_car, vp_labels)),
                        (tf.reduce_all(cate_labels == tf.constant(6, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_chair(pred_chair, vp_labels)),
                        (tf.reduce_all(cate_labels == tf.constant(7, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_table(pred_table, vp_labels)),
                        (tf.reduce_all(cate_labels == tf.constant(8, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_mbike(pred_mbike, vp_labels)),
                        (tf.reduce_all(cate_labels == tf.constant(9, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_sofa(pred_sofa, vp_labels)),
                        (tf.reduce_all(cate_labels == tf.constant(10, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_train(pred_train, vp_labels)),
                        (tf.reduce_all(cate_labels == tf.constant(11, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_tv(pred_tv, vp_labels))
                        ])
        # loss_aero = quaternion_cross_entropy_aero(pred_aero, cate_labels, vp_labels)
        # loss_bike = quaternion_cross_entropy_bike(pred_bike, cate_labels, vp_labels)
        # loss_boat = quaternion_cross_entropy_boat(pred_boat, cate_labels, vp_labels)
        # loss_bottle = quaternion_cross_entropy_bottle(pred_bottle, cate_labels, vp_labels)
        # loss_bus = quaternion_cross_entropy_bus(pred_bus, cate_labels, vp_labels)
        # loss_car = quaternion_cross_entropy_car(pred_car, cate_labels, vp_labels)
        # loss_chair = quaternion_cross_entropy_chair(pred_chair, cate_labels, vp_labels)
        # loss_table = quaternion_cross_entropy_table(pred_table, cate_labels, vp_labels)
        # loss_mbike = quaternion_cross_entropy_mbike(pred_mbike, cate_labels, vp_labels)
        # loss_sofa = quaternion_cross_entropy_sofa(pred_sofa, cate_labels, vp_labels)
        # loss_train = quaternion_cross_entropy_train(pred_train, cate_labels, vp_labels)
        # loss_tv = quaternion_cross_entropy_tv(pred_tv, cate_labels, vp_labels)
        # loss = loss_aero + loss_bike + loss_boat + loss_bottle + loss_bus + loss_car + loss_chair + loss_table\
        #         + loss_mbike + loss_sofa + loss_train + loss_tv
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=images.shape[0]) # strategy.num_replicas_in_sync*images.shape[0]
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(vp_labels, preds)
    return loss #, predictions

@tf.function
def test_step(images, cate_labels, vp_labels):
    predictions = model(images, training=False)
    (pred_aero, pred_bike, pred_boat, pred_bottle, pred_bus, pred_car,
    pred_chair, pred_table, pred_mbike, pred_sofa, pred_train, pred_tv) = predictions

    preds = tf.case([(tf.reduce_all(cate_labels == tf.constant(0, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_aero), 
                    (tf.reduce_all(cate_labels == tf.constant(1, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_bike), 
                    (tf.reduce_all(cate_labels == tf.constant(2, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_boat),
                    (tf.reduce_all(cate_labels == tf.constant(3, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_bottle),
                    (tf.reduce_all(cate_labels == tf.constant(4, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_bus),
                    (tf.reduce_all(cate_labels == tf.constant(5, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_car),
                    (tf.reduce_all(cate_labels == tf.constant(6, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_chair),
                    (tf.reduce_all(cate_labels == tf.constant(7, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_table),
                    (tf.reduce_all(cate_labels == tf.constant(8, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_mbike),
                    (tf.reduce_all(cate_labels == tf.constant(9, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_sofa),
                    (tf.reduce_all(cate_labels == tf.constant(10, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_train),
                    (tf.reduce_all(cate_labels == tf.constant(11, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: pred_tv)
                    ])
    test_loss = tf.case([(tf.reduce_all(cate_labels == tf.constant(0, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_aero(pred_aero, vp_labels)), 
                    (tf.reduce_all(cate_labels == tf.constant(1, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_bike(pred_bike, vp_labels)), 
                    (tf.reduce_all(cate_labels == tf.constant(2, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_boat(pred_boat, vp_labels)),
                    (tf.reduce_all(cate_labels == tf.constant(3, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_bottle(pred_bottle, vp_labels)),
                    (tf.reduce_all(cate_labels == tf.constant(4, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_bus(pred_bus, vp_labels)),
                    (tf.reduce_all(cate_labels == tf.constant(5, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_car(pred_car, vp_labels)),
                    (tf.reduce_all(cate_labels == tf.constant(6, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_chair(pred_chair, vp_labels)),
                    (tf.reduce_all(cate_labels == tf.constant(7, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_table(pred_table, vp_labels)),
                    (tf.reduce_all(cate_labels == tf.constant(8, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_mbike(pred_mbike, vp_labels)),
                    (tf.reduce_all(cate_labels == tf.constant(9, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_sofa(pred_sofa, vp_labels)),
                    (tf.reduce_all(cate_labels == tf.constant(10, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_train(pred_train, vp_labels)),
                    (tf.reduce_all(cate_labels == tf.constant(11, shape=cate_labels.shape, dtype=cate_labels.dtype)), lambda: quaternion_cross_entropy_tv(pred_tv, vp_labels)),
                    ])
    # test_loss = quaternion_cross_entropy(predictions, labels)
    test_loss = tf.reduce_sum(test_loss, axis=-1)  
    test_loss = tf.nn.compute_average_loss(test_loss, global_batch_size=images.shape[0]) #strategy.num_replicas_in_sync*images.shape[0]
    test_accuracy.update_state(vp_labels, preds)
    return test_loss

def encode_category_labels(category):
    cate_label_dict = dict([cate,index] for index, cate in enumerate(categories))
    return cate_label_dict[category]

# Load dataset
data_dir = "/scratch/hnkmah001/Datasets/PASCAL3D+_release1.1/"
# train_val_df = pd.read_csv(data_dir + "Image_sets/{}/{}_train_cls_{}.csv".format(args.category, args.category, args.samples), sep=",")
# img_path = train_val_df["image_path"].apply(lambda imgID: data_dir + imgID)
# q_class = train_val_df["sampled_class_label_data"].astype(int)
# nclasses = len(set(q_class))
# img_path_list = tf.constant(np.array(img_path))
# labels_list = tf.constant(np.array(q_class))
# dataset = tf.data.Dataset.from_tensor_slices((img_path_list, labels_list)).shuffle(len(img_path_list))
# val_data = dataset.take(int(0.2 * len(img_path_list))).batch(args.batch_size)
# train_data = dataset.skip(int(0.2 * len(img_path_list))).batch(args.batch_size)

df_aero = pd.read_csv(data_dir + f"Image_sets/aeroplane/aeroplane_train_realPlusSyn.csv", sep=",")
df_aero = df_aero.sample(frac=1)
df_bike = pd.read_csv(data_dir + f"Image_sets/bicycle/bicycle_train_realPlusSyn.csv", sep=",")
df_bike = df_bike.sample(frac=1)
df_boat = pd.read_csv(data_dir + f"Image_sets/boat/boat_train_realPlusSyn.csv", sep=",")
df_boat =  df_boat.sample(frac=1)
df_bottle = pd.read_csv(data_dir + f"Image_sets/bottle/bottle_train_realPlusSyn.csv", sep=",")
df_bottle = df_bottle.sample(frac=1)
df_bus = pd.read_csv(data_dir + f"Image_sets/bus/bus_train_realPlusSyn.csv", sep=",")
df_bus =  df_bus.sample(frac=1)
df_car = pd.read_csv(data_dir + f"Image_sets/car/car_train_realPlusSyn.csv", sep=",")
df_car = df_car.sample(frac=1)
df_chair = pd.read_csv(data_dir + f"Image_sets/chair/chair_train_realPlusSyn.csv", sep=",")
df_chair = df_chair.sample(frac=1)
df_table = pd.read_csv(data_dir + f"Image_sets/diningtable/diningtable_train_realPlusSyn.csv", sep=",")
df_table = df_table.sample(frac=1)
df_mbike = pd.read_csv(data_dir + f"Image_sets/motorbike/motorbike_train_realPlusSyn.csv", sep=",")
df_mbike = df_mbike.sample(frac=1)
df_sofa = pd.read_csv(data_dir + f"Image_sets/sofa/sofa_train_realPlusSyn.csv", sep=",")
df_sofa = df_sofa.sample(frac=1)
df_train = pd.read_csv(data_dir + f"Image_sets/train/train_train_realPlusSyn.csv", sep=",")
df_train =  df_train.sample(frac=1)
df_tv = pd.read_csv(data_dir + f"Image_sets/tvmonitor/tvmonitor_train_realPlusSyn.csv", sep=",")
df_tv = df_tv.sample(frac=1)

q_class_aero = df_aero["sampled_class_label_data"].astype(int)
nclasses_aero = len(set(q_class_aero))
q_class_bike = df_bike["sampled_class_label_data"].astype(int)
nclasses_bike = len(set(q_class_bike))
q_class_boat = df_boat["sampled_class_label_data"].astype(int)
nclasses_boat = len(set(q_class_boat))
q_class_bottle = df_bottle["sampled_class_label_data"].astype(int)
nclasses_bottle = len(set(q_class_bottle))
q_class_bus = df_bus["sampled_class_label_data"].astype(int)
nclasses_bus = len(set(q_class_bus))
q_class_car = df_car["sampled_class_label_data"].astype(int)
nclasses_car = len(set(q_class_car))
q_class_chair = df_chair["sampled_class_label_data"].astype(int)
nclasses_chair = len(set(q_class_chair))
q_class_table = df_table["sampled_class_label_data"].astype(int)
nclasses_table = len(set(q_class_table))
q_class_mbike = df_mbike["sampled_class_label_data"].astype(int)
nclasses_mbike = len(set(q_class_mbike))
q_class_sofa = df_sofa["sampled_class_label_data"].astype(int)
nclasses_sofa = len(set(q_class_sofa))
q_class_train = df_train["sampled_class_label_data"].astype(int)
nclasses_train = len(set(q_class_train))
q_class_tv = df_tv["sampled_class_label_data"].astype(int)
nclasses_tv = len(set(q_class_tv))


img_path_aero = df_aero["image_path"].apply(lambda imgID: data_dir + imgID)
categ_labels_aero = df_aero["class_label"].apply(encode_category_labels)
img_path_list_aero = tf.constant(np.array(img_path_aero))
vp_labels_list_aero = tf.constant(np.array(q_class_aero))
cate_labels_list_aero = tf.constant(np.array(categ_labels_aero))
ds_aero = tf.data.Dataset.from_tensor_slices((img_path_list_aero, cate_labels_list_aero, 
                            vp_labels_list_aero)) #.shuffle(len(img_path_list_aero))
ds_aero = ds_aero.batch(args.batch_size, drop_remainder=True)
ds_aero_val = ds_aero.take(int(0.2 * len(list(ds_aero.as_numpy_iterator()))))
ds_aero_train = ds_aero.skip(int(0.2 * len(list(ds_aero.as_numpy_iterator()))))

img_path_bike = df_bike["image_path"].apply(lambda imgID: data_dir + imgID)
categ_labels_bike = df_bike["class_label"].apply(encode_category_labels)
img_path_list_bike = tf.constant(np.array(img_path_bike))
vp_labels_list_bike = tf.constant(np.array(q_class_bike))
cate_labels_list_bike = tf.constant(np.array(categ_labels_bike))
ds_bike = tf.data.Dataset.from_tensor_slices((img_path_list_bike, cate_labels_list_bike, 
                            vp_labels_list_bike)) #.shuffle(len(img_path_list_bike))
ds_bike = ds_bike.batch(args.batch_size, drop_remainder=True)
ds_bike_val = ds_bike.take(int(0.2 * len(list(ds_bike.as_numpy_iterator()))))
ds_bike_train = ds_bike.skip(int(0.2 * len(list(ds_bike.as_numpy_iterator()))))

img_path_boat = df_boat["image_path"].apply(lambda imgID: data_dir + imgID)
categ_labels_boat = df_boat["class_label"].apply(encode_category_labels)
img_path_list_boat = tf.constant(np.array(img_path_boat))
vp_labels_list_boat = tf.constant(np.array(q_class_boat))
cate_labels_list_boat = tf.constant(np.array(categ_labels_boat))
ds_boat = tf.data.Dataset.from_tensor_slices((img_path_list_boat, cate_labels_list_boat, 
                            vp_labels_list_boat)) #.shuffle(len(img_path_list_boat))
ds_boat = ds_boat.batch(args.batch_size, drop_remainder=True)
ds_boat_val = ds_boat.take(int(0.2 * len(list(ds_boat.as_numpy_iterator()))))
ds_boat_train = ds_boat.skip(int(0.2 * len(list(ds_boat.as_numpy_iterator()))))

img_path_bottle = df_bottle["image_path"].apply(lambda imgID: data_dir + imgID)
categ_labels_bottle = df_bottle["class_label"].apply(encode_category_labels)
img_path_list_bottle = tf.constant(np.array(img_path_bottle))
vp_labels_list_bottle = tf.constant(np.array(q_class_bottle))
cate_labels_list_bottle = tf.constant(np.array(categ_labels_bottle))
ds_bottle = tf.data.Dataset.from_tensor_slices((img_path_list_bottle, cate_labels_list_bottle, 
                            vp_labels_list_bottle)) #.shuffle(len(img_path_list_bottle))
ds_bottle = ds_bottle.batch(args.batch_size, drop_remainder=True)
ds_bottle_val = ds_bottle.take(int(0.2 * len(list(ds_bottle.as_numpy_iterator()))))
ds_bottle_train = ds_bottle.skip(int(0.2 * len(list(ds_bottle.as_numpy_iterator()))))

img_path_bus = df_bus["image_path"].apply(lambda imgID: data_dir + imgID)
categ_labels_bus = df_bus["class_label"].apply(encode_category_labels)
img_path_list_bus = tf.constant(np.array(img_path_bus))
vp_labels_list_bus = tf.constant(np.array(q_class_bus))
cate_labels_list_bus = tf.constant(np.array(categ_labels_bus))
ds_bus = tf.data.Dataset.from_tensor_slices((img_path_list_bus, cate_labels_list_bus, 
                            vp_labels_list_bus))#.shuffle(len(img_path_list_bus))
ds_bus = ds_bus.batch(args.batch_size, drop_remainder=True)
ds_bus_val = ds_bus.take(int(0.2 * len(list(ds_bus.as_numpy_iterator()))))
ds_bus_train = ds_bus.skip(int(0.2 * len(list(ds_bus.as_numpy_iterator()))))

img_path_car = df_car["image_path"].apply(lambda imgID: data_dir + imgID)
categ_labels_car = df_car["class_label"].apply(encode_category_labels)
img_path_list_car = tf.constant(np.array(img_path_car))
vp_labels_list_car = tf.constant(np.array(q_class_car))
cate_labels_list_car = tf.constant(np.array(categ_labels_car))
ds_car = tf.data.Dataset.from_tensor_slices((img_path_list_car, cate_labels_list_car, 
                            vp_labels_list_car))#.shuffle(len(img_path_list_car))
ds_car = ds_car.batch(args.batch_size, drop_remainder=True)
ds_car_val = ds_car.take(int(0.2 * len(list(ds_car.as_numpy_iterator()))))
ds_car_train = ds_car.skip(int(0.2 * len(list(ds_car.as_numpy_iterator()))))

img_path_chair = df_chair["image_path"].apply(lambda imgID: data_dir + imgID)
categ_labels_chair = df_chair["class_label"].apply(encode_category_labels)
img_path_list_chair = tf.constant(np.array(img_path_chair))
vp_labels_list_chair = tf.constant(np.array(q_class_chair))
cate_labels_list_chair = tf.constant(np.array(categ_labels_chair))
ds_chair = tf.data.Dataset.from_tensor_slices((img_path_list_chair, cate_labels_list_chair, 
                            vp_labels_list_chair))#.shuffle(len(img_path_list_chair))
ds_chair = ds_chair.batch(args.batch_size, drop_remainder=True)
ds_chair_val = ds_chair.take(int(0.2 * len(list(ds_chair.as_numpy_iterator()))))
ds_chair_train = ds_chair.skip(int(0.2 * len(list(ds_chair.as_numpy_iterator()))))

img_path_table = df_table["image_path"].apply(lambda imgID: data_dir + imgID)
categ_labels_table = df_table["class_label"].apply(encode_category_labels)
img_path_list_table = tf.constant(np.array(img_path_table))
vp_labels_list_table = tf.constant(np.array(q_class_table))
cate_labels_list_table = tf.constant(np.array(categ_labels_table))
ds_table = tf.data.Dataset.from_tensor_slices((img_path_list_table, cate_labels_list_table, 
                            vp_labels_list_table))#.shuffle(len(img_path_list_table))
ds_table = ds_table.batch(args.batch_size, drop_remainder=True)
ds_table_val = ds_table.take(int(0.2 * len(list(ds_table.as_numpy_iterator()))))
ds_table_train = ds_table.skip(int(0.2 * len(list(ds_table.as_numpy_iterator()))))

img_path_mbike = df_mbike["image_path"].apply(lambda imgID: data_dir + imgID)
categ_labels_mbike = df_mbike["class_label"].apply(encode_category_labels)
img_path_list_mbike = tf.constant(np.array(img_path_mbike))
vp_labels_list_mbike = tf.constant(np.array(q_class_mbike))
cate_labels_list_mbike = tf.constant(np.array(categ_labels_mbike))
ds_mbike = tf.data.Dataset.from_tensor_slices((img_path_list_mbike, cate_labels_list_mbike, 
                            vp_labels_list_mbike))#.shuffle(len(img_path_list_mbike))
ds_mbike = ds_mbike.batch(args.batch_size, drop_remainder=True)
ds_mbike_val = ds_mbike.take(int(0.2 * len(list(ds_mbike.as_numpy_iterator()))))
ds_mbike_train = ds_mbike.skip(int(0.2 * len(list(ds_mbike.as_numpy_iterator()))))

img_path_sofa = df_sofa["image_path"].apply(lambda imgID: data_dir + imgID)
categ_labels_sofa = df_sofa["class_label"].apply(encode_category_labels)
img_path_list_sofa = tf.constant(np.array(img_path_sofa))
vp_labels_list_sofa = tf.constant(np.array(q_class_sofa))
cate_labels_list_sofa = tf.constant(np.array(categ_labels_sofa))
ds_sofa = tf.data.Dataset.from_tensor_slices((img_path_list_sofa, cate_labels_list_sofa, 
                            vp_labels_list_sofa))#.shuffle(len(img_path_list_sofa))
ds_sofa = ds_sofa.batch(args.batch_size, drop_remainder=True)
ds_sofa_val = ds_sofa.take(int(0.2 * len(list(ds_sofa.as_numpy_iterator()))))
ds_sofa_train = ds_sofa.skip(int(0.2 * len(list(ds_sofa.as_numpy_iterator()))))

img_path_train = df_train["image_path"].apply(lambda imgID: data_dir + imgID)
categ_labels_train = df_train["class_label"].apply(encode_category_labels)
img_path_list_train = tf.constant(np.array(img_path_train))
vp_labels_list_train = tf.constant(np.array(q_class_train))
cate_labels_list_train = tf.constant(np.array(categ_labels_train))
ds_train = tf.data.Dataset.from_tensor_slices((img_path_list_train, cate_labels_list_train, 
                            vp_labels_list_train))#.shuffle(len(img_path_list_train))    
ds_train = ds_train.batch(args.batch_size, drop_remainder=True)
ds_train_val = ds_train.take(int(0.2 * len(list(ds_train.as_numpy_iterator()))))
ds_train_train = ds_train.skip(int(0.2 * len(list(ds_train.as_numpy_iterator()))))

img_path_tv = df_tv["image_path"].apply(lambda imgID: data_dir + imgID)
categ_labels_tv = df_tv["class_label"].apply(encode_category_labels)
img_path_list_tv = tf.constant(np.array(img_path_tv))
vp_labels_list_tv = tf.constant(np.array(q_class_tv))
cate_labels_list_tv = tf.constant(np.array(categ_labels_tv))
ds_tv = tf.data.Dataset.from_tensor_slices((img_path_list_tv, cate_labels_list_tv, 
                            vp_labels_list_tv))#.shuffle(len(img_path_list_tv))
ds_tv = ds_tv.batch(args.batch_size, drop_remainder=True)
ds_tv_val = ds_tv.take(int(0.2 * len(list(ds_tv.as_numpy_iterator()))))
ds_tv_train = ds_tv.skip(int(0.2 * len(list(ds_tv.as_numpy_iterator()))))

data_train = ds_aero_train.concatenate(ds_bike_train).concatenate(ds_boat_train).concatenate(ds_bottle_train).concatenate(ds_bus_train).concatenate(ds_car_train)
data_train = data_train.concatenate(ds_chair_train).concatenate(ds_table_train).concatenate(ds_mbike_train).concatenate(ds_sofa_train)
data_train = data_train.concatenate(ds_train_train).concatenate(ds_tv_train)  

data_val = ds_aero_val.concatenate(ds_bike_val).concatenate(ds_boat_val).concatenate(ds_bottle_val).concatenate(ds_bus_val).concatenate(ds_car_val)
data_val = data_val.concatenate(ds_chair_val).concatenate(ds_table_val).concatenate(ds_mbike_val).concatenate(ds_sofa_val)
data_val = data_val.concatenate(ds_train_val).concatenate(ds_tv_val)  

data_train =  data_train.shuffle(len(list(data_train.as_numpy_iterator()))).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
data_val =  data_val.shuffle(len(list(data_val.as_numpy_iterator()))).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# data_train = data_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# # df = pd.concat([df_aero, df_bike, df_boat, df_bottle, df_bus, df_car, df_chair, df_table, df_mbike, df_sofa, df_train, df_tv], ignore_index=True)
# img_path_aero = df_aero["image_path"].apply(lambda imgID: data_dir + imgID)
# # q_class_aero = df_aero["sampled_class_label_data"].astype(int)
# categ_labels_aero = df_aero["class_label"].apply(encode_category_labels)
# img_path_list_aero = tf.constant(np.array(img_path_aero))
# vp_labels_list_aero = tf.constant(np.array(q_class_aero))
# cate_labels_list_aero = tf.constant(np.array(categ_labels_aero))
# ds_aero = tf.data.Dataset.from_tensor_slices((img_path_list_aero, cate_labels_list_aero, 
#                             vp_labels_list_aero)).shuffle(len(img_path_list_aero))


# val_data = dataset.take(int(0.2 * len(img_path_list))).batch(args.batch_size)
# train_data = dataset.skip(int(0.2 * len(img_path_list))).batch(args.batch_size)


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

# images = tf.random.uniform(shape=(32, 224, 224, 3), dtype=tf.float32)
# predictions = model(images, training=True)
# print(tf.tensordot(predictions[0], tf.one_hot(0, depth=32), axes=0))


# Define cost function, optimizer and metrics
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/classification_multiclass_syn/{}_samples/checkpoints/".format(args.samples)
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=args.patience+1)

if len(manager.checkpoints) > 0:
    checkpoint.restore(manager.checkpoints[-1])
    print("Pretrained model restored.")
model.summary()
# Save logs with TensorBoard Summary
train_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/classification_multiclass_syn/{}_samples/logs/train".format(args.samples)
val_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/classification_multiclass_syn/{}_samples/logs/val".format(args.samples)
train_summary_writer = tf.summary.create_file_writer(train_logdir)
val_summary_writer = tf.summary.create_file_writer(val_logdir)


# Training loop

step = 0
Lmin = tf.constant(np.inf)
for epoch in range(args.epochs):
    for images, categ, vp_labels in data_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        tic = time.time()
        train_loss = train_step(images, categ, vp_labels)
        #print(gradients)
        # train_step(images, labels)
        if step % 100 == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss, step=step)
                tf.summary.scalar("accuracy", train_accuracy.result(), step=step)
                tf.summary.image("image", images, step=step, max_outputs=1) 
        toc = time.time()
        step += 1
        
        print("Step {}: \t loss = {:.8f} \t acc = {:.6f} \t ({:.2f} seconds/step)".format(step, 
                train_loss, train_accuracy.result(), toc-tic))
        train_accuracy.reset_states()  
        # if step % 10 ==0:
        #     break

    test_it = 0
    test_loss = 0.
    for test_images, test_categ, test_vp_labels in tqdm(data_val.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE), 
                                                        desc="Validation"):
        test_loss += test_step(test_images, test_categ, test_vp_labels)
        test_it +=1

        # if test_it == 10:
        #     break
    test_loss = test_loss/tf.constant(test_it, dtype=tf.float32)
    with val_summary_writer.as_default():
        tf.summary.scalar("val_loss", test_loss, step=epoch)
        tf.summary.scalar("val_accuracy", test_accuracy.result(), step=epoch)
        tf.summary.image("val_images", test_images, step=epoch, max_outputs=1)

    ckpt_path = manager.save()
    template = "Epoch {}, Validation Loss: {:.8f}, Validation Accuracy: {:.6f}, ckpt {}\n\n"
    print(template.format(epoch+1, test_loss, test_accuracy.result(), ckpt_path))
    
    # Reset metrics for the next epoch
    test_accuracy.reset_states()
    # Early stopping
    if test_loss < Lmin:
        Lmin = test_loss
        Epoch_min = epoch
    if epoch >= Epoch_min + args.patience:
        print("Stopping after {} epochs...\n".format(epoch))
        break
    
