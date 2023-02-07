import os
import tensorflow as tf
num_threads = 4
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)

from tqdm import tqdm
import numpy as np
import argparse
import time
import pandas as pd
from utils_dense_real_syn import quaternion_cross_entropy


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


def load_and_resize_img(imgpath):
    raw_img = tf.io.read_file(imgpath)
    img = tf.io.decode_jpeg(raw_img, channels=3)
    img = tf.image.resize_with_pad(img, target_height=INPUT_SIZE[0], target_width=INPUT_SIZE[1], method="nearest")
    #img = tf.image.resize(img, size=INPUT_SIZE, method="nearest")
    return img

def preprocess(imgpath, label):
    img = tf.map_fn(load_and_resize_img, imgpath, fn_output_signature=tf.uint8)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.divide(img, tf.constant(255.0, dtype=tf.float32))
    return img, tf.one_hot(label, depth=nclasses)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = quaternion_cross_entropy(predictions, labels)
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=images.shape[0]) # strategy.num_replicas_in_sync*images.shape[0]
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(labels, predictions)
    return loss #, predictions

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    test_loss = quaternion_cross_entropy(predictions, labels)
    test_loss = tf.reduce_sum(test_loss, axis=-1)  
    test_loss = tf.nn.compute_average_loss(test_loss, global_batch_size=images.shape[0]) #strategy.num_replicas_in_sync*images.shape[0]
    test_accuracy.update_state(labels, predictions)
    return test_loss

# Load dataset
data_dir = "/scratch/hnkmah001/Datasets/PASCAL3D+_release1.1/"
train_val_df = pd.read_csv(data_dir + "Image_sets/{}/{}_train_realPlusSyn.csv".format(args.category, args.category), sep=",")
img_path = train_val_df["image_path"].apply(lambda imgID: data_dir + imgID)
q_class = train_val_df["sampled_class_label_data"].astype(int)

nclasses = len(set(q_class))

img_path_list = tf.constant(np.array(img_path))
labels_list = tf.constant(np.array(q_class))
dataset = tf.data.Dataset.from_tensor_slices((img_path_list, labels_list)).shuffle(len(img_path_list))
val_data = dataset.take(int(0.2 * len(img_path_list))).batch(args.batch_size)
train_data = dataset.skip(int(0.2 * len(img_path_list))).batch(args.batch_size)


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
outputs = tf.keras.layers.Dense(nclasses, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs) 
model.summary()

# Define cost function, optimizer and metrics
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/classification_dense_real_syn/{}/{}_samples/checkpoints/".format(args.category, args.samples)
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=args.patience+1)

# Save logs with TensorBoard Summary
train_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/classification_dense_real_syn/{}/{}_samples/logs/train".format(args.category, args.samples)
val_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-pascal/classification_dense_real_syn/{}/{}_samples/logs/val".format(args.category, args.samples)
train_summary_writer = tf.summary.create_file_writer(train_logdir)
val_summary_writer = tf.summary.create_file_writer(val_logdir)


# Training loop

step = 0
Lmin = tf.constant(np.inf)
for epoch in range(args.epochs):
    for images, labels in train_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        tic = time.time()
        train_loss = train_step(images, labels)
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
    for test_images, test_labels in tqdm(val_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE), 
                                                        desc="Validation"):
        test_loss += test_step(test_images, test_labels)
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
    
