# things to remember:
import glob
import os
import tensorflow as tf

train_path = 'dataset/train/'
train_list = glob.glob(train_path+"*")
train_images_list = glob.glob(train_path+"images/*")
train_ds = tf.data.Dataset.list_files(train_images_list)
a = train_ds.take(0)

# #data augmentation function
# IMG_WIDTH=224
# IMG_HEIGHT=224
# def decode_img(img):
#   img = tf.image.decode_jpeg(img, channels=3)
#   img = tf.image.convert_image_dtype(img, tf.float32) 
#   img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) 
#   img = tf.image.random_flip_left_right(img)
#   img = tf.image.random_flip_up_down(img)
#   img = tf.image.random_brightness(img, 0.3)
#   return img

# def get_label(path):
#   part_list = tf.strings.split(path, "/")
#   # in the case where each class of images is in one folder
#   return part_list[-2] == class_names

# def process_path(file_path):
#   label = get_label(file_path)
#   img = tf.io.read_file(file_path)
#   img = decode_img(img)
#   return img, label

print(a)