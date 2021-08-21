"""
@author: QQ
Note:: Loaded runtime CuDNN library: 8.0.5 but source was compiled with: 8.1.0. tf有些操作是基于应该是CuDnn_8.1.0训练的，所以整体模型 Cudnn8.1.0及以上
"""

import os
import json
import collections
import numpy as np
import tensorflow as tf

# File Location,Here we make allTrain datasets split into train and valid-----------------------------------------------
annotation = r'G:\Multimodal Fusion Seach\captions_train-val\annotations'
images_dir = r'G:\Multimodal Fusion Seach\train'
tfrecords_dir = r'G:\Multimodal Fusion Seach\tfrecords'
annotation_file = r'G:\Multimodal Fusion Seach\captions_train-val\annotations\captions_train.json'

# retrive all image_paths-----------------------------------------------------------------------------------------------
with open(annotation_file,'r') as f:
    annotations = json.load(f)['annotations']
image_path_to_caption = collections.defaultdict(list)
for element in annotations:
    caption = element['caption'].lower().rstrip('.')
    image_path = images_dir + r'\COCO_train_' + "%012d.jpg"%(element['image_id'])
    # print(image_path)                  ??????
    image_path_to_caption[image_path].append(caption)   # type:{'image_path_1':['caption_1','caption2'],'image_path_2':['caption_1','caption2']}
    # print(image_path_to_caption)
image_paths = list(image_path_to_caption.keys())

if __name__ == '__main__':
    print('number of images:',len(image_paths))


# blocks ready for tfrecord---------------------------------------------------------------------------------------------
train_size = 300
valid_size = 50
captions_per_image = 2
images_per_tffile = 20

train_image_paths = image_paths[:train_size]
num_train_files = int(np.ceil(train_size/images_per_tffile))
train_files_prefix = r'G:\CV\Multimodal Fusion Seach\tfrecords\train'

valid_image_paths =image_paths[-valid_size:]
num_valid_files = int(np.ceil(valid_size/images_per_tffile))
valid_files_prefix = r'G:\CV\Multimodal Fusion Seach\tfrecords\valid'

# tf.io.gfile.makedirs(r'G:\CV_JULY\多模态图文搜索\tfrecords')

# Save as tfrecords-----------------------------------------------------------------------------------------------------
def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList( value = [value]) )
def create_example(image_path, caption):
    feature = {'caption': bytes_feature(caption.encode()),
               'raw_image': bytes_feature(tf.io.read_file(image_path).numpy()) }
    return tf.train.Example(features = tf.train.Features(feature = feature))

def write_tfrecords(file_name, image_paths):
    caption_list = []
    image_path_list = []
    for image_path in image_paths:
        captions = image_path_to_caption[image_path][:captions_per_image]
        caption_list.extend(captions)

        # this will make one image doubles for equaling to length of captions list--------------------------------------
        image_path_list.extend([image_path]*len(captions))

    # write into tf_records a line as the formula <image_path,caption>--------------------------------------------------------
    with tf.io.TFRecordWriter(file_name) as writer:
        for example_idx in range(len(image_path_list)):
            example = create_example(image_path_list[example_idx],caption_list[example_idx])
            writer.write(example.SerializeToString())
        return example_idx+1

# Write data (split into a few files)-----------------------------------------------------------------------------------
def write_data(image_paths,num_files,files_prefix):
    example_counter = 0
    for file_idx in range(num_files):
        file_name = files_prefix + r'\%2d--.tfrecord' % (file_idx+1)
        start_idx = images_per_tffile * file_idx
        end_idx = start_idx + images_per_tffile
        example_counter += write_tfrecords(file_name,image_paths[start_idx : end_idx])
    return example_counter

if __name__ == '__main__':
    print(write_data(train_image_paths, num_train_files, r'G:\CV\Multimodal Fusion Seach\tfrecords\train'))
    print(write_data(valid_image_paths, num_valid_files, r'G:\CV\Multimodal Fusion Seach\tfrecords\valid'))




