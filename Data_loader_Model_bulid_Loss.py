"""
@author: QQ
Note:: Loaded runtime CuDNN library: 8.0.5 but source was compiled with: 8.1.0. tf有些操作是基于应该是CuDnn_8.1.0训练的，所以整体模型 Cudnn8.1.0及以上
"""
import os
import json
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow_hub as hub
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

# Create tf.data.Dataset for model training and evaluation
feature_description = {
    'caption': tf.io.FixedLenFeature([], tf.string),
    'raw_image': tf.io.FixedLenFeature([], tf.string) }

def read_example(example):
    features = tf.io.parse_single_example(example, feature_description)
    raw_image = features.pop('raw_image')
    features['image'] = tf.image.resize(tf.image.decode_jpeg(raw_image,channels=3),size=(299,299))
    return features

def get_dataset(file_pattern, batch_size):
    return ( tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
             .map(read_example,num_parallel_calls= tf.data.experimental.AUTOTUNE, deterministic= False)
             .shuffle(batch_size*10)
             .prefetch(buffer_size= tf.data.experimental.AUTOTUNE).batch(batch_size) )

# Build Model-----------------------------------------------------------------------------------------------------------
# Both two---> Projection head(transform image and text to same embedding space
# For example num_image_embedding =1024, num_text_embedding = 512
# Project image embedding: 1024->256->256; Project text embedding: 512->256->256

def project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate):
    projected_embeddings = layers.Dense(units= projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = layers.Dense(projection_dims)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([projected_embeddings,x])
        projected_embeddings = layers.LayerNormalization()(x)
    return projected_embeddings

# 1、Implement Vision Encoder
def create_vision_encoder( num_projection_layers, projection_dims, dropout_rate, trainable=False ):
    # Load any CNN /VIT model( pretrained on ImageNet ) as vision encoder.
    # Resnet-50 / VGG / Inception-X / Xception Model.
    cnn = tf.keras.applications.ResNet50(include_top= False,weights='imagenet',pooling='avg')
    for layer in cnn.layers:
        layer.trainable = trainable
    inputs = layers.Input(shape = (299,299,3),name='image_input')
    cnn_input = tf.keras.applications.resnet50.preprocess_input(inputs)
    embeddings = cnn(cnn_input)
    outputs = project_embeddings( embeddings, num_projection_layers, projection_dims,dropout_rate )
    vision_encoder_model = Model(inputs,outputs)
    return vision_encoder_model

# 2、Implement Text Encoder ( word2vec / RNN / LSTM / BERT )
def create_text_encoder( num_projection_layers, projection_dims, dropout_rate, trainable=False ):
    # Load pretrained Model( SMALL BERT )
    bert = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1','bert')
    preprocess = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',name='text_preprocessing')
    bert.trainable = trainable
    inputs = layers.Input(shape=(),dtype=tf.string,name='text_iuput')
    bert_inputs = preprocess(inputs)
    embeddings = bert(bert_inputs)['pooled_output']
    outputs = project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate)
    text_encoder_model = Model(inputs, outputs)
    return text_encoder_model


class DualEncoder(tf.keras.Model):
    def __init__(self,text_encoder,image_encoder,temperature=1.,**kwargs):
        super(DualEncoder, self).__init__(**kwargs)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self.loss_tracker]

    def __call__(self,features,training=False):
        with tf.device('/gpu:0'):
            caption_embeddings = self.text_encoder(features['caption'],training=training)
        with tf.device('/gpu:1'):
            image_embeddings = self.image_encoder(features['image'], training=training)
        return caption_embeddings,image_embeddings


    # Implement the LOSS Function-------------------------------------------------------------------------------------------
    # In order to compute the loss, compute the pair-wise inner product similarity between each text_vect_i and vision_vect_j in one batch_size
    # Then use Cross-Entropy LOSS to compute the final loss in view of both rows and columns (average)
    def compute_loss(self,caption_embeddings, image_embeddings):
        # generate Multimodal similarity Matrix
        logits = tf.keras.activations.softmax( tf.matmul( caption_embeddings, image_embeddings,transpose_b= True ) / self.temperature )

        # generate Singlemodal similarity Matrix
        image_similaritry = tf.matmul( image_embeddings, image_embeddings, transpose_b= True )
        captions_similaritry = tf.matmul( caption_embeddings, caption_embeddings, transpose_b= True )
        soft_targets = tf.keras.activations.softmax((image_similaritry + captions_similaritry) / (2*self.temperature))

        captions_loss = tf.keras.losses.categorical_crossentropy( y_true=soft_targets, y_pred=logits )
        image_loss = tf.keras.losses.categorical_crossentropy( y_true=tf.transpose(soft_targets), y_pred=tf.transpose(logits) )

        average_loss = (captions_loss + image_loss) /2.
        return average_loss

    def train_step(self,features):
        with tf.GradientTape() as tape:
            caption_embeddings, image_embeddings = self(features,training=True)
            loss = self.compute_loss(caption_embeddings,image_embeddings)
        gradients = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return { 'loss':self.loss_tracker.result() }

    def test_step(self,features):
        caption_embeddings, image_embeddings = self(features, training=False)
        loss = self.compute_loss(caption_embeddings,image_embeddings)
        self.loss_tracker.update_state(loss)
        return { 'loss': self.loss_tracker.result() }

