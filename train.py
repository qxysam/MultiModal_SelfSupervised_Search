"""
@author: QQ
Note:: Loaded runtime CuDNN library: 8.0.5 but source was compiled with: 8.1.0. tf有些操作是基于应该是CuDnn_8.1.0训练的，所以整体模型 Cudnn8.1.0及以上
"""
from Data_loader_Model_bulid_Loss import create_text_encoder,create_vision_encoder,DualEncoder,get_dataset
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow as tf
# 这里必须导入tensorflow_text，bert需要这个预先注册自定义操作，卡了很久，发现这个坑。。。。
import tensorflow_text as text

# 开始训练----------------------------------------------------------------------------------------------------------------
num_epochs = 5
batch_size = 128
vision_encoder = create_vision_encoder(num_projection_layers=1,projection_dims=256,dropout_rate=0.1)
text_encoder = create_text_encoder(num_projection_layers=1,projection_dims=256,dropout_rate=0.1)
dual_encoder = DualEncoder(text_encoder,vision_encoder,temperature=0.05) # 越小越尖锐，越大越平，这里为了拉开自身向量内积和其他的距离，设置小点

# optimier:SGD, Adam, RMSProp.........AdamW(Transformer-base model)
dual_encoder.compile(optimizer=tfa.optimizers.AdamW(learning_rate=0.001,weight_decay=0.001))

# get training and validation data
train_dataset = get_dataset('tfrecords/train/*.tfrecord',batch_size=batch_size)
valid_dataset = get_dataset('tfrecords/valid/*.tfrecord',batch_size=batch_size)

# callback function: learning rate decay / early_stopping
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss',factor=0.2,patience=3 )
early_stopping = tf.keras.callbacks.EarlyStopping( monitor = 'val_loss',patience=5,restore_best_weights=True )
histroy = dual_encoder.fit(train_dataset,epochs=num_epochs,validation_data=valid_dataset,callbacks=[reduce_lr,early_stopping])


# training complete save model and draw --------------------------------------------------------------------------------
vision_encoder.save('vision_encoder')
text_encoder.save('text_encoder')

plt.plot(histroy.history['loss'])
plt.plot(histroy.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train','valid'],loc='upper right')
plt.show()