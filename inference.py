"""
@author: QQ
Note:: Loaded runtime CuDNN library: 8.0.5 but source was compiled with: 8.1.0. tf有些操作是基于应该是CuDnn_8.1.0训练的，所以整体模型 Cudnn8.1.0及以上
"""

from Pre_Process_to_tfrecord import image_paths
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# 这里必须导入tensorflow_text，bert需要这个预先注册自定义操作，卡了很久，发现这个坑。。。。
import tensorflow_text as text

# 1.generate visual embeddings for all images>>>vision encoder
# 2.generate text embeddings for a query
# 3.compute aimilarities between query and images
# 4.display top ranking images or evaluation.

vision_encoder = tf.keras.models.load_model('vision_encoder')
text_encoder = tf.keras.models.load_model('text_encoder')

def read_image(image_path):
    image_array = tf.image.decode_jpeg(tf.io.read_file(image_path),channels=3)
    return tf.image.resize(image_array,(299,299))

print('Generating visual embeddings......')
def find_matches(image_embeddings,queries,k=9,normalize=True):
    query_embedding = text_encoder.predict(tf.convert_to_tensor(queries))
    if normalize:
        image_embeddings = tf.math.l2_normalize(image_embeddings,axis=1)
        query_embedding = tf.math.l2_normalize(query_embedding, axis=1)

    # compute simi between the query and image embeddings.
    dot_similarity = tf.matmul(query_embedding,image_embeddings,transpose_b=True)
    # Retrive top k indices.
    results = tf.math.top_k(dot_similarity,k).indices.numpy()
    # Return matching image_paths
    return [ [ image_paths[idx] for idx in indices] for indices in results ]

batch_size=128
image_embeddings = vision_encoder.predict( tf.data.Dataset.from_tensor_slices(image_paths).map(read_image).batch(batch_size), verbose=1 )

query = 'a family standing next to the ocean on a sandy beach with a surf board'
matches = find_matches(image_embeddings,[query],normalize=True)[0]
# Could sava matches into a file----------------------------------------------------------------------------------------

# Draw Optional---------------------------------------------------------------------------------------------------------
plt.figure(figsize=(20,20))
for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(mpimg.imread(matches[i]))
    plt.axis('off')

# define a evaluation metric, Top-K Accuracy----------------------------------------------------------------------------