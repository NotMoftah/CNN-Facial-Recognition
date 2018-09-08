import tensorflow as tf
import numpy as np


'''---------------------------------------- Load Data ----------------------------------------'''
faces_x = np.load('data/mit_faces/x_train.npy')
faces_y = np.load('data/mit_faces/y_train.npy')

print('Faces were loaded successfully.')

'''----------------------------------- Features Extraction -----------------------------------'''

# Load the keras pre-trained MobileNet model.
features_extractor = tf.keras.Sequential()
pure_mobile_net = tf.keras.applications.MobileNet(input_shape=(128, 128, 3), weights='imagenet', include_top= False)
features_extractor.add(pure_mobile_net)
features_extractor.add(tf.keras.layers.GlobalAveragePooling2D())

# extract the features into separate array to save performance while training

train_features_tensor = features_extractor.predict(faces_x)
np.save('data/mit_faces/features_tensor', train_features_tensor)


def get_train_patch(patch_size=100):
    idx = np.random.randint(0, len(faces_x), size=(patch_size,))

    x_train_patch = train_features_tensor[idx]
    y_train_patch = faces_y[idx]

    return x_train_patch, y_train_patch


print('Features were extracted successfully.')

'''----------------------------------- Hashing Network Setup -----------------------------------'''

# Setup placeholder for the in/out data
x_holder = tf.placeholder(shape=[None, 1024], dtype=tf.float32)
y_holder = tf.placeholder(shape=[None], dtype=tf.int8)

# Construct the fully connected hashing layers
fc_1 = tf.layers.Dense(units=512, activation=tf.nn.relu)(x_holder)
drop_layer = tf.layers.Dropout(0.25)(fc_1)
fc_2 = tf.layers.Dense(units=128, activation=tf.nn.sigmoid)(drop_layer)

# Semi-Hard triplet loss with Adam Optimizer
loss_function = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=y_holder, embeddings=fc_2, margin=3.0)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_function)

print('Hashing neural network were deployed successfully.')

'''----------------------------------- Training The Network -----------------------------------'''

saver = tf.train.Saver()

print('Training has started.')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    STEPS, PATCH_SIZE = 1500, 100

    for i in range(STEPS):
        x_train_patch, y_train_patch = get_train_patch(PATCH_SIZE)
        _, loss = sess.run([optimizer, loss_function], feed_dict={x_holder: x_train_patch, y_holder: y_train_patch})

        if i % 10 == 0:
            print(i, '\t\t', loss)

    saver.save(sess, 'data\\trained_model\\')

print('Training is finished.')
