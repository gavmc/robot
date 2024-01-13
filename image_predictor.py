import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


input_images = []
output_images = []

input_path = 'D:/robot/screenshots/32RGB/raw/'
output_path = 'D:/robot/screenshots/32RGB/checkpoint/'

filenames = next(os.walk(input_path), (None, None, []))[2]
for f in filenames:
    input_images.append(cv2.imread(input_path+f)/255)

filenames = next(os.walk(output_path), (None, None, []))[2]
for f in filenames:
    output_images.append(cv2.imread(output_path+f, cv2.IMREAD_GRAYSCALE)/255)

print(np.array(input_images).shape)
print(np.array(output_images).shape)

input_val = np.array(input_images[-100:]).reshape((len(input_images[-100:]), 32, 58, 3))
output_val = np.array(output_images[-100:]).reshape((len(output_images[-100:]), 32, 58, 1))

input_images = np.array(input_images[:-100]).reshape((len(input_images[:-100]), 32, 58, 3))
output_images = np.array(output_images[:-100]).reshape((len(output_images[:-100]), 32, 58, 1))

print(input_images.shape)
print(output_images.shape)

class model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, (3, 3), padding = 'same', activation = 'relu'),
            tf.keras.layers.MaxPooling2D(pool_size = (2, 2), padding ='same'),
            tf.keras.layers.Conv2D(16, (1, 1), padding = 'same', activation = 'relu'),
            tf.keras.layers.Conv2D(16, (3, 3), padding = 'same', activation = 'relu'),
            tf.keras.layers.Conv2D(32, (1, 1), padding = 'same', activation = 'relu'),
            ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(32, (3, 3), padding = 'same', activation = 'relu'),
            tf.keras.layers.Conv2DTranspose(16, (3, 3), padding = 'same', activation = 'relu'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2DTranspose(32, (3, 3), padding = 'same', activation = 'relu'),
            tf.keras.layers.Conv2DTranspose(1, (3, 3), padding = 'same', activation = 'relu'),
            ])

    def call(self, inputs, training=False):
        x=inputs
        x=self.encoder(inputs)
        x=self.decoder(x)
        return x



mask_model = model()
loss = tf.keras.losses.BinaryCrossentropy()
mask_model.compile(loss=loss, optimizer='adam', metrics = ['Accuracy'])

mask_model(np.expand_dims(input_images[0], axis=0))
mask_model.summary()


mask_model.fit(input_images,
               output_images,
               epochs=35,
               validation_data=(input_val, output_val),
               verbose=2)



for i in range(5):
    img = mask_model(np.expand_dims(input_val[i], axis=0))
    plt.imshow(output_val[i])
    plt.show()
    plt.imshow(img[0])
    plt.show()

mask_model.save('D:/robot/checkpoint_ml', save_format='tf')
    






        
    
