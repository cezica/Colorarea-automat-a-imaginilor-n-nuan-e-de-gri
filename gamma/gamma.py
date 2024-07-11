import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
import keras.backend as K
from keras.layers import Layer, Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, concatenate, concatenate, Activation, Dense, Dropout, Flatten, BatchNormalization
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
from PIL import Image, ImageFile

# Functia de redimensionare a imaginii
def resize_image(image, target_size=(256, 256)):
    image = image.resize(target_size, Image.LANCZOS)
    return img_to_array(image)

# # obtinere imagini din fisier
X = []
for filename in os.listdir('../archive/train_f/'):
     input_path = os.path.join('../archive/train_f/', filename)
     if os.path.getsize(input_path) > 0:  # verifica daca fisierul e gol
        try:
            img = load_img(input_path)
            img_array = resize_image(img)
            X.append(img_array)
        except (UnidentifiedImageError, OSError) as e:
            print(f"Nu s-a putut procesa imaginea {input_path}: {e}")

X = np.array(X, dtype=float)
Xtrain = 1.0/255*X

# ponderile modelului preantrenat
# (setul de date, stratul conectat complet)
inception = InceptionResNetV2(weights='imagenet', include_top=True)

# vector caracteristici
embed_input = Input(shape=(1000,))

#Encoder
encoder_input = Input(shape=(256, 256, 1,))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

#Fusion
fusion_output = RepeatVector(32 * 32)(embed_input) 
fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3) # concatenare dim 4
fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

#Decoder
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

# 2 straturi de intrare, 1 de iesire
model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    embed = inception.predict(grayscaled_rgb_resized)
    return embed

# augmentarea datelor
# deformari, scalari,
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

#Generate training data
batch_size = 30

def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size): # ImageDataGenerator.flow - augmentare pe loturi
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)

#Train model      
model.compile(optimizer='rmsprop', loss='mse')
model.fit(image_a_b_gen(batch_size), epochs=150, steps_per_epoch=266) 

# incarcare test
color_me = []
for filename in os.listdir('../archive/test_f/'):
    color_me.append(img_to_array(load_img('../archive/test_f/'+filename)))

# preprocesare
color_me = np.array(color_me, dtype=float) # pt operatii matematice
gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
color_me_embed = create_inception_embedding(gray_me)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))


# Test model
output = model.predict([color_me, color_me_embed])
output = output * 128

# Output colorizations
os.makedirs("result", exist_ok=True)
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    cur_rgb = lab2rgb(cur)
    cur_rgb = np.clip(cur_rgb, 0, 1)  # Clip negative values to zero
    cur_rgb_uint8 = (cur_rgb * 255).astype(np.uint8)
    Image.fromarray(cur_rgb_uint8).save("result/img_"+str(i)+".png")