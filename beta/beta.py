import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.callbacks import TensorBoard
from skimage.color import rgb2lab, lab2rgb
from PIL import Image, UnidentifiedImageError

# Functia de redimensionare a imaginii
def resize_image(image, target_size=(256, 256)):
    image = image.resize(target_size, Image.LANCZOS)
    return img_to_array(image)


# obtinere imagini din fisier
X = []
for filename in os.listdir('../archive/train_more/'):
    input_path = os.path.join('../archive/train_more/', filename)
    if os.path.getsize(input_path) > 0:  # verifica daca fisierul e gol
        try:
            img = load_img(input_path)
            img_array = resize_image(img)
            X.append(img_array)
        except (UnidentifiedImageError, OSError) as e:
            print(f"Nu s-a putut procesa imaginea {input_path}: {e}")

X = np.array(X, dtype=float)

# Set up train and test data
split = int(0.95 * len(X))
Xtrain = X[:split]
Xtrain = 1.0 / 255 * Xtrain

# mod straturi
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
# (canale de iesire, dim w convolutie, f.activ., pad, size)
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# w convoluție miscare orizontala si verticala
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# marire cu factor pe axe (orizontal,vertical)
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# tanh pt a avea iesiri [-1,1] - harti de culoare Lab
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse')

# augmentarea
# deformari aleatoare, scalari aleatoare, rotatii aleatoare, intoarceri orizontala
datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2, # [0.8, 1.2]
    rotation_range=20,
    horizontal_flip=True,
)

# generare date pentru antrenare
batch_size = 50
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:, :, :, 0]
        Y_batch = lab_batch[:, :, :, 1:] / 128
        yield (X_batch.reshape(X_batch.shape + (1,)), Y_batch)

# Train model
tensorboard = TensorBoard(log_dir="output/first_run")
model.fit(image_a_b_gen(batch_size), callbacks=[tensorboard], epochs=100, steps_per_epoch=200)

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

# Test images
Xtest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 0]
Xtest = Xtest.reshape(Xtest.shape + (1,))
Ytest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 1:]
Ytest = Ytest / 128
print(model.evaluate(Xtest, Ytest, batch_size=batch_size))

color_me = []
for filename in os.listdir('../archive/test_more/'):
    input_path = os.path.join('../archive/test_more/', filename)
    if os.path.getsize(input_path) > 0:  # verifica daca fisierul e gol
        try:
            img = load_img(input_path)
            img_array = resize_image(img)
            color_me.append(img_array)
        except (UnidentifiedImageError, OSError) as e:
            print(f"Nu s-a putut procesa imaginea {input_path}: {e}")

# preprocesare
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
color_me = color_me.reshape(color_me.shape + (1,))

# Test model
output = model.predict(color_me)
output = output * 128

# restrangerea valorilor
output = np.clip(output, 0, 255)

# salvare rezultate
os.makedirs("result", exist_ok=True)
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = color_me[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    cur_rgb = lab2rgb(cur)
    cur_rgb_uint8 = (cur_rgb * 255).astype(np.uint8)
    Image.fromarray(cur_rgb_uint8).save("result/img_" + str(i) + ".png")
