from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from tensorflow.keras.utils import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from PIL import Image
import numpy as np

# obtinere imagine
image = img_to_array(load_img('image5000.jpg'))
image = np.array(image, dtype=float)
X = rgb2lab(1.0/255*image)[:,:,0]
Y = rgb2lab(1.0/255*image)[:,:,1:]
Y /= 128
X = X.reshape(1, 400, 400, 1)
Y = Y.reshape(1, 400, 400, 2)

# mod straturi
model = Sequential()
# (canale de iesire, dim w convolutie, f.activ., pad, size)
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(400, 400, 1)))
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
model.compile(optimizer='rmsprop',loss='mse')

history = model.fit(x=X, 
	                  y=Y,
	                  batch_size=1,
	                  epochs=1000)
print(model.evaluate(X, Y, batch_size=1))
output = model.predict(X)
# a si b [-128,127]
output *= 128 

# concatenare
cur = np.zeros((400, 400, 3))
cur[:,:,0] = X[0][:,:,0]
cur[:,:,1:] = output[0]
cur_rgb = lab2rgb(cur)

# scalare
cur_rgb_scaled = (cur_rgb * 255).astype('uint8')

# ca sa avem varianta alb-negru salvata
cur_gray = rgb2gray(cur_rgb)
cur_gray_scaled = (cur_gray * 255).astype('uint8')

# Save the images
Image.fromarray(cur_rgb_scaled).save("img_result3.png")
Image.fromarray(cur_gray_scaled, mode='L').save("img_gray3.png")
