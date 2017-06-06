import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense
from SpatialPyramidPooling import SpatialPyramidPooling
from data_loader import BirdClassificationGenerator
from img_loader import ImageLoader

data_path = '/datasets/CUB_200_2011'
val_split = 0.3
batch_size = 64
num_channels = 3
num_classes = 10

loader_obj = ImageLoader()

print "[INFO] Loading Dataset"
data_obj = BirdClassificationGenerator(data_path, val_split, batch_size)
print "     > Complete"

print "[INFO] Initializing Model"
model = Sequential()
# uses theano ordering. Note that we leave the image size as None to allow multiple image sizes
model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(None, None, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(SpatialPyramidPooling([1, 2, 4]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd')
print "     > Complete"

print "[INFO] Training..."
for train_impaths, train_bbs, train_labs in data_obj.train_generator():
    train_imgs = loader_obj.get_img_numpy(train_impaths)

    model.fit(train_imgs, train_labs)

# train on 64x64x3 images
#model.fit(np.random.rand(batch_size, 64, 64, num_channels), np.zeros((batch_size, num_classes)))
# train on 32x32x3 images
#model.fit(np.random.rand(batch_size, 32, 32, num_channels), np.zeros((batch_size, num_classes)))
