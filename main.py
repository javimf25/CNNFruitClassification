#libraries used
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
#directories that contains the images
train_directory = 'C:/Users/jmore/Desktop/fruits-360/Training/'
test_directory =  'C:/Users/jmore/Desktop/fruits-360/Test/'

fruitnames=[]
imagepath=[]
#Preprocessing and fit the images into a dataset
for i in os.listdir(train_directory):
   for image_filename in os.listdir(train_directory + i):
       fruitnames.append(i)
       imagepath.append(i + '/' + image_filename)

train_fruits = pd.DataFrame(fruitnames, columns=["Fruits"])
train_fruits["Fruits Image"] = imagepath
counter = Counter(train_fruits["Fruits"])
most_fruits = counter.most_common(10)
label=[]
sizes=[]
for f in most_fruits:
   label.append(f[0])
   sizes.append(f[1])
   print(f)


fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=label, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
imageshape= (100,100,3)

#Model declaration
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = imageshape))
model.add(MaxPooling2D())
model.add(Activation("relu"))

model.add(Conv2D(32,(3,3)))
model.add(MaxPooling2D())
model.add(Activation("relu"))

model.add(Flatten())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(131))
model.add(Activation("softmax"))
#Model compilation
model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])
#train and test image data generation
train_datagen = ImageDataGenerator(rescale= 1./255,rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale= 1./255)

train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size= (100,100),
        batch_size = 32,
        color_mode= "rgb",
        class_mode= "categorical")

test_generator = test_datagen.flow_from_directory(
        test_directory,
        target_size= (100,100),
        batch_size = 32,
        color_mode= "rgb",
        class_mode= "categorical")
#Model fitting
hist = model.fit_generator(
        generator = train_generator,
        steps_per_epoch = 1600 // 32,
        epochs=50,
        validation_data = test_generator,
        validation_steps = 800 // 32)
#Plot about loss score
plt.figure()
plt.plot(hist.history["loss"],label = "Train Loss",color="black")
plt.plot(hist.history["val_loss"],label = "Validation Loss")
plt.legend()
plt.show()
#Plot about accuracy score
plt.figure()
plt.plot(hist.history["accuracy"],label = "Train Accuracy", color = "black")
plt.plot(hist.history["val_accuracy"],label = "Validation Accuracy", color = "blue")
plt.legend()
plt.show()