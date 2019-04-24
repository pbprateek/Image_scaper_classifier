'''
Just Run this file to check the performance of model:
Final test accuracy = 88.75%
Final Train Accuracy = 96.7 %
Final Validation Accuracy = 95.5%
'''



import os 
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix

base_dir=os.path.join(os.getcwd(),"ImagesProcessed")
test_dir = os.path.join(base_dir, 'test')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory( test_dir, target_size=(150, 150),shuffle=False,batch_size=20)
train_generator = datagen.flow_from_directory( train_dir, target_size=(150, 150),shuffle=False,batch_size=20)
validation_generator = datagen.flow_from_directory( validation_dir, target_size=(150, 150),shuffle=False,batch_size=20)

from keras.models import load_model
model = load_model('epoch20.h5')

model

test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)
train_steps_per_epoch = np.math.ceil(train_generator.samples / train_generator.batch_size) 
validation_steps_per_epoch = np.math.ceil(validation_generator.samples / validation_generator.batch_size)  

test_predictions = model.predict_generator(test_generator,steps=test_steps_per_epoch)
test_predicted_classes = np.argmax(test_predictions, axis=1)
train_predictions = model.predict_generator(train_generator,steps=train_steps_per_epoch)
train_predicted_classes = np.argmax(train_predictions, axis=1)
validation_predictions = model.predict_generator(validation_generator,steps=validation_steps_per_epoch)
validation_predicted_classes = np.argmax(validation_predictions, axis=1)

test_predicted_classes = np.reshape(test_predicted_classes,(80,-1))


test_true_classes = test_generator.classes.reshape(80,-1)
test_report = confusion_matrix(test_true_classes, test_predicted_classes)
print('TEST DATA CONFUSION MATRIX:')
print("")
print(test_report)
print("")
train_true_classes = train_generator.classes
train_report = confusion_matrix(train_true_classes, train_predicted_classes)
print('TRAINING DATA CONFUSION MATRIX:')
print("")
print(train_report)
print("")
validation_true_classes = validation_generator.classes
validation_report = confusion_matrix(validation_true_classes, validation_predicted_classes)
print('VALIDATION DATA CONFUSION MATRIX:')
print("")
print(validation_report)

from sklearn.metrics import accuracy_score
print('Test accuracy',accuracy_score(test_true_classes, test_predicted_classes))
print('Train accurace',accuracy_score(train_true_classes, train_predicted_classes))
print('Validation accuracy',accuracy_score(validation_true_classes, validation_predicted_classes))


from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize


base_dir=os.path.join(os.getcwd(),"ImagesProcessed")
test_dir = os.path.join(base_dir, 'test')
test_chair_dir = os.path.join(test_dir, 'chairs')
test_beds_dir = os.path.join(test_dir, 'beds')
test_lighting_dir = os.path.join(test_dir, 'lighting')
test_wardrobe_dir = os.path.join(test_dir, 'wardrobe')


imageNames = os.listdir(test_beds_dir)
filename = os.path.join(test_beds_dir , imageNames[3])

filename = os.path.join(os.getcwd() , 'wardrobe.jpg')


image = io.imread(filename)
image = resize(image, (150,150),anti_aliasing=True)
plt.imshow(image)

image_expanded = np.expand_dims(image, axis=0)

output = np.argmax(model.predict(image_expanded), axis=1).squeeze()

if output == 3:
    print('wardrobe')
elif output == 1:
    print('chair')
elif output == 0:
    print('beds')
else:
    print('lighting')




