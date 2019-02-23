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
model = load_model('vgg16_final.h5')

test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)
train_steps_per_epoch = np.math.ceil(train_generator.samples / train_generator.batch_size) 
validation_steps_per_epoch = np.math.ceil(validation_generator.samples / validation_generator.batch_size)  

test_predictions = model.predict_generator(test_generator,steps=test_steps_per_epoch)
test_predicted_classes = np.argmax(test_predictions, axis=1)
train_predictions = model.predict_generator(train_generator,steps=train_steps_per_epoch)
train_predicted_classes = np.argmax(train_predictions, axis=1)
validation_predictions = model.predict_generator(validation_generator,steps=validation_steps_per_epoch)
validation_predicted_classes = np.argmax(validation_predictions, axis=1)

test_true_classes = test_generator.classes
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
