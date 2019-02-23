import os, shutil
original_dataset_dir = os.path.join(os.getcwd(),"Images")
print(original_dataset_dir) 

base_dir=os.path.join(os.getcwd(),"ImagesProcessed")

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'validation')

test_dir = os.path.join(base_dir, 'test')

try:
    os.mkdir(base_dir)
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)
except FileExistsError:
    print("Folder Already Exists")

train_chair_dir = os.path.join(train_dir, 'chairs')
train_beds_dir = os.path.join(train_dir, 'beds')
train_lighting_dir = os.path.join(train_dir, 'lighting')
train_wardobe_dir = os.path.join(train_dir, 'wardrobe')

validation_chair_dir = os.path.join(validation_dir, 'chairs')
validation_beds_dir = os.path.join(validation_dir, 'beds')
validation_lighting_dir = os.path.join(validation_dir, 'lighting')
validation_wardrobe_dir = os.path.join(validation_dir, 'wardrobe')

test_chair_dir = os.path.join(test_dir, 'chairs')
test_beds_dir = os.path.join(test_dir, 'beds')
test_lighting_dir = os.path.join(test_dir, 'lighting')
test_wardrobe_dir = os.path.join(test_dir, 'wardrobe')

try:
    os.mkdir(train_chair_dir)
    os.mkdir(train_beds_dir)
    os.mkdir(train_lighting_dir)
    os.mkdir(train_wardobe_dir)
    os.mkdir(validation_chair_dir)
    os.mkdir(validation_beds_dir)
    os.mkdir(validation_lighting_dir)
    os.mkdir(validation_wardrobe_dir)
    os.mkdir(test_chair_dir)
    os.mkdir(test_beds_dir)
    os.mkdir(test_lighting_dir)
    os.mkdir(test_wardrobe_dir)
except FileExistsError:
    print("Folder Already Exists")

    

def copyImages(cat):
    imageNames = os.listdir(os.path.join(original_dataset_dir,cat))
    size = len(imageNames)
    train_size =size - 40
    for fname in imageNames[:train_size]:
        src = os.path.join(original_dataset_dir,cat,fname)
        dst = os.path.join(train_dir,cat)
        shutil.copy(src,dst)
    
    for fname in imageNames[train_size:size-20]:
        src = os.path.join(original_dataset_dir,cat,fname)
        dst = os.path.join(validation_dir,cat)
        shutil.copy(src,dst)

    for fname in imageNames[train_size+20:]:
        src = os.path.join(original_dataset_dir,cat,fname)
        dst = os.path.join(test_dir,cat)
        shutil.copy(src,dst)


copyImages("beds")
copyImages("chairs")
copyImages("lighting")
copyImages("wardrobe")

print('total training beds images:', len(os.listdir(train_beds_dir)))
print('total training chairs images:', len(os.listdir(train_chair_dir)))
print('total training lighting images:', len(os.listdir(train_lighting_dir)))
print('total training wardrobe images:', len(os.listdir(train_wardobe_dir)))
print('total validation beds images:', len(os.listdir(validation_beds_dir)))
print('total validation chairs images:', len(os.listdir(validation_chair_dir)))
print('total validation lighting images:', len(os.listdir(validation_lighting_dir)))
print('total validation wardrobe images:', len(os.listdir(validation_wardrobe_dir)))
print('total test beds images:', len(os.listdir(test_beds_dir)))
print('total test chairs images:', len(os.listdir(test_chair_dir)))
print('total test lighting images:', len(os.listdir(test_lighting_dir)))
print('total test wardrobe images:', len(os.listdir(test_wardrobe_dir)))

from keras.applications import VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(4, activation='softmax')) 
model.summary()

conv_base.trainable = False

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory( train_dir, target_size=(150, 150), batch_size=10)
validation_generator = test_datagen.flow_from_directory( validation_dir, target_size=(150, 150),
                                                        batch_size=10)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
history = model.fit_generator( train_generator, steps_per_epoch=10, epochs=20,
                              validation_data=validation_generator, validation_steps=50)


conv_base.trainable = True 
set_trainable = False 
for layer in conv_base.layers:
    if layer.name == 'block5_conv1': 
        set_trainable = True 
        if set_trainable:
            layer.trainable = True 
        else: 
            layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc']) 
history = model.fit_generator( train_generator, steps_per_epoch=50, epochs=10, validation_data=validation_generator,validation_steps=6)

model.save('vgg16_final.h5')




