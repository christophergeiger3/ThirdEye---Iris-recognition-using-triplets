"""Model generation"""
#import os
import keras.backend as K
from keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate, Conv2DTranspose, Conv2D, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.applications import densenet
from keras.layers.normalization import BatchNormalization
from keras.layers import Lambda
from keras.layers.merge import concatenate
from keras.models import Model
#from keras.callbacks import ModelCheckpoint,Callback

from main import display
 

def main(IMG_DIMENSION=None, FEATURES=None, **kwargs):
    inp = Input(shape = (3,IMG_DIMENSION,IMG_DIMENSION))
    base_model = densenet.DenseNet169(weights='imagenet',
            include_top=False,
            input_shape=(IMG_DIMENSION,IMG_DIMENSION,3))

    display("Loaded initial model")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    fc2 = Dense(FEATURES,activation = 'relu',name = 'embedding_weights',use_bias = False)(x)
    fc2 = BatchNormalization()(fc2)
    out = Lambda(lambda x: K.l2_normalize(x, axis=1))(fc2)
    label = Dense(1,activation='linear',name = 'label')(out)
    conc = concatenate([fc2,label], name='xF')
    triplet_model = Model(inputs=base_model.input, outputs=conc)

    display("Loaded additional layers")

    ## TODO: Change to os.path.join()
    ## TODO: Move callbacks_list to train.py?
    #cwd = os.getcwd()
    #filepath=cwd + "/batchHard_ResNet50-{epoch:02d}-{loss:.2f}.hdf5"

    #checkpoint = ModelCheckpoint(filepath, monitor='val_predictions_acc',
            #verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    #callbacks_list = [checkpoint]
    #print("Configured checkpointing")

    # print (triplet_model.summary())
    return triplet_model
