"""
    Training happens here, same training and validation generators due to laziness
"""
import os
import utils
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint,Callback
from main import display

def main(triplet_model):
    ## TODO: Change to os.path.join()
    ## TODO: Add checkpoint directory
    cwd = os.getcwd()
    filepath=cwd + "/batchHard_ResNet50-{epoch:02d}-{loss:.2f}.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_predictions_acc',
            verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callbacks_list = [checkpoint]
    display("Configured checkpointing")

    ## TODO: Add epochs, workers, steps_per_epoch.... to CLI
    gen_tr = utils.customGenerator()
    gen_te = utils.customGenerator()
    triplet_model.compile(loss=utils.batch_hard_triplet_loss, optimizer=SGD(0.0009))
    triplet_model.fit_generator(gen_tr,validation_data=gen_te,  
                              epochs=7, 
                              verbose=1,
                              workers=4,
                              steps_per_epoch=300, 
                              validation_steps=50,
                              use_multiprocessing = True,
                              callbacks = callbacks_list)

    return triplet_model
