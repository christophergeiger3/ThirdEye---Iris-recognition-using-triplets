import argparse
import os
import json
from sys import exit

def display(text):
    print('='*50)
    print(text)
    print('='*50)

def main():
    # Create the model
    display("Creating the model...")
    import model
    triplet_model = model.main(**params)

    display("Training the model...")
    import train
    triplet_model = train.main(triplet_model)

    display("Inferring...")
    import infer
    infer.main(triplet_model, OUTPUT=args.output, **params)

    display("Done!")

# Putting this in for some errors that occur on my setup
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser()
#parser.add_argument('--train_dir', default='/data/Training data', help="Directory containing the training data")
#parser.add_argument('--infer_dir', default='/data/infer', help="Directory containing the infer data")
parser.add_argument('--config', default=os.path.join(os.getcwd(), 'params.json'),
        help="Configuration file (e.g. params.json)")
parser.add_argument('--output', default='out.png', help="The output png file name")

# Get CLI args
args = parser.parse_args()
print(args)

# Get config params
assert os.path.isfile(args.config), "No json configuration file found at {}".format(args.config)
with open(args.config) as json_file:
    params = json.load(json_file)

if __name__ == "__main__":
    main()

