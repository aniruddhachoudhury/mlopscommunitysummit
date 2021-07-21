from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import logging
import pandas as pd
from datetime import datetime
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

logger = tf.get_logger()
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO)
print('Tensorflow-version: {0}'.format(tf.__version__))

import os
import argparse
import json



    

# build model
def build_model(learning_rate,drop_out,opt,denselayer,dense_neurons,intializer):
   

    if opt == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif opt =="ADAM":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
    else:
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(9,)))
    # Add fully connected layers.
    
    for _ in range(denselayer):
        model.add(tf.keras.layers.Dense(dense_neurons, kernel_initializer=intializer, activation="relu"))
        dense_neurons *= 2
    model.add(tf.keras.layers.Dropout(drop_out, )) #seed=rng.random()
    
    for _ in range(denselayer):
        model.add(tf.keras.layers.Dense(dense_neurons, kernel_initializer=intializer, activation="relu"))
        dense_neurons /= 2
    # Add the final output layer.
    model.add(tf.keras.layers.Dropout(drop_out))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

  
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    

    return model

    
# callbacks
def get_callbacks():
    # callbacks 
    # checkpoint directory
    checkpointdir = '/tmp/model-ckpt'

    class customLog(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            logging.info('epoch: {:.4f}'.format(epoch + 1))
            logging.info('loss={:.4f}'.format(logs['loss']))
            logging.info('accuracy={:.4f}'.format(logs['accuracy']))
            logging.info('val_accuracy={:.4f}'.format(logs['val_accuracy']))
    #logging.info("{{metricName: accuracy, metricValue: {:.4f}}};{{metricName: loss, metricValue: {:.4f}}}\n".format())

    callbacks = [
        #tf.keras.callbacks.TensorBoard(logdir),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpointdir),
        customLog()
    ]
    return callbacks


# parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
  
    parser.add_argument('--log-path',
                        type=str,
                        default="",
                        help='The number of training steps to perform.')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--drop-out',
                        type=float,
                        default=0.2,
                        help='Drop out rate for training.')
    parser.add_argument('--optimizer',
                        type=str,
                        default="sgd",
                        help='optimizer for training.')
    parser.add_argument('--units',
                        type=int,
                        default=8,
                        help='units for training.')
    parser.add_argument('--layers',
                        type=int,
                        default=1,
                        help='units for training.')
    parser.add_argument('--intializer',
                        type=str,
                        default='glorot_uniform',
                        help='intializer for training.')
    parser.add_argument('--data',
                        type=str,
                        default='gs://featurestore_artifacts/train.csv',
                        help='intializer for training.')
    parser.add_argument('--target',
                        type=str,
                        default='fare_statistics__target',
                        help='target for training.')

    args = parser.parse_known_args()[0]
    return args


def main():

    # parse arguments
    args = parse_arguments()

   
    if args.log_path == "":
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG)
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG,
            filename=args.log_path)
    
    features_outcome=pd.read_csv(args.data)
    features_outcome.dropna(inplace=True)

    


    data_target = features_outcome[args.target]

    data = features_outcome.drop([args.target,'driver_id','event_timestamp'], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data) 

    x_train, x_test, y_train, y_test = train_test_split(data, data_target, test_size=0.2, random_state=0)

     # build and compile model
    learning_rate = float(args.learning_rate)
    logging.info("learning rate : {0}".format(learning_rate))
    model = build_model(learning_rate,float(args.drop_out),args.optimizer,int(args.layers),int(args.units),args.intializer)

  
    history=model.fit(x_train, y_train,
            epochs=15,
            #steps_per_epoch=TF_STEPS_PER_EPOCHS, 
            validation_data=(x_test,y_test),
            validation_steps=1,
            callbacks=get_callbacks())
    print(history)

    logging.info("Training completed.")
    # successful completion
    exit(0)
  

if __name__ == "__main__":
    main()