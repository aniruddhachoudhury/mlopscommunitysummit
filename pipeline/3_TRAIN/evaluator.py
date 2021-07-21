import click
import os
import dill
import json
import logging
import time,uuid
import tensorflow as tf
from pickle import load
import pandas as pd
import ast
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.python.lib.io import file_io
from sklearn.metrics import roc_curve, auc
from sklearn import metrics as ms
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from storage import Storage

from sklearn.model_selection import train_test_split



    
def build_models(learning_rate,drop_out,opt,denselayer,dense_neurons,intializer):
   

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
        model.add(tf.keras.layers.Dense(dense_neurons,kernel_initializer=intializer, activation="relu"))
        dense_neurons *= 2
    model.add(tf.keras.layers.Dropout(drop_out, )) #seed=rng.random()
    
    for _ in range(denselayer):
        model.add(tf.keras.layers.Dense(dense_neurons, kernel_initializer=intializer,activation="relu"))
        dense_neurons /= 2
    # Add the final output layer.
    model.add(tf.keras.layers.Dropout(drop_out))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

  
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    

    return model


@click.command()
@click.option('--probability', default=0.5)
@click.option('--gcs-path', default="gs:///featurestore_artifacts/model")
@click.option('--gcs-path-confusion', default="gs://featurestore_artifacts/taxi/")
@click.option('--mode', default="local")
@click.option('--artifacts', default="local")
def evaluator_model(gcs_path,probability,gcs_path_confusion,mode,artifacts):


        artifacts=ast.literal_eval(artifacts)

        learning_rate=artifacts['learning-rate']
        optimizer=artifacts['optimizer']
        dropout=artifacts['drop-out']
        units=artifacts['units']
        layers=artifacts['layers']
        intializer=artifacts['intializer']

        print(learning_rate,optimizer,dropout,units,layers,intializer)
        
        #Load Model Artifacts for training
        with open("/mnt/training.data", 'rb') as in_f:
                x_train= dill.load(in_f)

        with open("/mnt/validation.data", 'rb') as in_f:
                x_test= dill.load(in_f)
        
        with open("/mnt/trainingtarget.data", 'rb') as in_f:
                y_train= dill.load(in_f)
  
        with open("/mnt/validationtarget.data", 'rb') as in_f:
                y_test= dill.load(in_f)

        
        #features_outcome=pd.read_csv('gs://featurestore_artifacts/train.csv')
       # features_outcome.dropna(inplace=True)

       # data_target = features_outcome['fare_statistics__target']

      #  data = features_outcome.drop(['fare_statistics__target','driver_id','event_timestamp'], axis=1)
     #   x_train, x_test, y_train, y_test = train_test_split(data, data_target, test_size=0.2, random_state=0)


        tensorboard_logs="/mnt/training/"
        #data =((x_train, y_train), (x_test, y_test))

        
        #Model Evaluation
        #model_results =pd.read_csv('/mnt/results.csv')
       # max_index=model_results.Accuracy.argmax()
       # model_results_list=model_results.loc[max_index:max_index+1,:].values.tolist()[0]

       # print(model_results_list)
        model_output_base_path="/mnt/saved_model"
        #metrics = [keras.metrics.BinaryAccuracy(name='accuracy')]

        #model_build(model_results_list,metrics,data,tensorboard_logs,model_output_base_path)
        model = build_models(float(learning_rate),float(dropout),optimizer.lower(),int(layers),int(units),intializer)
       # model = build_model(float(learning_rate),float(dropout),optimizer.lower(),x_train,int(units))
        model.fit(x_train, y_train,
            epochs=15,
            #steps_per_epoch=TF_STEPS_PER_EPOCHS, 
            validation_data=(x_test,y_test),
            validation_steps=1,
            callbacks=[tf.keras.callbacks.TensorBoard(tensorboard_logs)])
        model.save(model_output_base_path)   
        
        new_model = tf.keras.models.load_model(model_output_base_path)
        # Check its architecture
        print(new_model.summary())
        ann_prediction =new_model.predict(x_test,batch_size=32)
        ann_prediction = (ann_prediction > probability)*1 

        # Compute error between predicted data and true response and display it in confusion matrix
    
        acc_ann = round(ms.accuracy_score(y_test, ann_prediction) * 100, 2)
        print(acc_ann)
        metrics = {
                'metrics': [{
                    'name': 'Accuracy',
                    'numberValue':  acc_ann,
                    'format': "RAW",
                }]}    

        with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
          json.dump(metrics,f)
        #
        vocab =[0,1]
        cm = confusion_matrix(y_test, ann_prediction > probability,labels=vocab)


        data_conf= []
        for target_index, target_row in enumerate(cm):
            for predicted_index, count in enumerate(target_row):
                  data_conf.append((vocab[target_index], vocab[predicted_index], count))
    
        
        df_cm = pd.DataFrame(data_conf, columns=['target', 'predicted', 'count'])
        cm_file = os.path.join(gcs_path_confusion, 'confusion_matrix.csv')
        with file_io.FileIO(cm_file, 'w') as f:
            df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)   
        
      
        
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, ann_prediction)

        df_roc = pd.DataFrame({'fpr': false_positive_rate, 'tpr': true_positive_rate, 'thresholds': thresholds})
        roc_file = os.path.join(gcs_path_confusion, 'roc.csv')
        with file_io.FileIO(roc_file, 'w') as f:
            df_roc.to_csv(f, columns=['fpr', 'tpr', 'thresholds'], header=False,index=False)
        
        unique = uuid.uuid4().hex[:12]
        tensorboard_gcs_logs=f'gs://featurestore_artifacts/taxi/train/{unique}/logs'
        Storage.upload(tensorboard_logs,tensorboard_gcs_logs)


        metadata = {
            'outputs': [{
            'type': 'roc',
            'format': 'csv',
            'schema': [
                {'name': 'fpr', 'type': 'NUMBER'},
                {'name': 'tpr', 'type': 'NUMBER'},
                {'name': 'thresholds', 'type': 'NUMBER'},
            ],
            'source': roc_file
            },
           
            {
                        'type': 'tensorboard',
                        'source': tensorboard_gcs_logs,        
                },
            {
                'type': 'confusion_matrix',
                'format': 'csv',
                'schema': [
                    {'name': 'target', 'type': 'CATEGORY'},
                    {'name': 'predicted', 'type': 'CATEGORY'},
                    {'name': 'count', 'type': 'NUMBER'},
                ],
                'source': cm_file,
                # Convert vocab to string because for bealean values we want "True|False" to match csv data.
                'labels': list(map(str, vocab)),
                }]}
       
        with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        
        if mode!= 'local':
                print("uploading to {0}".format(gcs_path))
                Storage.upload(model_output_base_path,gcs_path)
        else:
                print("Model will not be uploaded")
                pass
        

if __name__ == "__main__":
    evaluator_model()