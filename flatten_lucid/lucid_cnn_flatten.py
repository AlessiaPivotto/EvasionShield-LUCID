# Copyright (c) 2022 @ FBK - Fondazione Bruno Kessler
# Author: Roberto Doriguzzi-Corin
# Project: LUCID: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Sample commands
# Training: python3 lucid_cnn_flatten.py --train ./DATASET_RIDOTTO/  --epochs 100 -cv 5
# Testing: python3  lucid_cnn_flatten.py --predict ./DATASET_RIDOTTO/ --model ./DATASET_RIDOTTO/10t-10n-SYN2020-LUCID-FLATTEN.h5

import tensorflow as tf
import numpy as np
import random as rn
import os
import csv
import pprint
import sys
import glob
import time
import argparse
from util_functions import *
# Seed Random Numbers
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)

from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lucid_dataset_parser import *

import tensorflow.keras.backend as K
tf.random.set_seed(SEED)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)

OUTPUT_FOLDER = "./output/"

VAL_HEADER = ['Model', 'Samples', 'Accuracy', 'F1Score', 'Hyper-parameters','Validation Set']
PREDICT_HEADER = ['Model', 'Time', 'Packets', 'Samples', 'DDOS%', 'Accuracy', 'F1Score', 'TPR', 'FPR','TNR', 'FNR', 'Source']

# hyperparameters for flattened model
PATIENCE = 10
DEFAULT_EPOCHS = 1000
hyperparamters = {
    "learning_rate": [0.1, 0.01, 0.001],
    "batch_size": [512, 1024, 2048],
    "hidden_units": [64, 128, 256],
    "regularization": [None, 'l1', 'l2'],
    "dropout": [None, 0.2, 0.5]
}

def DenseModel(model_name, input_shape, hidden_units=128, learning_rate=0.01, regularization=None, dropout=None):
    """
    Create a dense neural network model for flattened input.
    Input shape should be (2100,) for 100x21 flattened matrix.
    """
    K.clear_session()

    model = Sequential(name=model_name)
    regularizer = None
    
    if regularization == 'l1':
        regularizer = tf.keras.regularizers.l1(0.01)
    elif regularization == 'l2':
        regularizer = tf.keras.regularizers.l2(0.01)

    # Input layer - expects flattened input (2100 features for 100x21)
    model.add(Dense(hidden_units, input_shape=input_shape, kernel_regularizer=regularizer, name='dense0'))
    model.add(Activation('relu'))
    
    if dropout is not None and type(dropout) == float:
        model.add(Dropout(dropout))

    # Additional hidden layer for better feature learning
    model.add(Dense(hidden_units // 2, kernel_regularizer=regularizer, name='dense1'))
    model.add(Activation('relu'))
    
    if dropout is not None and type(dropout) == float:
        model.add(Dropout(dropout))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid', name='output'))

    print(model.summary())
    compileModel(model, learning_rate)
    return model

def compileModel(model, lr):
    # optimizer = SGD(learning_rate=lr, momentum=0.0, decay=0.0, nesterov=False)
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # here we specify the loss function

def main(argv):
    help_string = 'Usage: python3 lucid_cnn_flatten.py --train <dataset_folder> -e <epochs>'

    parser = argparse.ArgumentParser(
        description='DDoS attacks detection with dense neural networks (flattened input)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--train', nargs='+', type=str,
                        help='Start the training process')

    parser.add_argument('-e', '--epochs', default=DEFAULT_EPOCHS, type=int,
                        help='Training iterations')

    parser.add_argument('-cv', '--cross_validation', default=0, type=int,
                        help='Number of folds for cross-validation (default 0)')

    parser.add_argument('-a', '--attack_net', default=None, type=str,
                        help='Subnet of the attacker (used to compute the detection accuracy)')

    parser.add_argument('-v', '--victim_net', default=None, type=str,
                        help='Subnet of the victim (used to compute the detection accuracy)')

    parser.add_argument('-p', '--predict', nargs='?', type=str,
                        help='Perform a prediction on pre-preprocessed data')

    parser.add_argument('-pl', '--predict_live', nargs='?', type=str,
                        help='Perform a prediction on live traffic')

    parser.add_argument('-i', '--iterations', default=1, type=int,
                        help='Predict iterations')

    parser.add_argument('-m', '--model', type=str,
                        help='File containing the model')

    parser.add_argument('-y', '--dataset_type', default=None, type=str,
                        help='Type of the dataset. Available options are: DOS2017, DOS2018, DOS2019, SYN2020')

    args = parser.parse_args()

    if os.path.isdir(OUTPUT_FOLDER) == False:
        os.mkdir(OUTPUT_FOLDER)

    if args.train is not None:
        main_folder = args.train[0].rstrip('/')
        
        # First, check if there are general train/val files in the main directory
        main_train_files = glob.glob(main_folder + "/*-train.hdf5")
        main_val_files = glob.glob(main_folder + "/*-val.hdf5")
        
        print(f"Checking main folder: {main_folder}")
        print(f"Train files found: {main_train_files}")
        print(f"Val files found: {main_val_files}")
        
        if main_train_files and main_val_files:
            # Use the general dataset files from the main directory
            print(f"Found general train/val files in main directory: {main_folder}")
            subfolders = [main_folder + "/"]
        else:
            # Look for subfolders with individual datasets
            subfolders = glob.glob(args.train[0] + "/*/")
            if len(subfolders) == 0:  # for the case in which there is only one folder, and this folder is args.dataset_folder[0]
                subfolders = [args.train[0] + "/"]
            else:
                subfolders = sorted(subfolders)
        
        for full_path in subfolders:
            full_path = full_path.replace("//", "/")  # remove double slashes when needed
            folder = full_path.split("/")[-2]
            dataset_folder = full_path
            
            # Try to load from the current folder first
            try:
                X_train, Y_train = load_dataset(dataset_folder + "/*" + '-train.hdf5')
                X_val, Y_val = load_dataset(dataset_folder + "/*" + '-val.hdf5')
                train_file = glob.glob(dataset_folder + "/*" + '-train.hdf5')[0]
                print(f"Using dataset files from: {dataset_folder}")
            except (FileNotFoundError, IndexError):
                # If files not found in current folder, try parent directory (for subfolder case)
                parent_folder = os.path.dirname(dataset_folder.rstrip('/'))
                print(f"Train/Val files not found in {dataset_folder}, trying parent directory {parent_folder}")
                try:
                    X_train, Y_train = load_dataset(parent_folder + "/*" + '-train.hdf5')
                    X_val, Y_val = load_dataset(parent_folder + "/*" + '-val.hdf5')
                    train_file = glob.glob(parent_folder + "/*" + '-train.hdf5')[0]
                    print(f"Using dataset files from parent: {parent_folder}")
                except (FileNotFoundError, IndexError):
                    print(f"Warning: No train/val files found for {folder}, skipping...")
                    continue

            X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
            X_val, Y_val = shuffle(X_val, Y_val, random_state=SEED)

            # Flatten the input data from (batch, 100, 21, 1) to (batch, 2100)
            if len(X_train.shape) > 2:
                print(f"Original training shape: {X_train.shape}")
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_val = X_val.reshape(X_val.shape[0], -1)
                print(f"Flattened training shape: {X_train.shape}")

            # get the time_window and the flow_len from the filename
            filename = train_file.split('/')[-1].strip()
            time_window = int(filename.split('-')[0].strip().replace('t', ''))
            max_flow_len = int(filename.split('-')[1].strip().replace('n', ''))
            dataset_name = filename.split('-')[2].strip()

            print("\nCurrent dataset folder: ", dataset_folder)
            print(f"Input shape: {X_train.shape[1:]}")

            model_name = dataset_name + "-LUCID-FLATTEN"
            keras_classifier = KerasClassifier(build_fn=DenseModel, model_name=model_name, input_shape=X_train.shape[1:])
            rnd_search_cv = GridSearchCV(keras_classifier, hyperparamters, cv=args.cross_validation if args.cross_validation > 1 else [(slice(None), slice(None))], refit=True, return_train_score=True)

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
            best_model_filename = OUTPUT_FOLDER + str(time_window) + 't-' + str(max_flow_len) + 'n-' + model_name
            mc = ModelCheckpoint(best_model_filename + '.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
            
            # With K-Fold cross-validation, the validation set is only used for early stopping
            rnd_search_cv.fit(X_train, Y_train, epochs=args.epochs, validation_data=(X_val, Y_val), callbacks=[es, mc])

            # With refit=True (default) GridSearchCV refits the model on the whole training set (no folds) with the best
            # hyper-parameters and makes the resulting model available as rnd_search_cv.best_estimator_.model
            best_model = rnd_search_cv.best_estimator_.model

            # We overwrite the checkpoint models with the one trained on the whole training set (not only k-1 folds)
            best_model.save(best_model_filename + '.h5')

            # Alternatively, to save time, one could set refit=False and load the best model from the filesystem to test its performance
            # best_model = load_model(best_model_filename + '.h5')

            Y_pred_val = (best_model.predict(X_val) > 0.5)
            Y_true_val = Y_val.reshape((Y_val.shape[0], 1))
            f1_score_val = f1_score(Y_true_val, Y_pred_val)
            accuracy = accuracy_score(Y_true_val, Y_pred_val)

            # save best model performance on the validation set
            val_file = open(best_model_filename + '.csv', 'w', newline='')
            val_file.truncate(0)  # clean the file content (as we open the file in append mode)
            val_writer = csv.DictWriter(val_file, fieldnames=VAL_HEADER)
            val_writer.writeheader()
            val_file.flush()
            row = {'Model': model_name, 'Samples': Y_pred_val.shape[0], 'Accuracy': '{:05.4f}'.format(accuracy), 'F1Score': '{:05.4f}'.format(f1_score_val),
                  'Hyper-parameters': rnd_search_cv.best_params_, "Validation Set": glob.glob(dataset_folder + "/*" + '-val.hdf5')[0]}
            val_writer.writerow(row)
            val_file.close()

            print("Best parameters: ", rnd_search_cv.best_params_)
            print("Best model path: ", best_model_filename)
            print("F1 Score of the best model on the validation set: ", f1_score_val)

    if args.predict is not None:
        predict_file = open(OUTPUT_FOLDER + 'predictions-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a', newline='')
        predict_file.truncate(0)  # clean the file content (as we open the file in append mode)
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()

        iterations = args.iterations

        dataset_filelist = glob.glob(args.predict + "/*test.hdf5")

        if args.model is not None:
            model_list = [args.model]
        else:
            model_list = glob.glob(args.predict + "/*.h5")

        for model_path in model_list:
            model_filename = model_path.split('/')[-1].strip()
            filename_prefix = model_filename.split('-')[0].strip() + '-' + model_filename.split('-')[1].strip() + '-'
            model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
            model = load_model(model_path)

            # warming up the model (necessary for the GPU)
            warm_up_file = dataset_filelist[0]
            filename = warm_up_file.split('/')[-1].strip()
            if filename_prefix in filename:
                X, Y = load_dataset(warm_up_file)
                # Flatten the input data if needed
                if len(X.shape) > 2:
                    X = X.reshape(X.shape[0], -1)
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)

            for dataset_file in dataset_filelist:
                filename = dataset_file.split('/')[-1].strip()
                if filename_prefix in filename:
                    X, Y = load_dataset(dataset_file)
                    
                    # Flatten the input data if needed
                    if len(X.shape) > 2:
                        print(f"Original test shape: {X.shape}")
                        X = X.reshape(X.shape[0], -1)
                        print(f"Flattened test shape: {X.shape}")
                    
                    [packets] = count_packets_in_dataset([X])

                    Y_pred = None
                    Y_true = Y
                    avg_time = 0
                    for iteration in range(iterations):
                        pt0 = time.time()
                        Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)
                        pt1 = time.time()
                        avg_time += pt1 - pt0

                    avg_time = avg_time / iterations

                    report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, filename, avg_time, predict_writer)
                    predict_file.flush()

        predict_file.close()

    if args.predict_live is not None:
        predict_file = open(OUTPUT_FOLDER + 'predictions-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a', newline='')
        predict_file.truncate(0)  # clean the file content (as we open the file in append mode)
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()

        if args.predict_live is None:
            print("Please specify a valid network interface or pcap file!")
            exit(-1)
        elif args.predict_live.endswith('.pcap'):
            pcap_file = args.predict_live
            cap = pyshark.FileCapture(pcap_file)
            data_source = pcap_file.split('/')[-1].strip()
        else:
            cap = pyshark.LiveCapture(interface=args.predict_live)
            data_source = args.predict_live

        print("Prediction on network traffic from: ", data_source)

        # load the labels, if available
        labels = parse_labels(args.dataset_type, args.attack_net, args.victim_net)

        # do not forget command sudo ./jetson_clocks.sh on the TX2 board before testing
        if args.model is not None and args.model.endswith('.h5'):
            model_path = args.model
        else:
            print("No valid model specified!")
            exit(-1)

        model_filename = model_path.split('/')[-1].strip()
        filename_prefix = model_filename.split('n')[0] + 'n-'
        time_window = int(filename_prefix.split('t-')[0])
        max_flow_len = int(filename_prefix.split('t-')[1].split('n-')[0])
        model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
        model = load_model(args.model)

        mins, maxs = static_min_max(time_window)

        while (True):
            samples = process_live_traffic(cap, args.dataset_type, labels, max_flow_len, traffic_type="all", time_window=time_window)
            if len(samples) > 0:
                X, Y_true, keys = dataset_to_list_of_fragments(samples)
                X = np.array(normalize_and_padding(X, mins, maxs, max_flow_len))
                if labels is not None:
                    Y_true = np.array(Y_true)
                else:
                    Y_true = None

                # Flatten the input data for the dense model
                if len(X.shape) > 2:
                    X = X.reshape(X.shape[0], -1)
                    
                pt0 = time.time()
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)
                pt1 = time.time()
                prediction_time = pt1 - pt0

                [packets] = count_packets_in_dataset([X])
                report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, data_source, prediction_time, predict_writer)
                predict_file.flush()

            elif isinstance(cap, pyshark.FileCapture) == True:
                print("\nNo more packets in file ", data_source)
                break

        predict_file.close()

def report_results(Y_true, Y_pred, packets, model_name, data_source, prediction_time, writer):
    ddos_rate = '{:04.3f}'.format(sum(Y_pred) / Y_pred.shape[0])

    if Y_true is not None and len(Y_true.shape) > 0:  # if we have the labels, we can compute the classification accuracy
        Y_true = Y_true.reshape((Y_true.shape[0], 1))
        accuracy = accuracy_score(Y_true, Y_pred)

        f1 = f1_score(Y_true, Y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0, 1]).ravel()
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        tpr = tp / (tp + fn)

        row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets,
               'Samples': Y_pred.shape[0], 'DDOS%': ddos_rate, 'Accuracy': '{:05.4f}'.format(accuracy), 'F1Score': '{:05.4f}'.format(f1),
               'TPR': '{:05.4f}'.format(tpr), 'FPR': '{:05.4f}'.format(fpr), 'TNR': '{:05.4f}'.format(tnr), 'FNR': '{:05.4f}'.format(fnr), 'Source': data_source}
    else:
        row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets,
               'Samples': Y_pred.shape[0], 'DDOS%': ddos_rate, 'Accuracy': "N/A", 'F1Score': "N/A",
               'TPR': "N/A", 'FPR': "N/A", 'TNR': "N/A", 'FNR': "N/A", 'Source': data_source}
    pprint.pprint(row, sort_dicts=False)
    writer.writerow(row)

if __name__ == "__main__":
    main(sys.argv[1:])
