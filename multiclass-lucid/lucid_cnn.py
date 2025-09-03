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
# Training: python3 lucid_cnn.py --train ./sample-dataset/  --epochs 100 -cv 5
# Testing: python3  lucid_cnn.py --predict ./sample-dataset/ --model ./sample-dataset/10t-10n-SYN2020-LUCID.h5

import tensorflow as tf
import numpy as np
import random as rn
import os
import csv
import pprint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from util_functions import *
# Seed Random Numbers
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)

from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D
from tensorflow.keras.layers import Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.utils import to_categorical
from lucid_dataset_parser import *

import tensorflow.keras.backend as K
tf.random.set_seed(SEED)
K.set_image_data_format('channels_last')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)

# TODO: Fix model name and dataset name

OUTPUT_FOLDER = "./output/"

VAL_HEADER = ['Model', 'Samples', 'Accuracy', 'F1Score', 'Hyper-parameters','Validation Set']
PREDICT_HEADER = ['Model', 'Time', 'Packets', 'Samples', 'Attack%', 'Accuracy', 'F1Score', 'Source']

# hyperparameters
PATIENCE = 10
DEFAULT_EPOCHS = 1000
hyperparamters = {
    "learning_rate": [0.1,0.01],
    "batch_size": [1024,2048],
    "kernels": [32,64],
    "regularization" : [None,'l1'],
    "dropout" : [None,0.2]
}

def Conv2DModel(model_name,input_shape,kernel_col, kernels=64,kernel_rows=3,learning_rate=0.01,regularization=None,dropout=None):
    K.clear_session()

    model = Sequential(name=model_name)
    regularizer = regularization

    model.add(Conv2D(kernels, (kernel_rows,kernel_col), strides=(1, 1), input_shape=input_shape, kernel_regularizer=regularizer, name='conv0'))
    if dropout != None and type(dropout) == float:
        model.add(Dropout(dropout))
    model.add(Activation('relu'))

    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    # Output layer for multiclass classification
    model.add(Dense(NUM_CLASSES, activation='softmax', name='fc1'))

    print(model.summary())
    compileModel(model, learning_rate)
    return model

def compileModel(model,lr):
    # optimizer = SGD(learning_rate=lr, momentum=0.0, decay=0.0, nesterov=False)
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # Use categorical crossentropy for multiclass classification
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])  # here we specify the loss function

def main(argv):
    help_string = 'Usage: python3 lucid_cnn.py --train <dataset_folder> -e <epocs>'

    parser = argparse.ArgumentParser(
        description='DDoS attacks detection with convolutional neural networks',
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
        subfolders = glob.glob(args.train[0] +"/*/")
        if len(subfolders) == 0: # for the case in which the is only one folder, and this folder is args.dataset_folder[0]
            subfolders = [args.train[0] + "/"]
        else:
            subfolders = sorted(subfolders)
        for full_path in subfolders:
            full_path = full_path.replace("//", "/")  # remove double slashes when needed
            folder = full_path.split("/")[-2]
            dataset_folder = full_path
            X_train, Y_train = load_dataset(dataset_folder + "/*" + '-train.hdf5')
            X_val, Y_val = load_dataset(dataset_folder + "/*" + '-val.hdf5')
            
            # Convert labels to categorical for multiclass
            Y_train_cat = to_categorical(Y_train, num_classes=NUM_CLASSES)
            Y_val_cat = to_categorical(Y_val, num_classes=NUM_CLASSES)

            X_train, Y_train_cat = shuffle(X_train, Y_train_cat, random_state=SEED)
            X_val, Y_val_cat = shuffle(X_val, Y_val_cat, random_state=SEED)

            # get the time_window and the flow_len from the filename
            train_file = glob.glob(dataset_folder + "/*" + '-train.hdf5')[0]
            filename = train_file.split('/')[-1].strip()
            time_window = int(filename.split('-')[0].strip().replace('t', ''))
            max_flow_len = int(filename.split('-')[1].strip().replace('n', ''))
            dataset_name = filename.split('-')[2].strip()

            print ("\nCurrent dataset folder: ", dataset_folder)

            model_name = dataset_name + "-LUCID"
            keras_classifier = KerasClassifier(build_fn=Conv2DModel,model_name=model_name, input_shape=X_train.shape[1:],kernel_col=X_train.shape[2])
            rnd_search_cv = GridSearchCV(keras_classifier, hyperparamters, cv=args.cross_validation if args.cross_validation > 1 else [(slice(None), slice(None))], refit=True, return_train_score=True)

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
            best_model_filename = OUTPUT_FOLDER + str(time_window) + 't-' + str(max_flow_len) + 'n-' + model_name
            mc = ModelCheckpoint(best_model_filename + '.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
            # With K-Fold cross-validation, the validation set is only used for early stopping
            rnd_search_cv.fit(X_train, Y_train_cat, epochs=args.epochs, validation_data=(X_val, Y_val_cat), callbacks=[es, mc])

            # With refit=True (default) GridSearchCV refits the model on the whole training set (no folds) with the best
            # hyper-parameters and makes the resulting model available as rnd_search_cv.best_estimator_.model
            best_model = rnd_search_cv.best_estimator_.model

            # We overwrite the checkpoint models with the one trained on the whole training set (not only k-1 folds)
            best_model.save(best_model_filename + '.h5')

            # Alternatively, to save time, one could set refit=False and load the best model from the filesystem to test its performance
            #best_model = load_model(best_model_filename + '.h5')

            Y_pred_val_prob = best_model.predict(X_val)
            Y_pred_val = np.argmax(Y_pred_val_prob, axis=1)
            Y_true_val = Y_val
            f1_score_val = f1_score(Y_true_val, Y_pred_val, average='weighted')
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
        
        # Extract dataset name from the prediction path
        dataset_name = args.predict.rstrip('/').split('/')[-1] if args.predict else "UNKNOWN-DATASET"

        if args.model is not None:
            model_list = [args.model]
        else:
            model_list = glob.glob(args.predict + "/*.h5")

        for model_path in model_list:
            model_filename = model_path.split('/')[-1].strip()
            filename_prefix = model_filename.split('-')[0].strip() + '-' + model_filename.split('-')[1].strip() + '-'
            
            # Use dataset name instead of trying to parse model filename
            model_name_string = dataset_name
            model = load_model(model_path)

            # warming up the model (necessary for the GPU)
            warm_up_file = dataset_filelist[0]
            filename = warm_up_file.split('/')[-1].strip()
            if filename_prefix in filename:
                X, Y = load_dataset(warm_up_file)
                Y_pred_prob = model.predict(X, batch_size=2048)
                Y_pred = np.argmax(Y_pred_prob, axis=1)

            for dataset_file in dataset_filelist:
                filename = dataset_file.split('/')[-1].strip()
                if filename_prefix in filename:
                    X, Y = load_dataset(dataset_file)
                    [packets] = count_packets_in_dataset([X])

                    # Convert to categorical if not already
                    if len(Y.shape) == 1:
                        Y_cat = to_categorical(Y, num_classes=NUM_CLASSES)
                    else:
                        Y_cat = Y

                    Y_pred = None
                    Y_true = Y  # Keep original for metrics
                    avg_time = 0
                    
                    for iteration in range(iterations):
                        pt0 = time.time()
                        Y_pred_prob = model.predict(X, batch_size=2048)
                        Y_pred = np.argmax(Y_pred_prob, axis=1)
                        pt1 = time.time()
                        avg_time += pt1 - pt0

                    avg_time = avg_time / iterations

                    # Use dataset name instead of just filename for better plot labeling
                    report_results(Y_true, Y_pred, packets, model_name_string, dataset_name, avg_time, predict_writer, Y_pred_prob)
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
            cap =  pyshark.LiveCapture(interface=args.predict_live)
            data_source = args.predict_live

        print ("Prediction on network traffic from: ", data_source)

        # load the labels, if available
        labels = parse_labels(args.dataset_type, args.attack_net, args.victim_net)

        # do not forget command sudo ./jetson_clocks.sh on the TX2 board before testing
        if args.model is not None and args.model.endswith('.h5'):
            model_path = args.model
        else:
            print ("No valid model specified!")
            exit(-1)

        model_filename = model_path.split('/')[-1].strip()
        filename_prefix = model_filename.split('n')[0] + 'n-'
        time_window = int(filename_prefix.split('t-')[0])
        max_flow_len = int(filename_prefix.split('t-')[1].split('n-')[0])
        
        # Use data source name instead of trying to parse model filename
        model_name_string = data_source.split('.')[0] if data_source else "LIVE-TRAFFIC"
        model = load_model(args.model)

        mins, maxs = static_min_max(time_window)

        while (True):
            samples = process_live_traffic(cap, args.dataset_type, labels, max_flow_len, traffic_type="all", time_window=time_window)
            if len(samples) > 0:
                X,Y_true,keys = dataset_to_list_of_fragments(samples)
                X = np.array(normalize_and_padding(X, mins, maxs, max_flow_len))
                if labels is not None:
                    Y_true = np.array(Y_true)
                else:
                    Y_true = None

                X = np.expand_dims(X, axis=3)
                pt0 = time.time()
                Y_pred_prob = model.predict(X, batch_size=2048)
                Y_pred = np.argmax(Y_pred_prob, axis=1)
                pt1 = time.time()
                prediction_time = pt1 - pt0

                [packets] = count_packets_in_dataset([X])
                report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, data_source, prediction_time, predict_writer, Y_pred_prob)
                predict_file.flush()

            elif isinstance(cap, pyshark.FileCapture) == True:
                print("\nNo more packets in file ", data_source)
                break

        predict_file.close()

def calculate_multiclass_roc(Y_true, Y_pred_proba, num_classes, class_names=None):
    """
    Calculate ROC curves and AUC scores for multiclass classification
    
    Args:
        Y_true: True labels (categorical or integer format)
        Y_pred_proba: Predicted probabilities (shape: [n_samples, n_classes])
        num_classes: Number of classes
        class_names: Optional list of class names
    
    Returns:
        Dictionary containing ROC data for each class
    """
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    # Convert categorical to binary format if needed
    if len(Y_true.shape) > 1:
        Y_true_binary = Y_true
    else:
        Y_true_binary = label_binarize(Y_true, classes=list(range(num_classes)))
        if num_classes == 2:
            Y_true_binary = np.hstack([1 - Y_true_binary, Y_true_binary])
    
    # Ensure Y_pred_proba is in the right format
    if len(Y_pred_proba.shape) == 1:
        Y_pred_proba = np.eye(num_classes)[Y_pred_proba]
    
    roc_data = {}
    
    # Calculate ROC curve and AUC for each class
    for i in range(num_classes):
        try:
            fpr, tpr, _ = roc_curve(Y_true_binary[:, i], Y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            roc_data[class_names[i]] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc,
                'class_idx': i
            }
        except Exception as e:
            print(f"Warning: Could not calculate ROC for {class_names[i]}: {e}")
            roc_data[class_names[i]] = {
                'fpr': np.array([0, 1]),
                'tpr': np.array([0, 1]),
                'auc': 0.5,
                'class_idx': i
            }
    
    # Calculate macro and micro averages
    try:
        # Micro-average ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(Y_true_binary.ravel(), Y_pred_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        # Macro-average ROC curve
        all_fpr = np.unique(np.concatenate([roc_data[name]['fpr'] for name in class_names]))
        mean_tpr = np.zeros_like(all_fpr)
        for name in class_names:
            mean_tpr += np.interp(all_fpr, roc_data[name]['fpr'], roc_data[name]['tpr'])
        mean_tpr /= num_classes
        roc_auc_macro = auc(all_fpr, mean_tpr)
        
        roc_data['micro_avg'] = {'fpr': fpr_micro, 'tpr': tpr_micro, 'auc': roc_auc_micro}
        roc_data['macro_avg'] = {'fpr': all_fpr, 'tpr': mean_tpr, 'auc': roc_auc_macro}
        
    except Exception as e:
        print(f"Warning: Could not calculate averaged ROC curves: {e}")
        roc_data['micro_avg'] = {'fpr': np.array([0, 1]), 'tpr': np.array([0, 1]), 'auc': 0.5}
        roc_data['macro_avg'] = {'fpr': np.array([0, 1]), 'tpr': np.array([0, 1]), 'auc': 0.5}
    
    return roc_data

def extract_dataset_name(data_source):
    """
    Extract dataset name from the data source string.
    """
    if isinstance(data_source, str):
        # Extract from path like "./DATASETS/MANIPULATED-MULTICLASS/" or "10t-100n-DOS2019-test.hdf5"
        if 'MANIPULATED-MULTICLASS' in data_source:
            return 'MANIPULATED-MULTICLASS'
        elif 'MULTICLASS-BASELINE' in data_source:
            return 'MULTICLASS-BASELINE'
        elif 'DOS2019' in data_source:
            return 'DOS2019-LUCID'
        elif 'CICIDS' in data_source:
            return 'CICIDS'
        elif '.hdf5' in data_source:
            # Extract dataset name from HDF5 filename like "10t-100n-DOS2019-test.hdf5"
            basename = data_source.split('/')[-1].replace('-test.hdf5', '').replace('-train.hdf5', '')
            if basename:
                return basename
        elif '/' in data_source:
            # Extract from folder path
            parts = data_source.strip('/').split('/')
            if parts:
                return parts[-1]  # Last part of path
        else:
            # Use the data source as-is if it's a simple string
            return data_source.split('.')[0]  # Remove file extension if any
    
    return "UNKNOWN-DATASET"

def create_roc_plots(roc_data, model_name, data_source, save_plot=True):
    """
    Create and save ROC curve plots for multiclass classification
    """

    dataset_name = extract_dataset_name(data_source)

    plt.figure(figsize=(12, 8))
    
    # Define colors for classes
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black'])
    
    # Plot ROC curve for each class
    for class_name, color in zip([k for k in roc_data.keys() if k not in ['micro_avg', 'macro_avg']], colors):
        data = roc_data[class_name]
        plt.plot(data['fpr'], data['tpr'], color=color, lw=2,
                label=f'{class_name} (AUC = {data["auc"]:.3f})')
    
    # Plot micro and macro averages
    if 'micro_avg' in roc_data:
        plt.plot(roc_data['micro_avg']['fpr'], roc_data['micro_avg']['tpr'],
                label=f'Micro-avg (AUC = {roc_data["micro_avg"]["auc"]:.3f})',
                color='deeppink', linestyle=':', linewidth=4)
    
    if 'macro_avg' in roc_data:
        plt.plot(roc_data['macro_avg']['fpr'], roc_data['macro_avg']['tpr'],
                label=f'Macro-avg (AUC = {roc_data["macro_avg"]["auc"]:.3f})',
                color='navy', linestyle=':', linewidth=4)
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {model_name}\nDataset: {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"roc_curves_{dataset_name}_{model_name}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"ROC curves plot saved as: {filename}")
    
    plt.close()
    
    return roc_data

def analyze_roc_performance(roc_data, threshold_analysis=True):
    """
    Analyze ROC performance and suggest improvements
    """
    print("\n" + "="*60)
    print("🎯 ROC ANALYSIS AND PERFORMANCE INSIGHTS")
    print("="*60)
    
    # Overall performance assessment
    micro_auc = roc_data.get('micro_avg', {}).get('auc', 0)
    macro_auc = roc_data.get('macro_avg', {}).get('auc', 0)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Micro-average AUC: {micro_auc:.4f}")
    print(f"   Macro-average AUC: {macro_auc:.4f}")
    
    if micro_auc > 0.9:
        print("   ✅ Excellent overall performance!")
    elif micro_auc > 0.8:
        print("   👍 Good overall performance")
    elif micro_auc > 0.7:
        print("   ⚠️  Fair performance - room for improvement")
    else:
        print("   ❌ Poor performance - significant improvements needed")
    
    # Per-class analysis
    print(f"\n📈 PER-CLASS ANALYSIS:")
    class_aucs = []
    poor_classes = []
    excellent_classes = []
    
    for class_name, data in roc_data.items():
        if class_name not in ['micro_avg', 'macro_avg']:
            auc_score = data['auc']
            class_aucs.append((class_name, auc_score))
            
            if auc_score < 0.7:
                poor_classes.append((class_name, auc_score))
            elif auc_score > 0.9:
                excellent_classes.append((class_name, auc_score))
    
    # Sort by AUC score
    class_aucs.sort(key=lambda x: x[1], reverse=True)
    
    print("   Best performing classes:")
    for class_name, auc_score in class_aucs[:5]:
        print(f"     {class_name}: AUC = {auc_score:.4f}")
    
    if poor_classes:
        print("   ⚠️  Classes needing improvement (AUC < 0.7):")
        for class_name, auc_score in poor_classes:
            print(f"     {class_name}: AUC = {auc_score:.4f}")
    
    # Recommendations
    print(f"\n💡 IMPROVEMENT RECOMMENDATIONS:")
    
    if len(poor_classes) > 0:
        print("   🔧 For poor-performing classes:")
        print("     - Increase class weights for minority classes")
        print("     - Use focal loss to focus on hard examples")
        print("     - Apply data augmentation for minority classes")
        print("     - Consider ensemble methods")
    
    if macro_auc < micro_auc - 0.1:
        print("   ⚖️  Class imbalance detected:")
        print("     - Macro AUC significantly lower than Micro AUC")
        print("     - Apply stronger class balancing techniques")
        print("     - Consider cost-sensitive learning")
    
    if micro_auc < 0.8:
        print("   🏗️  General model improvements:")
        print("     - Increase model complexity (more layers/neurons)")
        print("     - Improve feature engineering")
        print("     - Use advanced regularization techniques")
        print("     - Consider different architectures (ResNet, DenseNet)")
    
    return {
        'micro_auc': micro_auc,
        'macro_auc': macro_auc,
        'poor_classes': poor_classes,
        'excellent_classes': excellent_classes,
        'class_aucs': class_aucs
    }

def roc_based_threshold_optimization(Y_true, Y_pred_proba, num_classes):
    """
    Optimize classification thresholds based on ROC analysis for better multiclass performance
    """
    print("\n🔧 ROC-BASED THRESHOLD OPTIMIZATION")
    print("="*50)
    
    # Convert to binary format if needed
    if len(Y_true.shape) == 1:
        Y_true_binary = label_binarize(Y_true, classes=list(range(num_classes)))
        if num_classes == 2:
            Y_true_binary = np.hstack([1 - Y_true_binary, Y_true_binary])
    else:
        Y_true_binary = Y_true
    
    optimal_thresholds = []
    improvements = []
    
    for class_idx in range(num_classes):
        try:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(Y_true_binary[:, class_idx], Y_pred_proba[:, class_idx])
            
            # Find optimal threshold (Youden's index: maximize TPR - FPR)
            youden_scores = tpr - fpr
            optimal_idx = np.argmax(youden_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_tpr = tpr[optimal_idx]
            optimal_fpr = fpr[optimal_idx]
            
            optimal_thresholds.append(optimal_threshold)
            
            print(f"Class_{class_idx}:")
            print(f"  Optimal threshold: {optimal_threshold:.4f}")
            print(f"  TPR at threshold: {optimal_tpr:.4f}")
            print(f"  FPR at threshold: {optimal_fpr:.4f}")
            print(f"  Youden index: {youden_scores[optimal_idx]:.4f}")
            
            # Calculate improvement over default threshold (0.5)
            default_pred = (Y_pred_proba[:, class_idx] > 0.5).astype(int)
            optimal_pred = (Y_pred_proba[:, class_idx] > optimal_threshold).astype(int)
            
            default_f1 = f1_score(Y_true_binary[:, class_idx], default_pred)
            optimal_f1 = f1_score(Y_true_binary[:, class_idx], optimal_pred)
            
            improvement = optimal_f1 - default_f1
            improvements.append(improvement)
            
            print(f"  F1 improvement: {improvement:+.4f}")
            print("-" * 30)
            
        except Exception as e:
            print(f"  Warning: Could not optimize threshold for Class_{class_idx}: {e}")
            optimal_thresholds.append(0.5)
            improvements.append(0.0)
    
    avg_improvement = np.mean(improvements)
    print(f"\nAverage F1 improvement: {avg_improvement:+.4f}")
    
    return optimal_thresholds, improvements

def apply_roc_optimized_predictions(Y_pred_proba, optimal_thresholds):
    """
    Apply ROC-optimized thresholds to get better predictions
    """
    Y_pred_optimized = np.zeros_like(Y_pred_proba)
    
    for class_idx, threshold in enumerate(optimal_thresholds):
        Y_pred_optimized[:, class_idx] = (Y_pred_proba[:, class_idx] > threshold).astype(float)
    
    # Convert to class indices (argmax)
    Y_pred_indices = np.argmax(Y_pred_optimized, axis=1)
    
    return Y_pred_indices, Y_pred_optimized

def create_metrics_plots(class_metrics, roc_data,model_name, data_source):
    """
    Create and save plots for F1 score, ROC, FPR, and FNR
    """
    dataset_name = extract_dataset_name(data_source)
    num_classes = len(class_metrics)
    class_labels = [f'Class_{i}' for i in range(num_classes)]
    
    # Extract metrics for plotting
    f1_scores = [metrics['f1'] for metrics in class_metrics]
    fprs = [metrics['fpr'] for metrics in class_metrics]
    fnrs = [metrics['fnr'] for metrics in class_metrics]
    
    # Create a 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Performance Analysis: {model_name}\n', fontsize=16, fontweight='bold')
    
    # Colors for bars
    bar_colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    
    # Plot F1 Score
    bars1 = ax1.bar(class_labels, f1_scores, color=bar_colors, alpha=0.7, edgecolor='black')
    ax1.set_title('F1 Score by Class', fontweight='bold')
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar, score in zip(bars1, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot ROC Curves (replacing per-class accuracy)
    if roc_data:
        # Define colors for ROC curves
        roc_colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive'])
        
        # Plot ROC curve for each class
        for class_name, color in zip([k for k in roc_data.keys() if k not in ['micro_avg', 'macro_avg']], roc_colors):
            data = roc_data[class_name]
            ax2.plot(data['fpr'], data['tpr'], color=color, lw=2,
                    label=f'{class_name} (AUC={data["auc"]:.3f})')
        
        # Plot micro and macro averages
        if 'micro_avg' in roc_data:
            ax2.plot(roc_data['micro_avg']['fpr'], roc_data['micro_avg']['tpr'],
                    label=f'Micro-avg (AUC={roc_data["micro_avg"]["auc"]:.3f})',
                    color='deeppink', linestyle=':', linewidth=3)
        
        if 'macro_avg' in roc_data:
            ax2.plot(roc_data['macro_avg']['fpr'], roc_data['macro_avg']['tpr'],
                    label=f'Macro-avg (AUC={roc_data["macro_avg"]["auc"]:.3f})',
                    color='navy', linestyle=':', linewidth=3)
        
        # Plot diagonal line
        ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontsize=11)
        ax2.set_ylabel('True Positive Rate', fontsize=11)
        ax2.set_title('ROC Curves', fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'ROC Data Not Available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12, style='italic')
        ax2.set_title('ROC Curves', fontweight='bold')
    
    # Plot FPR
    bars3 = ax3.bar(class_labels, fprs, color=bar_colors, alpha=0.7, edgecolor='black')
    ax3.set_title('False Positive Rate by Class', fontweight='bold')
    ax3.set_ylabel('FPR')
    ax3.set_ylim(0, max(max(fprs), 0.1) + 0.05)
    ax3.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar, fpr in zip(bars3, fprs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{fpr:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot FNR
    bars4 = ax4.bar(class_labels, fnrs, color=bar_colors, alpha=0.7, edgecolor='black')
    ax4.set_title('False Negative Rate by Class', fontweight='bold')
    ax4.set_ylabel('FNR')
    ax4.set_ylim(0, max(max(fnrs), 0.1) + 0.05)
    ax4.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar, fnr in zip(bars4, fnrs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{fnr:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Rotate x-axis labels if there are many classes
    if num_classes > 5:
        for ax in [ax1, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"metrics_plot_{model_name}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nMetrics plot saved as: {filename}")
    plt.close()

def report_results(Y_true, Y_pred, packets, model_name, data_source, prediction_time, writer, Y_pred_proba=None):
    # Convert from categorical to class indices if needed
    if len(Y_true.shape) > 1:
        Y_true_indices = np.argmax(Y_true, axis=1)
        Y_true_categorical = Y_true
    else:
        Y_true_indices = Y_true
        Y_true_categorical = None
        
    if len(Y_pred.shape) > 1:
        Y_pred_indices = np.argmax(Y_pred, axis=1)
    else:
        Y_pred_indices = Y_pred
    
    accuracy = accuracy_score(Y_true_indices, Y_pred_indices)
    f1 = f1_score(Y_true_indices, Y_pred_indices, average='weighted')
    
    # Compute confusion matrix
    cm = confusion_matrix(Y_true_indices, Y_pred_indices)
    
    # Calculate metrics for each class
    num_classes = len(np.unique(Y_true_indices))
    attack_percentage = np.sum(Y_true_indices != 0) / len(Y_true_indices) * 100
    
    row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets, 
           'Samples': len(Y_true_indices), 'Attack%': '{:04.3f}'.format(attack_percentage),
           'Accuracy': '{:05.4f}'.format(accuracy), 'F1Score': '{:05.4f}'.format(f1), 
           'Source': data_source}
    
    pprint.pprint(row, sort_dicts=False)
    writer.writerow(row)
    
    # Calculate TNR, FNR, TPR, FPR for each class
    print(f"\nDetailed Metrics for {model_name}:")
    print("="*60)
    
    # Store metrics for plotting
    class_metrics = []
    
    for class_idx in range(num_classes):
        # For each class, calculate TP, TN, FP, FN
        TP = cm[class_idx, class_idx]  # True Positives
        FP = np.sum(cm[:, class_idx]) - TP  # False Positives
        FN = np.sum(cm[class_idx, :]) - TP  # False Negatives
        TN = np.sum(cm) - TP - FP - FN  # True Negatives
        
        # Calculate rates
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Sensitivity/Recall)
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate (Specificity)
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # False Negative Rate
        
        # Calculate per-class accuracy and F1 score
        class_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        class_f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
        
        # Store metrics for plotting
        class_metrics.append({
            'accuracy': class_accuracy,
            'f1': class_f1,
            'fpr': FPR,
            'fnr': FNR,
            'tpr': TPR,
            'tnr': TNR
        })
        
        print(f"Class_{class_idx}:")
        print(f"  TPR: {TPR:.4f}")
        print(f"  TNR: {TNR:.4f}")
        print(f"  FPR: {FPR:.4f}")
        print(f"  FNR: {FNR:.4f}")
        print(f"  Support: {TP + FN}")
        print("-" * 30)
    
    # 🎯 ROC ANALYSIS AND UNIFIED VISUALIZATION
    roc_data = None
    if Y_pred_proba is not None and Y_true is not None:
        print(f"\n🎯 PERFORMING ROC ANALYSIS for {model_name}...")
        
        # Calculate ROC curves and AUC scores
        try:
            roc_data = calculate_multiclass_roc(Y_true_indices, Y_pred_proba, num_classes)
            
            # Analyze performance and provide recommendations
            analysis = analyze_roc_performance(roc_data)
            
            # Add ROC metrics to output
            micro_auc = roc_data.get('micro_avg', {}).get('auc', 0)
            macro_auc = roc_data.get('macro_avg', {}).get('auc', 0)
            
            print(f"\n📊 ROC SUMMARY:")
            print(f"   Micro-average AUC: {micro_auc:.4f}")
            print(f"   Macro-average AUC: {macro_auc:.4f}")
            
        except Exception as e:
            print(f"⚠️  ROC analysis failed: {e}")
            roc_data = None
    else:
        print("⚠️  Skipping ROC analysis (missing probability predictions or labels)")
    
    # Create and save metrics plot (includes F1, ROC curves, FPR, FNR)
    create_metrics_plots(class_metrics, roc_data, model_name, data_source)

if __name__ == "__main__":
    main(sys.argv[1:])