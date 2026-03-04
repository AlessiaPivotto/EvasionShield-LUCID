#!/usr/bin/env python3

# Copyright (c) 2025 @ EvasionShield-LUCID
# Author: AI Assistant
# Project: Random Forest Binary Classification for DDoS Attack Detection
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

# Sample commands
# Training: python3 random_forest_binary.py --train ../DATASETS/merged_flatten_train_dataset.hdf5 --val ../DATASETS/merged_flatten_val_dataset.hdf5
# Training with hyperparameter tuning: python3 random_forest_binary.py --train ../DATASETS/merged_flatten_train_dataset.hdf5 --val ../DATASETS/merged_flatten_val_dataset.hdf5 --tune
# Testing: python3 random_forest_binary.py --predict ../DATASETS/merged_flatten_test_dataset.hdf5 --model ./random_forest_model.pkl
# Cross-validation: python3 random_forest_binary.py --train ../DATASETS/merged_flatten_train_dataset.hdf5 --cv 5

import numpy as np
import h5py
import pickle
import argparse
import glob
import time
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils import shuffle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import util_functions
sys.path.append('../flatten_lucid')
try:
    from util_functions import SEED
except ImportError:
    # Define SEED locally if import fails
    SEED = 1

# Set random seed for reproducibility
np.random.seed(SEED)

class RandomForestBinaryClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', random_state=SEED):
        """
        Initialize Random Forest Binary Classifier
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at a leaf node
            max_features: Number of features to consider when looking for the best split
            random_state: Random state for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        self.is_trained = False
    
    def load_dataset(self, path):
        """
        Load dataset from HDF5 file (compatible with Flatten LUCID format)
        
        Args:
            path: Path to HDF5 file or glob pattern
            
        Returns:
            X: Feature matrix
            y: Label vector
        """
        if isinstance(path, str):
            files = glob.glob(path)
            if not files:
                raise FileNotFoundError(f"No files found matching pattern: {path}")
            filename = files[0]
        else:
            filename = path
            
        print(f"Loading dataset from: {filename}")
        
        with h5py.File(filename, "r") as dataset:
            X = np.array(dataset["set_x"][:])  # features
            y = np.array(dataset["set_y"][:])  # labels
            
        # Flatten the input if it's not already flattened (same as Flatten LUCID)
        if len(X.shape) > 2:
            print(f"Original shape: {X.shape}")
            X = X.reshape(X.shape[0], -1)
            print(f"Flattened shape: {X.shape}")
        
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Label distribution: {np.bincount(y.astype(int))}")
        
        return X, y
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print("Training Random Forest model...")
        start_time = time.time()
        
        # Shuffle the training data
        X_train, y_train = shuffle(X_train, y_train, random_state=SEED)
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, average='binary')
            val_recall = recall_score(y_val, val_pred, average='binary')
            val_f1 = f1_score(y_val, val_pred, average='binary')
            
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation Precision: {val_precision:.4f}")
            print(f"Validation Recall: {val_recall:.4f}")
            print(f"Validation F1-Score: {val_f1:.4f}")
            
            # Print confusion matrix
            cm = confusion_matrix(y_val, val_pred)
            print(f"Validation Confusion Matrix:\n{cm}")
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            predictions: Predicted labels
            probabilities: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            metrics: Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        print("Evaluating model on test set...")
        start_time = time.time()
        
        predictions, probabilities = self.predict(X_test)
        
        evaluation_time = time.time() - start_time
        print(f"Evaluation completed in {evaluation_time:.4f} seconds")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='binary')
        recall = recall_score(y_test, predictions, average='binary')
        f1 = f1_score(y_test, predictions, average='binary')
        cm = confusion_matrix(y_test, predictions)
        
        # Calculate False Negative Rate (FNR)
        # FNR = FN / (FN + TP) = 1 - Recall
        fnr = 1 - recall
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fnr': fnr,
            'confusion_matrix': cm,
            'evaluation_time': evaluation_time
        }
        
        # Print results - Focus on F1 Score, FNR, and Accuracy
        print("=== TEST RESULTS ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"False Negative Rate (FNR): {fnr:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Test Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{classification_report(y_test, predictions)}")
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            cv_scores: Cross-validation scores
        """
        print(f"Performing {cv}-fold cross-validation...")
        
        # Shuffle the data
        X, y = shuffle(X, y, random_state=SEED)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def tune_hyperparameters(self, X_train, y_train, method='random', n_iter=20):
        """
        Tune hyperparameters using GridSearchCV or RandomizedSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
            n_iter: Number of iterations for RandomizedSearchCV
            
        Returns:
            best_params: Best hyperparameters found
        """
        print(f"Tuning hyperparameters using {method} search...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        if method == 'grid':
            search = GridSearchCV(
                RandomForestClassifier(random_state=SEED, n_jobs=-1),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                RandomForestClassifier(random_state=SEED, n_jobs=-1),
                param_grid,
                n_iter=n_iter,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=SEED
            )
        
        # Perform search
        search.fit(X_train, y_train)
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best cross-validation score: {search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = search.best_estimator_
        self.is_trained = True
        
        return search.best_params_
    
    def save_model(self, filepath):
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from: {filepath}")
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importances from the trained model
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            feature_importance: DataFrame with feature importances
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importances")
            
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance


def main():
    parser = argparse.ArgumentParser(
        description='Random Forest Binary Classifier for DDoS Detection (Compatible with Flatten LUCID)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--train', type=str,
                        help='Path to training dataset (HDF5 file)')
    parser.add_argument('-v', '--val', type=str,
                        help='Path to validation dataset (HDF5 file)')
    parser.add_argument('-p', '--predict', type=str,
                        help='Path to test dataset for prediction (HDF5 file)')
    parser.add_argument('-m', '--model', type=str,
                        help='Path to save/load model file (.pkl)')
    parser.add_argument('-cv', '--cross_validation', type=int, default=0,
                        help='Number of cross-validation folds (0 to disable)')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--tune_method', choices=['grid', 'random'], default='random',
                        help='Hyperparameter tuning method')
    parser.add_argument('--n_iter', type=int, default=20,
                        help='Number of iterations for random search')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='Maximum depth of trees')
    parser.add_argument('--feature_importance', action='store_true',
                        help='Show feature importance after training')

    args = parser.parse_args()

    # Initialize classifier
    rf_classifier = RandomForestBinaryClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )

    # Training mode
    if args.train:
        print("=== RANDOM FOREST BINARY CLASSIFICATION ===")
        print("Loading training data...")
        
        X_train, y_train = rf_classifier.load_dataset(args.train)
        
        # Load validation data if provided
        X_val, y_val = None, None
        if args.val:
            print("Loading validation data...")
            X_val, y_val = rf_classifier.load_dataset(args.val)
        
        # Hyperparameter tuning
        if args.tune:
            best_params = rf_classifier.tune_hyperparameters(
                X_train, y_train, 
                method=args.tune_method, 
                n_iter=args.n_iter
            )
            print(f"Best hyperparameters: {best_params}")
        else:
            # Train with default or specified parameters
            rf_classifier.train(X_train, y_train, X_val, y_val)
        
        # Cross-validation
        if args.cross_validation > 0:
            rf_classifier.cross_validate(X_train, y_train, cv=args.cross_validation)
        
        # Save model
        model_path = args.model if args.model else f"random_forest_binary_{int(time.time())}.pkl"
        rf_classifier.save_model(model_path)
        
        # Show feature importance if requested
        if args.feature_importance:
            print("\nTop 20 Feature Importances:")
            importance_df = rf_classifier.get_feature_importance()
            print(importance_df.head(20))

    # Prediction mode
    elif args.predict:
        if not args.model:
            print("Error: Model file path required for prediction mode")
            sys.exit(1)
        
        print("=== RANDOM FOREST PREDICTION ===")
        print("Loading model...")
        rf_classifier.load_model(args.model)
        
        print("Loading test data...")
        X_test, y_test = rf_classifier.load_dataset(args.predict)
        
        # Evaluate the model
        metrics = rf_classifier.evaluate(X_test, y_test)
        
        # Save predictions
        predictions, probabilities = rf_classifier.predict(X_test)
        timestamp = int(time.time())
        pred_filename = f"rf_predictions_{timestamp}.csv"
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': predictions,
            'probability_class_0': probabilities[:, 0],
            'probability_class_1': probabilities[:, 1]
        })
        
        pred_df.to_csv(pred_filename, index=False)
        print(f"Predictions saved to: {pred_filename}")

    else:
        print("Error: Please specify either --train for training or --predict for prediction")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
