#!/usr/bin/env python3
"""
LightGBM Algorithm Implementation
=================================
A comprehensive implementation of LightGBM for various ML tasks including:
1. Binary Classification
2. Multiclass Classification  
3. Regression
4. Early Stopping
5. Feature Importance Analysis
6. Hyperparameter Tuning
7. Cross-Validation
8. Advanced Features (Dart, GOSS)

LightGBM advantages:
- Extremely fast training speed
- Lower memory usage
- Better accuracy than XGBoost
- Supports parallel and GPU learning
- Built-in feature selection
- Handles large datasets efficiently
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           mean_squared_error, r2_score, roc_auc_score, log_loss)
from sklearn.datasets import load_breast_cancer, load_wine, make_regression, make_classification
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Install LightGBM: pip install lightgbm
import lightgbm as lgb

class LightGBMSuite:
    """
    Complete LightGBM Machine Learning Suite
    Supports classification and regression with advanced features
    """
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.feature_importance = None
        
    def binary_classification_example(self):
        """
        Binary Classification using Breast Cancer Dataset
        """
        print("=" * 70)
        print("LIGHTGBM BINARY CLASSIFICATION EXAMPLE")
        print("=" * 70)
        
        # Load dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Parameters for binary classification
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': 42
        }
        
        print("Training LightGBM Binary Classifier...")
        
        # Train model with early stopping
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)  # Silent training
            ]
        )
        
        self.model_type = 'binary_classification'
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        
        print(f"\nResults:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print(f"Best iteration: {self.model.best_iteration}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=data.target_names))
        
        # Feature importance
        self.feature_importance = self.model.feature_importance(importance_type='gain')
        self.plot_feature_importance(feature_names, self.feature_importance)
        
        return accuracy
    
    def multiclass_classification_example(self):
        """
        Multiclass Classification using Wine Dataset
        """
        print("\n" + "=" * 70)
        print("LIGHTGBM MULTICLASS CLASSIFICATION EXAMPLE")
        print("=" * 70)
        
        # Load dataset
        data = load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Target distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Parameters for multiclass classification
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': 42
        }
        
        print("Training LightGBM Multiclass Classifier...")
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Best iteration: {self.model.best_iteration}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=data.target_names))
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred, data.target_names)
        
        return accuracy
    
    def regression_example(self):
        """
        Regression Example
        """
        print("\n" + "=" * 70)
        print("LIGHTGBM REGRESSION EXAMPLE")
        print("=" * 70)
        
        # Create regression dataset
        X, y = make_regression(n_samples=2000, n_features=15, noise=0.1, random_state=42)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Parameters for regression
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': 42
        }
        
        print("Training LightGBM Regressor...")
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        
        self.model_type = 'regression'
        
        # Make predictions
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nRegression Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Best iteration: {self.model.best_iteration}")
        
        # Plot results
        self.plot_regression_results(y_test, y_pred)
        
        return r2
    
    def advanced_features_example(self):
        """
        Advanced LightGBM features: DART, GOSS, etc.
        """
        print("\n" + "=" * 70)
        print("LIGHTGBM ADVANCED FEATURES EXAMPLE")
        print("=" * 70)
        
        # Load dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Different boosting types to compare
        boosting_types = {
            'gbdt': 'Gradient Boosting Decision Tree',
            'dart': 'Dropouts meet Multiple Additive Regression Trees',
            'goss': 'Gradient-based One-Side Sampling'
        }
        
        results = {}
        
        for boost_type, description in boosting_types.items():
            print(f"\nTesting {boost_type.upper()}: {description}")
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            # Parameters
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': boost_type,
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'verbose': 0,
                'random_state': 42
            }
            
            # Additional params for DART
            if boost_type == 'dart':
                params.update({
                    'drop_rate': 0.1,
                    'max_drop': 50,
                    'skip_drop': 0.5,
                    'uniform_drop': False
                })
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=300,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # Predict and evaluate
            y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
            y_pred = (y_pred_proba > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[boost_type] = {
                'accuracy': accuracy,
                'auc': auc,
                'best_iteration': model.best_iteration
            }
            
            print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Best Iteration: {model.best_iteration}")
        
        # Compare results
        print(f"\n{'='*50}")
        print("COMPARISON OF BOOSTING TYPES")
        print(f"{'='*50}")
        for boost_type, metrics in results.items():
            print(f"{boost_type.upper()}: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
        
        return results
    
    def hyperparameter_tuning_example(self):
        """
        Hyperparameter tuning with LightGBM
        """
        print("\n" + "=" * 70)
        print("LIGHTGBM HYPERPARAMETER TUNING")
        print("=" * 70)
        
        # Use breast cancer dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define parameter search space
        param_grid = {
            'num_leaves': [31, 50, 70],
            'learning_rate': [0.01, 0.1, 0.2],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.8, 0.9, 1.0],
            'min_data_in_leaf': [20, 50, 100]
        }
        
        print("Performing manual hyperparameter search...")
        
        best_score = 0
        best_params = None
        
        # Manual grid search (simplified for demo)
        test_combinations = [
            {'num_leaves': 31, 'learning_rate': 0.1, 'feature_fraction': 0.9, 
             'bagging_fraction': 0.8, 'min_data_in_leaf': 20},
            {'num_leaves': 50, 'learning_rate': 0.05, 'feature_fraction': 0.9, 
             'bagging_fraction': 0.9, 'min_data_in_leaf': 50},
            {'num_leaves': 70, 'learning_rate': 0.1, 'feature_fraction': 0.8, 
             'bagging_fraction': 0.8, 'min_data_in_leaf': 20}
        ]
        
        for i, params in enumerate(test_combinations):
            print(f"Testing combination {i+1}/3...")
            
            # Add fixed parameters
            full_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'verbose': 0,
                'random_state': 42,
                **params
            }
            
            # Cross-validation
            train_data = lgb.Dataset(X_train, label=y_train)
            cv_results = lgb.cv(
                full_params,
                train_data,
                num_boost_round=500,
                nfold=5,
                stratified=True,
                shuffle=True,
                seed=42,
                return_cvbooster=True,
                callbacks=[lgb.log_evaluation(period=0)]
            )
            
            # Get best score
            best_cv_score = max(cv_results['valid binary_logloss-mean'])
            
            if best_cv_score > best_score:
                best_score = best_cv_score
                best_params = params
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best CV score: {best_score:.4f}")
        
        # Train final model with best parameters
        final_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': 0,
            'random_state': 42,
            **best_params
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        final_model = lgb.train(
            final_params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Test final model
        y_pred_proba = final_model.predict(X_test, num_iteration=final_model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Final model test accuracy: {test_accuracy:.4f}")
        
        return test_accuracy
    
    def feature_selection_example(self):
        """
        Feature selection with LightGBM
        """
        print("\n" + "=" * 70)
        print("LIGHTGBM FEATURE SELECTION")
        print("=" * 70)
        
        # Create dataset with noise features
        X_base, y = load_breast_cancer(return_X_y=True)
        
        # Add noise features
        np.random.seed(42)
        noise_features = np.random.randn(X_base.shape[0], 50)
        X = np.hstack([X_base, noise_features])
        
        feature_names = (list(load_breast_cancer().feature_names) + 
                        [f'noise_{i}' for i in range(50)])
        
        print(f"Dataset shape with noise features: {X.shape}")
        print(f"Original features: {X_base.shape[1]}, Noise features: {noise_features.shape[1]}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model for feature importance
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'verbose': 0,
            'random_state': 42
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=300,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Get feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Select top features
        top_features = feature_importance_df.head(20)['feature'].tolist()
        print(f"Selected top {len(top_features)} features")
        
        # Check how many original vs noise features were selected
        original_selected = sum(1 for f in top_features if not f.startswith('noise_'))
        noise_selected = len(top_features) - original_selected
        
        print(f"Original features selected: {original_selected}")
        print(f"Noise features selected: {noise_selected}")
        
        # Train model with selected features
        feature_indices = [feature_names.index(f) for f in top_features]
        X_train_selected = X_train[:, feature_indices]
        X_test_selected = X_test[:, feature_indices]
        
        train_data_selected = lgb.Dataset(X_train_selected, label=y_train)
        valid_data_selected = lgb.Dataset(X_test_selected, label=y_test, reference=train_data_selected)
        
        model_selected = lgb.train(
            params,
            train_data_selected,
            valid_sets=[valid_data_selected],
            num_boost_round=300,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Compare performance
        y_pred_all = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred_selected = model_selected.predict(X_test_selected, num_iteration=model_selected.best_iteration)
        
        acc_all = accuracy_score(y_test, (y_pred_all > 0.5).astype(int))
        acc_selected = accuracy_score(y_test, (y_pred_selected > 0.5).astype(int))
        
        print(f"\nAccuracy with all features: {acc_all:.4f}")
        print(f"Accuracy with selected features: {acc_selected:.4f}")
        print(f"Feature reduction: {X.shape[1]} -> {len(top_features)} ({(1-len(top_features)/X.shape[1])*100:.1f}% reduction)")
        
        return acc_selected
    
    def plot_feature_importance(self, feature_names, importance_values, top_n=15):
        """Plot feature importance"""
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - LightGBM')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - LightGBM')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_regression_results(self, y_true, y_pred):
        """Plot regression results"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.show()
    
    def learning_curve_example(self):
        """
        Plot learning curves to analyze training progress
        """
        print("\n" + "=" * 70)
        print("LIGHTGBM LEARNING CURVES")
        print("=" * 70)
        
        # Load dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Parameters
        params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': 0,
            'random_state': 42
        }
        
        # Train with evaluation history
        evals_result = {}
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'eval'],
            num_boost_round=500,
            evals_result=evals_result,
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        # Plot learning curves
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(evals_result['train']['binary_logloss'], label='Train Loss')
        plt.plot(evals_result['eval']['binary_logloss'], label='Validation Loss')
        plt.xlabel('Boosting Round')
        plt.ylabel('Log Loss')
        plt.title('Learning Curve - Loss')
        plt.legend()
        
        # Plot AUC
        plt.subplot(1, 2, 2)
        plt.plot(evals_result['train']['auc'], label='Train AUC')
        plt.plot(evals_result['eval']['auc'], label='Validation AUC')
        plt.xlabel('Boosting Round')
        plt.ylabel('AUC')
        plt.title('Learning Curve - AUC')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return model

def main():
    """
    Main function to run all LightGBM examples
    """
    print("LIGHTGBM MACHINE LEARNING ALGORITHM SUITE")
    print("==========================================")
    print("LightGBM is Microsoft's gradient boosting framework that:")
    print("- Provides extremely fast training speed")
    print("- Uses lower memory compared to XGBoost")
    print("- Achieves better accuracy in many cases")
    print("- Supports parallel and GPU learning")
    print("- Handles large datasets efficiently")
    print("- Built-in feature selection capabilities")
    
    # Initialize the suite
    lgb_suite = LightGBMSuite()
    
    # Run all examples
    try:
        results = {}
        
        # Binary Classification
        print("\n🔸 Running Binary Classification Example...")
        results['binary_acc'] = lgb_suite.binary_classification_example()
        
        # Multiclass Classification
        print("\n🔸 Running Multiclass Classification Example...")
        results['multi_acc'] = lgb_suite.multiclass_classification_example()
        
        # Regression
        print("\n🔸 Running Regression Example...")
        results['reg_r2'] = lgb_suite.regression_example()
        
        # Advanced Features
        print("\n🔸 Running Advanced Features Example...")
        results['advanced'] = lgb_suite.advanced_features_example()
        
        # Hyperparameter Tuning
        print("\n🔸 Running Hyperparameter Tuning Example...")
        results['tuned_acc'] = lgb_suite.hyperparameter_tuning_example()
        
        # Feature Selection
        print("\n🔸 Running Feature Selection Example...")
        results['feature_sel_acc'] = lgb_suite.feature_selection_example()
        
        # Learning Curves
        print("\n🔸 Running Learning Curves Example...")
        lgb_suite.learning_curve_example()
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY OF RESULTS")
        print("=" * 70)
        print(f"Binary Classification Accuracy: {results['binary_acc']:.4f}")
        print(f"Multiclass Classification Accuracy: {results['multi_acc']:.4f}")
        print(f"Regression R² Score: {results['reg_r2']:.4f}")
        print(f"Hyperparameter Tuned Accuracy: {results['tuned_acc']:.4f}")
        print(f"Feature Selection Accuracy: {results['feature_sel_acc']:.4f}")
        
        print(f"\nAdvanced Features Comparison:")
        for boost_type, metrics in results['advanced'].items():
            print(f"  {boost_type.upper()}: {metrics['accuracy']:.4f} (AUC: {metrics['auc']:.4f})")
        
        print(f"\n✅ All LightGBM examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        print("Make sure to install required packages:")
        print("pip install lightgbm scikit-learn matplotlib seaborn pandas numpy")

if __name__ == "__main__":
    main()
