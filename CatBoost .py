#!/usr/bin/env python3
"""
CatBoost Algorithm Implementation
================================
A comprehensive implementation of CatBoost for various ML tasks including:
1. Binary Classification
2. Multiclass Classification  
3. Regression
4. Feature Importance Analysis
5. Hyperparameter Tuning

CatBoost advantages:
- Handles categorical features automatically (no preprocessing needed)
- Built-in cross-validation
- GPU support
- Robust to overfitting
- Great performance out-of-the-box
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.datasets import load_breast_cancer, load_wine, load_boston, make_classification
import warnings
warnings.filterwarnings('ignore')

# Install CatBoost: pip install catboost
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

class CatBoostMLSuite:
    """
    Complete CatBoost Machine Learning Suite
    Supports classification and regression tasks
    """
    
    def __init__(self):
        self.model = None
        self.model_type = None
        
    def binary_classification_example(self):
        """
        Binary Classification using Breast Cancer Dataset
        """
        print("=" * 60)
        print("CATBOOST BINARY CLASSIFICATION EXAMPLE")
        print("=" * 60)
        
        # Load dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        
        # Create DataFrame for better handling
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        print(f"Dataset shape: {df.shape}")
        print(f"Target distribution:\n{pd.Series(y).value_counts()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize CatBoost Classifier
        self.model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.1,
            loss_function='Logloss',
            random_seed=42,
            verbose=False  # Set to True to see training progress
        )
        
        self.model_type = 'classification'
        
        # Train the model
        print("\nTraining CatBoost Binary Classifier...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=data.target_names))
        
        # Feature importance
        self.plot_feature_importance(feature_names[:10])  # Top 10 features
        
        return accuracy
    
    def multiclass_classification_example(self):
        """
        Multiclass Classification using Wine Dataset
        """
        print("\n" + "=" * 60)
        print("CATBOOST MULTICLASS CLASSIFICATION EXAMPLE")
        print("=" * 60)
        
        # Load dataset
        data = load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Target distribution:\n{pd.Series(y).value_counts()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize CatBoost Classifier for multiclass
        self.model = CatBoostClassifier(
            iterations=500,
            depth=4,
            learning_rate=0.1,
            loss_function='MultiClass',
            random_seed=42,
            verbose=False
        )
        
        # Train the model
        print("\nTraining CatBoost Multiclass Classifier...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=data.target_names))
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred, data.target_names)
        
        return accuracy
    
    def regression_example(self):
        """
        Regression using Boston Housing Dataset
        """
        print("\n" + "=" * 60)
        print("CATBOOST REGRESSION EXAMPLE")
        print("=" * 60)
        
        # Load dataset (Note: Boston dataset is deprecated, using make_regression for demo)
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=1000, n_features=13, noise=0.1, random_state=42)
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize CatBoost Regressor
        self.model = CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.1,
            loss_function='RMSE',
            random_seed=42,
            verbose=False
        )
        
        self.model_type = 'regression'
        
        # Train the model
        print("\nTraining CatBoost Regressor...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nRegression Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Plot predictions vs actual
        self.plot_regression_results(y_test, y_pred)
        
        return r2
    
    def categorical_features_example(self):
        """
        Example with categorical features (CatBoost's main strength)
        """
        print("\n" + "=" * 60)
        print("CATBOOST WITH CATEGORICAL FEATURES")
        print("=" * 60)
        
        # Create synthetic dataset with categorical features
        np.random.seed(42)
        n_samples = 1000
        
        # Numerical features
        num_features = np.random.randn(n_samples, 3)
        
        # Categorical features
        categories_1 = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
        categories_2 = np.random.choice(['High', 'Medium', 'Low'], n_samples)
        categories_3 = np.random.choice(['Type1', 'Type2', 'Type3', 'Type4', 'Type5'], n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'num_feat_1': num_features[:, 0],
            'num_feat_2': num_features[:, 1],
            'num_feat_3': num_features[:, 2],
            'cat_feat_1': categories_1,
            'cat_feat_2': categories_2,
            'cat_feat_3': categories_3
        })
        
        # Create target based on features (with some relationship)
        target = (
            num_features[:, 0] * 0.5 + 
            num_features[:, 1] * 0.3 + 
            (categories_1 == 'A').astype(int) * 0.8 +
            (categories_2 == 'High').astype(int) * 0.6 +
            np.random.randn(n_samples) * 0.1
        )
        target = (target > np.median(target)).astype(int)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Categorical features: {df.select_dtypes(include=['object']).columns.tolist()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df, target, test_size=0.2, random_state=42, stratify=target
        )
        
        # Identify categorical features
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        categorical_indices = [df.columns.get_loc(col) for col in categorical_features]
        
        # Initialize CatBoost with categorical features specified
        self.model = CatBoostClassifier(
            iterations=500,
            depth=4,
            learning_rate=0.1,
            cat_features=categorical_indices,  # This is the key!
            random_seed=42,
            verbose=False
        )
        
        print(f"\nCategorical feature indices: {categorical_indices}")
        print("Training CatBoost with categorical features...")
        
        # Train (CatBoost handles categorical encoding automatically!)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy with categorical features: {accuracy:.4f}")
        
        # Feature importance
        feature_importance = self.model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': df.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance:")
        print(importance_df)
        
        return accuracy
    
    def hyperparameter_tuning_example(self):
        """
        Hyperparameter tuning with CatBoost
        """
        print("\n" + "=" * 60)
        print("CATBOOST HYPERPARAMETER TUNING")
        print("=" * 60)
        
        # Use breast cancer dataset for quick demo
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define parameter grid
        param_grid = {
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'iterations': [100, 500, 1000]
        }
        
        # Initialize base model
        base_model = CatBoostClassifier(
            random_seed=42,
            verbose=False
        )
        
        print("Performing Grid Search...")
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Test best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test accuracy with best parameters: {test_accuracy:.4f}")
        
        return test_accuracy
    
    def plot_feature_importance(self, feature_names, top_n=10):
        """Plot feature importance"""
        if self.model is None:
            print("No model trained yet!")
            return
            
        feature_importance = self.model.get_feature_importance()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(feature_importance)],
            'importance': feature_importance
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - CatBoost')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - CatBoost')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_regression_results(self, y_true, y_pred):
        """Plot regression results"""
        plt.figure(figsize=(10, 6))
        
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
    
    def cross_validation_example(self):
        """
        Cross-validation with CatBoost
        """
        print("\n" + "=" * 60)
        print("CATBOOST CROSS-VALIDATION")
        print("=" * 60)
        
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        # Initialize model
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
        
        # Perform cross-validation
        print("Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores.mean()

def main():
    """
    Main function to run all CatBoost examples
    """
    print("CATBOOST MACHINE LEARNING ALGORITHM SUITE")
    print("=========================================")
    print("CatBoost is a gradient boosting framework that:")
    print("- Handles categorical features automatically")
    print("- Provides great performance out-of-the-box")
    print("- Has built-in overfitting protection")
    print("- Supports GPU training")
    print("- Offers excellent interpretability")
    
    # Initialize the suite
    catboost_suite = CatBoostMLSuite()
    
    # Run all examples
    try:
        # Binary Classification
        binary_acc = catboost_suite.binary_classification_example()
        
        # Multiclass Classification
        multi_acc = catboost_suite.multiclass_classification_example()
        
        # Regression
        reg_r2 = catboost_suite.regression_example()
        
        # Categorical Features
        cat_acc = catboost_suite.categorical_features_example()
        
        # Cross-validation
        cv_score = catboost_suite.cross_validation_example()
        
        # Hyperparameter Tuning
        tuned_acc = catboost_suite.hyperparameter_tuning_example()
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY OF RESULTS")
        print("=" * 60)
        print(f"Binary Classification Accuracy: {binary_acc:.4f}")
        print(f"Multiclass Classification Accuracy: {multi_acc:.4f}")
        print(f"Regression R² Score: {reg_r2:.4f}")
        print(f"Categorical Features Accuracy: {cat_acc:.4f}")
        print(f"Cross-validation Score: {cv_score:.4f}")
        print(f"Tuned Model Accuracy: {tuned_acc:.4f}")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure to install required packages:")
        print("pip install catboost scikit-learn matplotlib seaborn pandas numpy")

if __name__ == "__main__":
    main()
