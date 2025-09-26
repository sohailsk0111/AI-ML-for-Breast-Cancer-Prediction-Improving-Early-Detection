#!/usr/bin/env python3
"""
Breast Cancer Prediction using Machine Learning
Author: Manus AI
Date: September 2025

This script implements a machine learning system to predict breast cancer
from Fine Needle Aspiration (FNA) cytological features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_curve, auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BreastCancerPredictor:
    """
    A comprehensive breast cancer prediction system using machine learning.
    """
    
    def __init__(self):
        """Initialize the predictor with default parameters."""
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the Wisconsin Breast Cancer dataset."""
        print("Loading Wisconsin Breast Cancer Dataset...")
        
        # Load dataset from sklearn
        self.data = load_breast_cancer()
        self.X = pd.DataFrame(self.data.data, columns=self.data.feature_names)
        self.y = pd.Series(self.data.target, name='diagnosis')
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.X.shape}")
        print(f"Features: {len(self.data.feature_names)}")
        print(f"Classes: {self.data.target_names}")
        
        return self.X, self.y
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic information
        print("\nDataset Info:")
        print(f"Number of samples: {len(self.X)}")
        print(f"Number of features: {len(self.X.columns)}")
        print(f"Missing values: {self.X.isnull().sum().sum()}")
        
        # Target distribution
        print(f"\nTarget Distribution:")
        target_counts = pd.Series(self.y).value_counts()
        print(f"Malignant (1): {target_counts[1]} ({target_counts[1]/len(self.y)*100:.1f}%)")
        print(f"Benign (0): {target_counts[0]} ({target_counts[0]/len(self.y)*100:.1f}%)")
        
        # Feature statistics
        print(f"\nFeature Statistics:")
        print(self.X.describe())
        
        return self.X.describe(), target_counts
    
    def visualize_data(self):
        """Create visualizations for data exploration."""
        print("\nCreating data visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Target distribution
        plt.subplot(2, 3, 1)
        target_counts = pd.Series(self.y).value_counts()
        labels = ['Benign', 'Malignant']
        colors = ['lightblue', 'lightcoral']
        plt.pie(target_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Diagnosis', fontsize=14, fontweight='bold')
        
        # 2. Correlation heatmap (first 10 features for readability)
        plt.subplot(2, 3, 2)
        corr_matrix = self.X.iloc[:, :10].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix (First 10 Features)', fontsize=14, fontweight='bold')
        
        # 3. Box plot for mean features
        plt.subplot(2, 3, 3)
        mean_features = [col for col in self.X.columns if 'mean' in col][:5]
        data_for_box = []
        labels_for_box = []
        for feature in mean_features:
            data_for_box.append(self.X[feature])
            labels_for_box.append(feature.replace('mean ', '').title())
        
        plt.boxplot(data_for_box, labels=labels_for_box)
        plt.title('Distribution of Mean Features', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 4. Feature comparison by diagnosis
        plt.subplot(2, 3, 4)
        feature_to_plot = 'mean radius'
        benign_data = self.X[self.y == 0][feature_to_plot]
        malignant_data = self.X[self.y == 1][feature_to_plot]
        
        plt.hist(benign_data, alpha=0.7, label='Benign', color='lightblue', bins=20)
        plt.hist(malignant_data, alpha=0.7, label='Malignant', color='lightcoral', bins=20)
        plt.xlabel(feature_to_plot.title())
        plt.ylabel('Frequency')
        plt.title(f'{feature_to_plot.title()} Distribution by Diagnosis', fontsize=14, fontweight='bold')
        plt.legend()
        
        # 5. Scatter plot
        plt.subplot(2, 3, 5)
        colors = ['blue' if x == 0 else 'red' for x in self.y]
        plt.scatter(self.X['mean radius'], self.X['mean texture'], c=colors, alpha=0.6)
        plt.xlabel('Mean Radius')
        plt.ylabel('Mean Texture')
        plt.title('Mean Radius vs Mean Texture', fontsize=14, fontweight='bold')
        
        # 6. Feature importance preview
        plt.subplot(2, 3, 6)
        # Quick random forest to get feature importance
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_temp.fit(self.X, self.y)
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/data_exploration.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Data exploration plots saved as 'data_exploration.png'")
        
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess the data for machine learning."""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled using StandardScaler")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def initialize_models(self):
        """Initialize various machine learning models."""
        print("\n" + "="*50)
        print("MODEL INITIALIZATION")
        print("="*50)
        
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Support Vector Machine': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def train_models(self):
        """Train all models and evaluate their performance."""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best models."""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Support Vector Machine': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        self.best_models = {}
        
        for model_name in ['Random Forest', 'Support Vector Machine', 'Logistic Regression']:
            print(f"\nTuning {model_name}...")
            
            # Get the base model
            base_model = self.models[model_name]
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model, 
                param_grids[model_name], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            # Store the best model
            self.best_models[model_name] = grid_search.best_estimator_
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Best CV score: {grid_search.best_score_:.4f}")
            
            # Update results with tuned model
            y_pred = grid_search.best_estimator_.predict(self.X_test_scaled)
            y_pred_proba = grid_search.best_estimator_.predict_proba(self.X_test_scaled)[:, 1]
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            self.results[f'{model_name} (Tuned)'] = {
                'model': grid_search.best_estimator_,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': grid_search.best_score_,
                'cv_std': 0,  # Not available from GridSearchCV
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'best_params': grid_search.best_params_
            }
            
            print(f"  Test Accuracy: {accuracy:.4f}")
    
    def select_best_model(self):
        """Select the best performing model."""
        print("\n" + "="*50)
        print("MODEL SELECTION")
        print("="*50)
        
        # Create a comparison dataframe
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'CV Score': result['cv_mean']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Select best model based on accuracy
        best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = self.results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
        
        return comparison_df
    
    def generate_detailed_report(self):
        """Generate a detailed classification report."""
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*50)
        
        best_result = self.results[self.best_model_name]
        
        print(f"\nBest Model: {self.best_model_name}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, best_result['y_pred'], 
                                  target_names=['Benign', 'Malignant']))
        
        print("\nConfusion Matrix:")
        cm = best_result['confusion_matrix']
        print(f"True Negatives (Benign correctly classified): {cm[0,0]}")
        print(f"False Positives (Benign misclassified as Malignant): {cm[0,1]}")
        print(f"False Negatives (Malignant misclassified as Benign): {cm[1,0]}")
        print(f"True Positives (Malignant correctly classified): {cm[1,1]}")
        
        # Calculate additional metrics
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])  # True Positive Rate
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])  # True Negative Rate
        
        print(f"\nSensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        
        return cm, sensitivity, specificity

if __name__ == "__main__":
    # Initialize the predictor
    predictor = BreastCancerPredictor()
    
    # Load and explore data
    X, y = predictor.load_data()
    stats, target_dist = predictor.explore_data()
    predictor.visualize_data()
    
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test = predictor.preprocess_data()
    
    # Initialize and train models
    predictor.initialize_models()
    predictor.train_models()
    
    # Perform hyperparameter tuning
    predictor.hyperparameter_tuning()
    
    # Select best model and generate report
    comparison_df = predictor.select_best_model()
    cm, sensitivity, specificity = predictor.generate_detailed_report()
    
    print("\nMachine learning model development completed successfully!")

