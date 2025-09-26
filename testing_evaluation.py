#!/usr/bin/env python3
"""
Testing and Performance Evaluation for Breast Cancer Prediction System
Author: Manus AI
Date: September 2025

This script implements comprehensive testing and evaluation of the breast cancer prediction system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_curve, auc, roc_auc_score, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelTester:
    """
    Comprehensive testing and evaluation system for breast cancer prediction models.
    """
    
    def __init__(self):
        """Initialize the tester."""
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        print("Loading and preparing data for testing...")
        
        # Load dataset
        self.data = load_breast_cancer()
        self.X = pd.DataFrame(self.data.data, columns=self.data.feature_names)
        self.y = pd.Series(self.data.target, name='diagnosis')
        
        # Split and scale data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train the best model (Logistic Regression based on previous results)
        self.best_model = LogisticRegression(random_state=42, max_iter=1000)
        self.best_model.fit(self.X_train_scaled, self.y_train)
        
        print("Data loaded and best model trained successfully!")
        
    def test_model_robustness(self):
        """Test model robustness with different scenarios."""
        print("\n" + "="*50)
        print("MODEL ROBUSTNESS TESTING")
        print("="*50)
        
        robustness_results = {}
        
        # Test 1: Different train-test splits
        print("\nTesting with different train-test splits...")
        split_accuracies = []
        for i, test_size in enumerate([0.1, 0.15, 0.2, 0.25, 0.3]):
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42+i, stratify=self.y
            )
            
            X_train_scaled_temp = self.scaler.fit_transform(X_train_temp)
            X_test_scaled_temp = self.scaler.transform(X_test_temp)
            
            model_temp = LogisticRegression(random_state=42, max_iter=1000)
            model_temp.fit(X_train_scaled_temp, y_train_temp)
            
            y_pred_temp = model_temp.predict(X_test_scaled_temp)
            accuracy = accuracy_score(y_test_temp, y_pred_temp)
            split_accuracies.append(accuracy)
            
            print(f"  Test size {test_size:.1f}: Accuracy = {accuracy:.4f}")
        
        robustness_results['split_test'] = {
            'accuracies': split_accuracies,
            'mean': np.mean(split_accuracies),
            'std': np.std(split_accuracies)
        }
        
        # Test 2: Cross-validation stability
        print("\nTesting cross-validation stability...")
        cv_scores = cross_val_score(self.best_model, self.X_train_scaled, self.y_train, cv=10)
        robustness_results['cv_stability'] = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        print(f"  10-fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Test 3: Feature subset performance
        print("\nTesting with feature subsets...")
        feature_subset_results = []
        feature_groups = {
            'mean_features': [col for col in self.X.columns if 'mean' in col],
            'se_features': [col for col in self.X.columns if 'se' in col],
            'worst_features': [col for col in self.X.columns if 'worst' in col]
        }
        
        for group_name, features in feature_groups.items():
            X_subset = self.X[features]
            X_train_subset, X_test_subset, _, _ = train_test_split(
                X_subset, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
            
            # Create a new scaler for each subset
            scaler_subset = StandardScaler()
            X_train_subset_scaled = scaler_subset.fit_transform(X_train_subset)
            X_test_subset_scaled = scaler_subset.transform(X_test_subset)
            
            model_subset = LogisticRegression(random_state=42, max_iter=1000)
            model_subset.fit(X_train_subset_scaled, self.y_train)
            
            y_pred_subset = model_subset.predict(X_test_subset_scaled)
            accuracy = accuracy_score(self.y_test, y_pred_subset)
            feature_subset_results.append(accuracy)
            
            print(f"  {group_name}: Accuracy = {accuracy:.4f}")
        
        robustness_results['feature_subsets'] = feature_subset_results
        
        return robustness_results
    
    def evaluate_performance_metrics(self):
        """Comprehensive performance evaluation."""
        print("\n" + "="*50)
        print("COMPREHENSIVE PERFORMANCE EVALUATION")
        print("="*50)
        
        # Make predictions
        y_pred = self.best_model.predict(self.X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate all metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        # Confusion matrix analysis
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        sensitivity = tp / (tp + fn)  # True Positive Rate
        specificity = tn / (tn + fp)  # True Negative Rate
        ppv = tp / (tp + fp)  # Positive Predictive Value
        npv = tn / (tn + fn)  # Negative Predictive Value
        
        metrics.update({
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        })
        
        print("\nPerformance Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Positive Predictive Value: {metrics['ppv']:.4f}")
        print(f"Negative Predictive Value: {metrics['npv']:.4f}")
        
        print(f"\nConfusion Matrix Analysis:")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        
        return metrics, y_pred, y_pred_proba
    
    def create_performance_visualizations(self, metrics, y_pred, y_pred_proba):
        """Create comprehensive performance visualizations."""
        print("\nCreating performance visualizations...")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix Heatmap
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malignant'], 
                   yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. ROC Curve
        plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        
        # 3. Precision-Recall Curve
        plt.subplot(2, 3, 3)
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        
        # 4. Feature Importance
        plt.subplot(2, 3, 4)
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': np.abs(self.best_model.coef_[0])
        }).sort_values('importance', ascending=False).head(10)
        
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Feature Importance (|Coefficient|)')
        plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # 5. Performance Metrics Bar Chart
        plt.subplot(2, 3, 5)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['specificity'], metrics['f1_score']]
        
        bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        plt.ylim(0, 1)
        plt.title('Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Prediction Probability Distribution
        plt.subplot(2, 3, 6)
        benign_probs = y_pred_proba[self.y_test == 0]
        malignant_probs = y_pred_proba[self.y_test == 1]
        
        plt.hist(benign_probs, alpha=0.7, label='Benign', color='lightblue', bins=20)
        plt.hist(malignant_probs, alpha=0.7, label='Malignant', color='lightcoral', bins=20)
        plt.xlabel('Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Threshold')
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/performance_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Performance evaluation plots saved as 'performance_evaluation.png'")
    
    def test_edge_cases(self):
        """Test the model with edge cases and boundary conditions."""
        print("\n" + "="*50)
        print("EDGE CASE TESTING")
        print("="*50)
        
        edge_case_results = {}
        
        # Test with extreme values
        print("\nTesting with extreme feature values...")
        
        # Create test cases with extreme values
        extreme_cases = []
        
        # Case 1: All minimum values
        min_case = self.X.min().values.reshape(1, -1)
        extreme_cases.append(("All minimum values", min_case))
        
        # Case 2: All maximum values
        max_case = self.X.max().values.reshape(1, -1)
        extreme_cases.append(("All maximum values", max_case))
        
        # Case 3: All mean values
        mean_case = self.X.mean().values.reshape(1, -1)
        extreme_cases.append(("All mean values", mean_case))
        
        for case_name, case_data in extreme_cases:
            case_data_scaled = self.scaler.transform(case_data)
            prediction = self.best_model.predict(case_data_scaled)[0]
            probability = self.best_model.predict_proba(case_data_scaled)[0]
            
            edge_case_results[case_name] = {
                'prediction': prediction,
                'probability': probability
            }
            
            print(f"  {case_name}:")
            print(f"    Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
            print(f"    Probability: {probability[1]:.4f}")
        
        return edge_case_results
    
    def generate_test_report(self, robustness_results, metrics, edge_case_results):
        """Generate a comprehensive test report."""
        print("\n" + "="*50)
        print("COMPREHENSIVE TEST REPORT")
        print("="*50)
        
        print("\n1. MODEL ROBUSTNESS:")
        print(f"   - Split Test Stability: {robustness_results['split_test']['mean']:.4f} ± {robustness_results['split_test']['std']:.4f}")
        print(f"   - Cross-Validation Stability: {robustness_results['cv_stability']['mean']:.4f} ± {robustness_results['cv_stability']['std']:.4f}")
        
        print("\n2. PERFORMANCE METRICS:")
        print(f"   - Accuracy: {metrics['accuracy']:.4f}")
        print(f"   - Precision: {metrics['precision']:.4f}")
        print(f"   - Recall: {metrics['recall']:.4f}")
        print(f"   - Specificity: {metrics['specificity']:.4f}")
        print(f"   - F1-Score: {metrics['f1_score']:.4f}")
        print(f"   - ROC AUC: {metrics['roc_auc']:.4f}")
        
        print("\n3. CLINICAL RELEVANCE:")
        print(f"   - False Negative Rate: {metrics['false_negatives']/(metrics['false_negatives']+metrics['true_positives']):.4f}")
        print(f"   - False Positive Rate: {metrics['false_positives']/(metrics['false_positives']+metrics['true_negatives']):.4f}")
        
        print("\n4. EDGE CASE HANDLING:")
        for case_name, result in edge_case_results.items():
            print(f"   - {case_name}: {'Malignant' if result['prediction'] == 1 else 'Benign'} ({result['probability'][1]:.4f})")
        
        print("\n5. OVERALL ASSESSMENT:")
        if metrics['accuracy'] >= 0.95 and metrics['recall'] >= 0.95:
            print("   ✓ EXCELLENT: Model meets clinical requirements")
        elif metrics['accuracy'] >= 0.90 and metrics['recall'] >= 0.90:
            print("   ✓ GOOD: Model performance is acceptable")
        else:
            print("   ⚠ NEEDS IMPROVEMENT: Model requires further optimization")

if __name__ == "__main__":
    # Initialize the tester
    tester = ModelTester()
    
    # Load data and prepare model
    tester.load_and_prepare_data()
    
    # Perform comprehensive testing
    robustness_results = tester.test_model_robustness()
    metrics, y_pred, y_pred_proba = tester.evaluate_performance_metrics()
    tester.create_performance_visualizations(metrics, y_pred, y_pred_proba)
    edge_case_results = tester.test_edge_cases()
    
    # Generate final test report
    tester.generate_test_report(robustness_results, metrics, edge_case_results)
    
    print("\nTesting and performance evaluation completed successfully!")

