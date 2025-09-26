#!/usr/bin/env python3
"""
Additional Visualizations for Breast Cancer Prediction Project
Author: Manus AI
Date: September 2025

This script creates additional visualizations and diagrams for the project report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VisualizationGenerator:
    """
    Generate additional visualizations for the breast cancer prediction project.
    """
    
    def __init__(self):
        """Initialize the visualization generator."""
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        
    def load_data(self):
        """Load and prepare the dataset."""
        print("Loading data for visualization generation...")
        
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
        
        print("Data loaded successfully!")
        
    def create_methodology_flowchart(self):
        """Create a methodology flowchart diagram."""
        print("Creating methodology flowchart...")
        
        # Create a flowchart using matplotlib
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Define boxes and their positions
        boxes = [
            {"text": "Data Collection\n(Wisconsin Breast Cancer Dataset)", "pos": (5, 11), "color": "lightblue"},
            {"text": "Exploratory Data Analysis\n(569 samples, 30 features)", "pos": (5, 9.5), "color": "lightgreen"},
            {"text": "Data Preprocessing\n(Scaling, Train-Test Split)", "pos": (5, 8), "color": "lightyellow"},
            {"text": "Model Training\n(5 ML Algorithms)", "pos": (5, 6.5), "color": "lightcoral"},
            {"text": "Hyperparameter Tuning\n(GridSearchCV)", "pos": (5, 5), "color": "lightpink"},
            {"text": "Model Evaluation\n(Cross-validation, Metrics)", "pos": (5, 3.5), "color": "lightgray"},
            {"text": "Best Model Selection\n(Logistic Regression)", "pos": (5, 2), "color": "gold"},
            {"text": "Final Testing & Validation\n(98.25% Accuracy)", "pos": (5, 0.5), "color": "lightsteelblue"}
        ]
        
        # Draw boxes and text
        for i, box in enumerate(boxes):
            # Draw rectangle
            rect = plt.Rectangle((box["pos"][0]-1.5, box["pos"][1]-0.4), 3, 0.8, 
                               facecolor=box["color"], edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Add text
            ax.text(box["pos"][0], box["pos"][1], box["text"], 
                   ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Draw arrows (except for the last box)
            if i < len(boxes) - 1:
                ax.arrow(box["pos"][0], box["pos"][1]-0.5, 0, -0.6, 
                        head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        plt.title('Breast Cancer Prediction System Methodology', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('/home/ubuntu/methodology_flowchart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Methodology flowchart saved as 'methodology_flowchart.png'")
    
    def create_model_comparison_chart(self):
        """Create a comprehensive model comparison chart."""
        print("Creating model comparison chart...")
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Support Vector Machine': SVC(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }
        
        # Train models and collect results
        results = []
        for name, model in models.items():
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)
            results.append({'Model': name, 'Accuracy': accuracy})
        
        results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=True)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        bars = ax.barh(results_df['Model'], results_df['Accuracy'], color=colors)
        
        # Add value labels on bars
        for i, (bar, accuracy) in enumerate(zip(bars, results_df['Accuracy'])):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{accuracy:.3f}', va='center', fontweight='bold')
        
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Machine Learning Models', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison\nBreast Cancer Prediction', fontsize=14, fontweight='bold')
        ax.set_xlim(0.85, 1.0)
        
        # Add grid for better readability
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Model comparison chart saved as 'model_comparison.png'")
    
    def create_feature_analysis_visualization(self):
        """Create detailed feature analysis visualizations."""
        print("Creating feature analysis visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Feature correlation with target
        ax1 = axes[0, 0]
        correlations = []
        for feature in self.X.columns:
            corr = np.corrcoef(self.X[feature], self.y)[0, 1]
            correlations.append(abs(corr))
        
        feature_corr = pd.DataFrame({
            'feature': self.X.columns,
            'correlation': correlations
        }).sort_values('correlation', ascending=False).head(15)
        
        bars = ax1.barh(range(len(feature_corr)), feature_corr['correlation'])
        ax1.set_yticks(range(len(feature_corr)))
        ax1.set_yticklabels(feature_corr['feature'], fontsize=8)
        ax1.set_xlabel('Absolute Correlation with Target')
        ax1.set_title('Top 15 Features by Correlation with Diagnosis')
        ax1.invert_yaxis()
        
        # 2. Feature distribution by diagnosis
        ax2 = axes[0, 1]
        feature_to_plot = 'worst concave points'
        benign_data = self.X[self.y == 0][feature_to_plot]
        malignant_data = self.X[self.y == 1][feature_to_plot]
        
        ax2.hist(benign_data, alpha=0.7, label='Benign', color='lightblue', bins=20)
        ax2.hist(malignant_data, alpha=0.7, label='Malignant', color='lightcoral', bins=20)
        ax2.set_xlabel(feature_to_plot.title())
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{feature_to_plot.title()} Distribution')
        ax2.legend()
        
        # 3. Feature importance from Random Forest
        ax3 = axes[1, 0]
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train_scaled, self.y_train)
        
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        bars = ax3.barh(range(len(feature_importance)), feature_importance['importance'])
        ax3.set_yticks(range(len(feature_importance)))
        ax3.set_yticklabels(feature_importance['feature'], fontsize=8)
        ax3.set_xlabel('Feature Importance')
        ax3.set_title('Top 15 Features by Random Forest Importance')
        ax3.invert_yaxis()
        
        # 4. Pairwise feature relationships
        ax4 = axes[1, 1]
        important_features = ['worst perimeter', 'worst concave points']
        
        colors = ['blue' if x == 0 else 'red' for x in self.y]
        scatter = ax4.scatter(self.X['worst perimeter'], self.X['worst concave points'], 
                            c=colors, alpha=0.6)
        ax4.set_xlabel('Worst Perimeter')
        ax4.set_ylabel('Worst Concave Points')
        ax4.set_title('Feature Relationship: Perimeter vs Concave Points')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Benign'),
                          Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Malignant')]
        ax4.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Feature analysis visualization saved as 'feature_analysis.png'")
    
    def create_learning_curve_visualization(self):
        """Create learning curve visualization."""
        print("Creating learning curve visualization...")
        
        # Create learning curves for the best model
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, self.X_train_scaled, self.y_train, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Accuracy')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy')
        ax.set_title('Learning Curve - Logistic Regression Model')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/learning_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Learning curve visualization saved as 'learning_curve.png'")
    
    def create_system_architecture_diagram(self):
        """Create a system architecture diagram."""
        print("Creating system architecture diagram...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Define components
        components = [
            # Input Layer
            {"text": "FNA Sample\nData Input", "pos": (2, 8.5), "size": (2, 1), "color": "lightblue"},
            
            # Preprocessing Layer
            {"text": "Data\nPreprocessing", "pos": (6, 8.5), "size": (2, 1), "color": "lightgreen"},
            {"text": "Feature\nScaling", "pos": (6, 7), "size": (2, 1), "color": "lightgreen"},
            
            # Model Layer
            {"text": "Logistic\nRegression\nModel", "pos": (10, 7.75), "size": (2.5, 1.5), "color": "lightcoral"},
            
            # Output Layer
            {"text": "Prediction\nOutput", "pos": (6, 5), "size": (2, 1), "color": "gold"},
            {"text": "Confidence\nScore", "pos": (10, 5), "size": (2, 1), "color": "gold"},
            
            # Evaluation Components
            {"text": "Performance\nMetrics", "pos": (2, 3), "size": (2, 1), "color": "lightgray"},
            {"text": "Confusion\nMatrix", "pos": (6, 3), "size": (2, 1), "color": "lightgray"},
            {"text": "ROC Curve\nAnalysis", "pos": (10, 3), "size": (2, 1), "color": "lightgray"},
            
            # Database
            {"text": "Training\nDatabase\n(569 samples)", "pos": (2, 6), "size": (2, 1.5), "color": "lightyellow"}
        ]
        
        # Draw components
        for comp in components:
            rect = plt.Rectangle((comp["pos"][0]-comp["size"][0]/2, comp["pos"][1]-comp["size"][1]/2), 
                               comp["size"][0], comp["size"][1], 
                               facecolor=comp["color"], edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            ax.text(comp["pos"][0], comp["pos"][1], comp["text"], 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw arrows
        arrows = [
            # Main flow
            ((3, 8.5), (5, 8.5)),  # Input to Preprocessing
            ((7, 8.5), (7, 7.5)),  # Preprocessing to Scaling
            ((8, 7.75), (8.75, 7.75)),  # Scaling to Model
            ((10, 7), (7, 5.5)),  # Model to Prediction
            ((10, 6.5), (10, 6)),  # Model to Confidence
            
            # Training flow
            ((3, 6.75), (5, 7.5)),  # Database to Preprocessing
            
            # Evaluation flow
            ((6, 4.5), (3, 3.5)),  # Output to Metrics
            ((6, 4.5), (6, 4)),   # Output to Confusion Matrix
            ((10, 4.5), (10, 4))  # Confidence to ROC
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        plt.title('Breast Cancer Prediction System Architecture', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('/home/ubuntu/system_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("System architecture diagram saved as 'system_architecture.png'")

if __name__ == "__main__":
    # Initialize the visualization generator
    viz_gen = VisualizationGenerator()
    
    # Load data
    viz_gen.load_data()
    
    # Create all visualizations
    viz_gen.create_methodology_flowchart()
    viz_gen.create_model_comparison_chart()
    viz_gen.create_feature_analysis_visualization()
    viz_gen.create_learning_curve_visualization()
    viz_gen.create_system_architecture_diagram()
    
    print("\nAll additional visualizations created successfully!")

