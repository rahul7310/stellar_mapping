import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from pathlib import Path
import pandas as pd
from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime

class VisualizationManager:
    """Manager class for all visualization and analysis tasks"""
    
    def __init__(self, save_dir: Path, class_names: List[str]):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Setup consistent plot style"""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
    
    def plot_training_history(self, history: Dict, save_name: str = 'training_history.png'):
        """Plot training and validation metrics over time"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Mean Average Precision', 'Learning Rate', 'Exact Match Rate')
        )
        
        # Loss subplot
        fig.add_trace(
            go.Scatter(y=history['train_loss'], name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # MAP subplot
        fig.add_trace(
            go.Scatter(y=history['train_map'], name='Train MAP', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history['val_map'], name='Val MAP', line=dict(color='red')),
            row=1, col=2
        )
        
        # Learning rate subplot
        fig.add_trace(
            go.Scatter(y=history['learning_rate'], name='Learning Rate', line=dict(color='green')),
            row=2, col=1
        )
        
        # Exact match rate subplot
        fig.add_trace(
            go.Scatter(y=history['train_exact_match'], name='Train Exact Match', line=dict(color='blue')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(y=history['val_exact_match'], name='Val Exact Match', line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Training History")
        fig.write_html(self.save_dir / 'training_history.html')
        # fig.write_image(self.save_dir / save_name)
    
    def plot_confusion_matrices(self, confusion_matrices: Dict[str, np.ndarray]):
        """Plot confusion matrices for each class"""
        num_classes = len(self.class_names)
        fig = plt.figure(figsize=(20, 4 * ((num_classes + 3) // 4)))
        
        for idx, class_name in enumerate(self.class_names, 1):
            cm = confusion_matrices[class_name]
            plt.subplot((num_classes + 3) // 4, 4, idx)
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive']
            )
            plt.title(f'{class_name} Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrices.png')
        plt.close()
    
    def plot_roc_curves(self, predictions: np.ndarray, targets: np.ndarray):
        """Plot ROC curves for each class"""
        fig = go.Figure()
        
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
            auc_score = average_precision_score(targets[:, i], predictions[:, i])
            
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=f'{class_name} (AUC = {auc_score:.3f})',
                    mode='lines'
                )
            )
        
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name='Random',
                mode='lines',
                line=dict(dash='dash', color='gray')
            )
        )
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=800
        )
        
        fig.write_html(self.save_dir / 'roc_curves.html')
        # fig.write_image(self.save_dir / 'roc_curves.png')
    
    def plot_precision_recall_curves(self, predictions: np.ndarray, targets: np.ndarray):
        """Plot precision-recall curves for each class"""
        fig = go.Figure()
        
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(targets[:, i], predictions[:, i])
            ap_score = average_precision_score(targets[:, i], predictions[:, i])
            
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    name=f'{class_name} (AP = {ap_score:.3f})',
                    mode='lines'
                )
            )
        
        fig.update_layout(
            title='Precision-Recall Curves',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=800
        )
        
        fig.write_html(self.save_dir / 'pr_curves.html')
        # fig.write_image(self.save_dir / 'pr_curves.png')
    
    def plot_per_class_metrics(self, metrics: Dict):
        """Plot per-class performance metrics"""
        metrics_df = pd.DataFrame({
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'AP': metrics['ap'],
            'AUC': metrics['auc']
        })
        
        fig = px.bar(
            metrics_df,
            barmode='group',
            title='Per-Class Performance Metrics'
        )
        
        fig.update_layout(
            xaxis_title='Metric',
            yaxis_title='Score',
            height=600
        )
        
        fig.write_html(self.save_dir / 'per_class_metrics.html')
        # fig.write_image(self.save_dir / 'per_class_metrics.png')
    
    def visualize_predictions(self, images: torch.Tensor, predictions: torch.Tensor, 
                            targets: torch.Tensor, num_samples: int = 16):
        """Visualize model predictions on sample images"""
        num_samples = min(num_samples, len(images))
        fig = plt.figure(figsize=(20, 4 * ((num_samples + 3) // 4)))
        
        for idx in range(num_samples):
            plt.subplot((num_samples + 3) // 4, 4, idx + 1)
            
            # Convert tensor to numpy and denormalize
            img = images[idx].cpu().numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
            
            plt.imshow(img)
            
            # Get predictions and targets
            pred_classes = [self.class_names[i] for i in range(len(self.class_names)) 
                          if predictions[idx][i] > 0.5]
            true_classes = [self.class_names[i] for i in range(len(self.class_names)) 
                          if targets[idx][i] > 0.5]
            
            plt.title(f'Pred: {", ".join(pred_classes)}\nTrue: {", ".join(true_classes)}',
                     fontsize=8)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'prediction_samples.png')
        plt.close()
    
    def create_evaluation_report(self, metrics: Dict, history: Dict):
        """Create a comprehensive evaluation report"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall_metrics': metrics['overall_metrics'],
            'per_class_metrics': metrics['per_class_metrics'],
            'training_history': {
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'best_val_map': max(history['val_map']),
                'epochs_trained': len(history['train_loss'])
            }
        }
        
        # Save report as JSON
        with open(self.save_dir / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        # Create markdown report
        markdown_report = f"""
# Constellation Classification Evaluation Report

## Overview
- Timestamp: {report['timestamp']}
- Epochs Trained: {report['training_history']['epochs_trained']}

## Overall Performance
- Mean Average Precision: {report['overall_metrics']['mean_ap']:.4f}
- Exact Match Accuracy: {report['overall_metrics']['exact_match']:.4f}
- Hamming Loss: {report['overall_metrics']['hamming_loss']:.4f}
- Subset Accuracy: {report['overall_metrics']['subset_accuracy']:.4f}

## Training History
- Final Training Loss: {report['training_history']['final_train_loss']:.4f}
- Final Validation Loss: {report['training_history']['final_val_loss']:.4f}
- Best Validation MAP: {report['training_history']['best_val_map']:.4f}

## Per-Class Performance
"""
        
        # Add per-class metrics
        for class_name in self.class_names:
            markdown_report += f"\n### {class_name}\n"
            markdown_report += f"- Precision: {metrics['per_class_metrics']['precision'][class_name]:.4f}\n"
            markdown_report += f"- Recall: {metrics['per_class_metrics']['recall'][class_name]:.4f}\n"
            markdown_report += f"- F1 Score: {metrics['per_class_metrics']['f1'][class_name]:.4f}\n"
            markdown_report += f"- Average Precision: {metrics['per_class_metrics']['ap'][class_name]:.4f}\n"
            markdown_report += f"- AUC-ROC: {metrics['per_class_metrics']['auc'][class_name]:.4f}\n"
        
        # Save markdown report
        with open(self.save_dir / 'evaluation_report.md', 'w') as f:
            f.write(markdown_report)
