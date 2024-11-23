import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json
import pandas as pd
from torch.utils.data import DataLoader

class ModelEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        class_names: List[str],
        device: torch.device,
        output_dir: str
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run complete model evaluation"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        for batch in self.test_loader:
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(pixel_values)
            predictions = torch.sigmoid(outputs)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        predictions = np.array(all_preds)
        labels = np.array(all_labels)
        
        results = {
            'overall_metrics': self.calculate_overall_metrics(predictions, labels),
            'per_class_metrics': self.calculate_per_class_metrics(predictions, labels),
            'thresholds': self.find_optimal_thresholds(predictions, labels)
        }
        
        # Save results
        self.save_results(results)
        self.plot_results(predictions, labels)
        
        return results
    
    def calculate_overall_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate overall model performance metrics"""
        # Binary predictions using 0.5 threshold
        binary_preds = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        exact_match = np.mean(np.all(binary_preds == labels, axis=1))
        hamming_loss = np.mean(binary_preds != labels)
        
        # Sample-averaged metrics
        precision = np.mean([np.mean((labels[i] == 1) & (binary_preds[i] == 1)) 
                           for i in range(len(labels))])
        recall = np.mean([np.mean((labels[i] == 1) & (binary_preds[i] == 1)) / 
                         (np.sum(labels[i] == 1) + 1e-10) for i in range(len(labels))])
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        return {
            'exact_match': float(exact_match),
            'hamming_loss': float(hamming_loss),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    
    def calculate_per_class_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate per-class performance metrics"""
        metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            class_preds = predictions[:, i]
            class_labels = labels[:, i]
            binary_preds = (class_preds > 0.5).astype(int)
            
            # Calculate metrics
            precision = np.mean((class_labels == 1) & (binary_preds == 1)) / (np.sum(binary_preds == 1) + 1e-10)
            recall = np.mean((class_labels == 1) & (binary_preds == 1)) / (np.sum(class_labels == 1) + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            # Calculate AUC
            fpr, tpr, _ = roc_curve(class_labels, class_preds)
            auc_score = auc(fpr, tpr)
            
            metrics[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc_score)
            }
        
        return metrics
    
    def find_optimal_thresholds(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Find optimal classification thresholds for each class"""
        thresholds = {}
        
        for i, class_name in enumerate(self.class_names):
            precision, recall, thrs = precision_recall_curve(labels[:, i], predictions[:, i])
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thrs[optimal_idx] if optimal_idx < len(thrs) else 0.5
            thresholds[class_name] = float(optimal_threshold)
        
        return thresholds
    
    def plot_results(self, predictions: np.ndarray, labels: np.ndarray):
        """Create visualization plots for results"""
        # 1. ROC curves
        plt.figure(figsize=(10, 6))
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png')
        plt.close()
        
        # 2. Confusion matrices
        binary_preds = (predictions > 0.5).astype(int)
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.class_names):
            cm = pd.crosstab(
                labels[:, i],
                binary_preds[:, i],
                margins=True
            )
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
            axes[i].set_title(f'{class_name} Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png')
        plt.close()
    
    def save_results(self, results: Dict):
        """Save evaluation results"""
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
