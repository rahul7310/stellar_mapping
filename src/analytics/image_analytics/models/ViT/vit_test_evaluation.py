import torch
from pathlib import Path
from data_preprocessing import DataModule
from vit_model import create_model
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix
)
import pandas as pd
import json
from datetime import datetime

class ViTEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        class_names: List[str],
        save_dir: Path
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Full model evaluation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_image_paths = []
        
        # Collect predictions
        for batch in self.test_loader:
            # For ViT, input is in batch['pixel_values']
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels']
            
            # Forward pass
            outputs = self.model(pixel_values)
            predictions = torch.sigmoid(outputs)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(labels.numpy())
            
        # Concatenate all predictions and targets
        self.predictions = np.vstack(all_predictions)
        self.targets = np.vstack(all_targets)
        
        # Calculate all metrics
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall_metrics': self.calculate_overall_metrics(),
            'per_class_metrics': self.calculate_per_class_metrics(),
            'thresholds': self.find_optimal_thresholds()
        }
        
        # Generate visualizations
        self.generate_all_plots()
        
        # Save results
        self.save_results(results)
        
        return results
    
    def calculate_overall_metrics(self) -> Dict:
        """Calculate overall model performance metrics"""
        # Convert probabilities to binary predictions
        binary_preds = (self.predictions > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'exact_match_ratio': np.mean(np.all(binary_preds == self.targets, axis=1)),
            'hamming_loss': np.mean(binary_preds != self.targets),
            'mean_average_precision': average_precision_score(
                self.targets, self.predictions, average='samples'
            ),
            'subset_accuracy': np.mean(np.all(binary_preds == self.targets, axis=1)),
            'micro_precision': precision_score(
                self.targets, binary_preds, average='micro'
            ),
            'micro_recall': recall_score(
                self.targets, binary_preds, average='micro'
            ),
            'micro_f1': f1_score(
                self.targets, binary_preds, average='micro'
            ),
            'macro_precision': precision_score(
                self.targets, binary_preds, average='macro'
            ),
            'macro_recall': recall_score(
                self.targets, binary_preds, average='macro'
            ),
            'macro_f1': f1_score(
                self.targets, binary_preds, average='macro'
            ),
            'samples_precision': precision_score(
                self.targets, binary_preds, average='samples'
            ),
            'samples_recall': recall_score(
                self.targets, binary_preds, average='samples'
            ),
            'samples_f1': f1_score(
                self.targets, binary_preds, average='samples'
            )
        }
        
        return {k: float(v) for k, v in metrics.items()}
    
    def calculate_per_class_metrics(self) -> Dict:
        """Calculate per-class performance metrics"""
        binary_preds = (self.predictions > 0.5).astype(int)
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            # Basic metrics
            metrics = {
                'precision': precision_score(
                    self.targets[:, i], binary_preds[:, i]
                ),
                'recall': recall_score(
                    self.targets[:, i], binary_preds[:, i]
                ),
                'f1': f1_score(
                    self.targets[:, i], binary_preds[:, i]
                ),
                'average_precision': average_precision_score(
                    self.targets[:, i], self.predictions[:, i]
                ),
                'roc_auc': roc_auc_score(
                    self.targets[:, i], self.predictions[:, i]
                ),
                'support': int(np.sum(self.targets[:, i])),
                'confusion_matrix': confusion_matrix(
                    self.targets[:, i], binary_preds[:, i]
                ).tolist()
            }
            
            per_class_metrics[class_name] = {
                k: float(v) if not isinstance(v, list) else v 
                for k, v in metrics.items()
            }
        
        return per_class_metrics
    
    def find_optimal_thresholds(self) -> Dict[str, float]:
        """Find optimal classification thresholds for each class"""
        thresholds = {}
        
        for i, class_name in enumerate(self.class_names):
            precision, recall, thrs = precision_recall_curve(
                self.targets[:, i], self.predictions[:, i]
            )
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thrs[optimal_idx] if optimal_idx < len(thrs) else 0.5
            thresholds[class_name] = float(optimal_threshold)
        
        return thresholds
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_confusion_matrices()
        self.plot_prediction_distributions()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all classes"""
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(self.targets[:, i], self.predictions[:, i])
            auc = roc_auc_score(self.targets[:, i], self.predictions[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'roc_curves.png')
        plt.close()
    
    def plot_precision_recall_curves(self):
        """Plot precision-recall curves for all classes"""
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(
                self.targets[:, i], self.predictions[:, i]
            )
            ap = average_precision_score(self.targets[:, i], self.predictions[:, i])
            plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'precision_recall_curves.png')
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all classes"""
        binary_preds = (self.predictions > 0.5).astype(int)
        
        n_cols = 4
        n_rows = (len(self.class_names) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.class_names):
            cm = confusion_matrix(self.targets[:, i], binary_preds[:, i])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{class_name}')
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrices.png')
        plt.close()
    
    def plot_prediction_distributions(self):
        """Plot prediction score distributions for each class"""
        n_cols = 4
        n_rows = (len(self.class_names) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.class_names):
            pos_scores = self.predictions[:, i][self.targets[:, i] == 1]
            neg_scores = self.predictions[:, i][self.targets[:, i] == 0]
            
            axes[i].hist(pos_scores, bins=50, alpha=0.5, label='Positive', density=True)
            axes[i].hist(neg_scores, bins=50, alpha=0.5, label='Negative', density=True)
            axes[i].set_title(f'{class_name}')
            axes[i].legend()
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'prediction_distributions.png')
        plt.close()
    
    def save_results(self, results: Dict):
        """Save evaluation results in multiple formats"""
        # Save JSON
        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save Excel report
        self.save_excel_report(results)
        
        # Save Markdown report
        self.save_markdown_report(results)
    
    def save_excel_report(self, results: Dict):
        """Save results as Excel report with multiple sheets"""
        with pd.ExcelWriter(self.save_dir / 'evaluation_results.xlsx') as writer:
            # Overall metrics
            pd.DataFrame([results['overall_metrics']]).T.to_excel(
                writer, sheet_name='Overall Metrics'
            )
            
            # Per-class metrics
            per_class_df = pd.DataFrame.from_dict(results['per_class_metrics'], orient='index')
            per_class_df = per_class_df.drop('confusion_matrix', axis=1)
            per_class_df.to_excel(writer, sheet_name='Per-Class Metrics')
            
            # Thresholds
            pd.Series(results['thresholds']).to_excel(
                writer, sheet_name='Optimal Thresholds'
            )
    
    def save_markdown_report(self, results: Dict):
        """Save results as markdown report"""
        report = f"# ViT Model Evaluation Report\n\n"
        report += f"Evaluation Time: {results['timestamp']}\n\n"
        
        # Overall metrics
        report += "## Overall Metrics\n\n"
        for metric, value in results['overall_metrics'].items():
            report += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        # Per-class metrics
        report += "\n## Per-Class Performance\n\n"
        for class_name, metrics in results['per_class_metrics'].items():
            report += f"\n### {class_name}\n"
            for metric, value in metrics.items():
                if metric != 'confusion_matrix':
                    report += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        # Thresholds
        report += "\n## Optimal Thresholds\n\n"
        for class_name, threshold in results['thresholds'].items():
            report += f"- {class_name}: {threshold:.4f}\n"
        
        with open(self.save_dir / 'evaluation_report.md', 'w') as f:
            f.write(report)

def evaluate_vit_model(
    model_path: str,
    data_module: DataModule,
    device: torch.device,
    save_dir: str = "vit_evaluation_results"
):
    """Main function to evaluate ViT model"""
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = create_model(num_classes=16)  # Adjust based on your config
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Get test dataloader
    test_loader = data_module.get_dataloaders()[1]  # Assuming it returns (train, test)
    
    # Get class names
    class_names = data_module.get_class_names()
    
    # Create evaluator and run evaluation
    evaluator = ViTEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        save_dir=save_dir
    )
    
    results = evaluator.evaluate()
    
    print("\nEvaluation completed! Results saved to:", save_dir)
    print("\nOverall Metrics:")
    for metric, value in results['overall_metrics'].items():
        print(f"{metric:25s}: {value:.4f}")
    
    return results