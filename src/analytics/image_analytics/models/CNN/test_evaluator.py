import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd

class TestEvaluator:
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
        """Run complete model evaluation on test set"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        # Collect predictions and targets
        for images, targets in self.test_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            predictions = torch.sigmoid(outputs)
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        # Convert to numpy arrays
        self.predictions = np.vstack(all_preds)
        self.targets = np.vstack(all_targets)
        
        # Calculate all metrics
        results = {
            'overall_metrics': self.calculate_overall_metrics(),
            'per_class_metrics': self.calculate_per_class_metrics(),
            'thresholds': self.find_optimal_thresholds()
        }
        
        # Generate visualizations
        self.plot_all_visualizations()
        
        # Save detailed results
        self.save_detailed_results(results)
        
        return results
    
    def calculate_overall_metrics(self) -> Dict:
        """Calculate overall model performance metrics"""
        # Binary predictions using 0.5 threshold
        binary_preds = (self.predictions > 0.5).astype(int)
        
        # Exact match ratio (all labels correct)
        exact_match = np.mean(np.all(binary_preds == self.targets, axis=1))
        
        # Hamming loss (ratio of wrong labels)
        hamming_loss = np.mean(binary_preds != self.targets)
        
        # Sample-wise average precision
        mean_ap = average_precision_score(self.targets, self.predictions, average='samples')
        
        # Subset accuracy (all labels in sample correct)
        subset_accuracy = np.mean(np.all(binary_preds == self.targets, axis=1))
        
        # Micro-averaged metrics
        micro_precision = precision_score(self.targets, binary_preds, average='micro')
        micro_recall = recall_score(self.targets, binary_preds, average='micro')
        micro_f1 = f1_score(self.targets, binary_preds, average='micro')
        
        # Macro-averaged metrics
        macro_precision = precision_score(self.targets, binary_preds, average='macro')
        macro_recall = recall_score(self.targets, binary_preds, average='macro')
        macro_f1 = f1_score(self.targets, binary_preds, average='macro')
        
        return {
            'exact_match': float(exact_match),
            'hamming_loss': float(hamming_loss),
            'mean_average_precision': float(mean_ap),
            'subset_accuracy': float(subset_accuracy),
            'micro_precision': float(micro_precision),
            'micro_recall': float(micro_recall),
            'micro_f1': float(micro_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1)
        }
    
    def calculate_per_class_metrics(self) -> Dict:
        """Calculate per-class performance metrics"""
        binary_preds = (self.predictions > 0.5).astype(int)
        metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            # Basic metrics
            precision = precision_score(self.targets[:, i], binary_preds[:, i])
            recall = recall_score(self.targets[:, i], binary_preds[:, i])
            f1 = f1_score(self.targets[:, i], binary_preds[:, i])
            
            # ROC AUC
            roc_auc = roc_auc_score(self.targets[:, i], self.predictions[:, i])
            
            # Average Precision
            ap = average_precision_score(self.targets[:, i], self.predictions[:, i])
            
            # Confusion Matrix
            cm = confusion_matrix(self.targets[:, i], binary_preds[:, i])
            
            metrics[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'average_precision': float(ap),
                'confusion_matrix': cm.tolist(),
                'support': int(np.sum(self.targets[:, i]))
            }
        
        return metrics
    
    def find_optimal_thresholds(self) -> Dict[str, float]:
        """Find optimal classification thresholds for each class"""
        thresholds = {}
        
        for i, class_name in enumerate(self.class_names):
            precision, recall, thrs = precision_recall_curve(
                self.targets[:, i], self.predictions[:, i]
            )
            
            # Calculate F1 score for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            # Find threshold with best F1 score
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thrs[optimal_idx] if optimal_idx < len(thrs) else 0.5
            
            thresholds[class_name] = float(optimal_threshold)
        
        return thresholds
    
    def plot_all_visualizations(self):
        """Generate all visualization plots"""
        self._plot_roc_curves()
        self._plot_precision_recall_curves()
        self._plot_confusion_matrices()
        self._plot_score_distributions()
    
    def _plot_roc_curves(self):
        """Plot ROC curves for each class"""
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
    
    def _plot_precision_recall_curves(self):
        """Plot precision-recall curves for each class"""
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
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for each class"""
        binary_preds = (self.predictions > 0.5).astype(int)
        
        n_cols = 4
        n_rows = (len(self.class_names) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.class_names):
            cm = confusion_matrix(self.targets[:, i], binary_preds[:, i])
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                ax=axes[i],
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive']
            )
            axes[i].set_title(f'{class_name}')
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrices.png')
        plt.close()
    
    def _plot_score_distributions(self):
        """Plot score distributions for each class"""
        n_cols = 4
        n_rows = (len(self.class_names) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.class_names):
            # Get predictions for positive and negative samples
            pos_scores = self.predictions[:, i][self.targets[:, i] == 1]
            neg_scores = self.predictions[:, i][self.targets[:, i] == 0]
            
            # Plot distributions
            if len(pos_scores) > 0:
                axes[i].hist(pos_scores, bins=50, alpha=0.5, label='Positive', density=True)
            if len(neg_scores) > 0:
                axes[i].hist(neg_scores, bins=50, alpha=0.5, label='Negative', density=True)
            
            axes[i].set_title(f'{class_name}')
            axes[i].legend()
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'score_distributions.png')
        plt.close()
    
    def save_detailed_results(self, results: Dict):
        """Save detailed evaluation results"""
        # Save raw results as JSON
        import json
        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Create detailed markdown report
        report = self._create_markdown_report(results)
        with open(self.save_dir / 'evaluation_report.md', 'w') as f:
            f.write(report)
        
        # Create Excel report with multiple sheets
        self._create_excel_report(results)
    
    def _create_markdown_report(self, results: Dict) -> str:
        """Create detailed markdown report"""
        report = "# Model Evaluation Report\n\n"
        
        # Overall Metrics
        report += "## Overall Metrics\n\n"
        for metric, value in results['overall_metrics'].items():
            report += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        # Per-Class Metrics
        report += "\n## Per-Class Performance\n\n"
        for class_name, metrics in results['per_class_metrics'].items():
            report += f"\n### {class_name}\n"
            for metric, value in metrics.items():
                if metric != 'confusion_matrix':
                    report += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        # Optimal Thresholds
        report += "\n## Optimal Thresholds\n\n"
        for class_name, threshold in results['thresholds'].items():
            report += f"- {class_name}: {threshold:.4f}\n"
        
        return report
    
    def _create_excel_report(self, results: Dict):
        """Create detailed Excel report with multiple sheets"""
        writer = pd.ExcelWriter(self.save_dir / 'evaluation_results.xlsx', engine='xlsxwriter')
        
        # Overall metrics sheet
        pd.DataFrame([results['overall_metrics']]).T.to_excel(writer, sheet_name='Overall Metrics')
        
        # Per-class metrics sheet
        per_class_df = pd.DataFrame.from_dict(results['per_class_metrics'], orient='index')
        per_class_df = per_class_df.drop('confusion_matrix', axis=1)
        per_class_df.to_excel(writer, sheet_name='Per-Class Metrics')
        
        # Optimal thresholds sheet
        pd.Series(results['thresholds']).to_excel(writer, sheet_name='Optimal Thresholds')
        
        writer.close()

def run_evaluation(model, test_loader, device, class_names, save_dir):
    """Main function to run evaluation"""
    evaluator = TestEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        save_dir=save_dir
    )
    
    results = evaluator.evaluate()
    print("\nEvaluation completed!")
    print("\nOverall Metrics:")
    for metric, value in results['overall_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    return results
