import json
from matplotlib import pyplot as plt
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
from datetime import datetime
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, auc, confusion_matrix
)
from tqdm import tqdm
import seaborn as sns
import yaml

from data_preprocessing import DataModule, DatasetType

class EnsembleEvaluator:
    """Comprehensive evaluation for ensemble model"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        data_module: 'DataModule',
        device: torch.device,
        class_names: List[str],
        save_dir: str
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.class_names = class_names
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.save_dir / 'visualizations').mkdir(exist_ok=True)
        (self.save_dir / 'predictions').mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run complete model evaluation"""
        self.model.eval()
        test_loader = self.data_module.get_dataloader(DatasetType.TEST)
        
        # Initialize storage
        all_ensemble_preds = []
        all_cnn_preds = []
        all_vit_preds = []
        all_fusion_preds = []
        all_targets = []
        
        logging.info("Starting evaluation...")
        
        # Collect predictions
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            cnn_inputs = batch['cnn_input'].to(self.device)
            vit_inputs = batch['vit_input'].to(self.device)
            targets = batch['labels']
            
            # Get predictions
            outputs = self.model(
                cnn_input=cnn_inputs,
                vit_input=vit_inputs,
                return_features=True
            )
            
            # Apply sigmoid to logits
            ensemble_preds = torch.sigmoid(outputs['ensemble_logits'])
            cnn_preds = torch.sigmoid(outputs['cnn_logits'])
            vit_preds = torch.sigmoid(outputs['vit_logits'])
            fusion_preds = torch.sigmoid(outputs['fusion_logits'])
            
            # Store predictions and targets
            all_ensemble_preds.append(ensemble_preds.cpu().numpy())
            all_cnn_preds.append(cnn_preds.cpu().numpy())
            all_vit_preds.append(vit_preds.cpu().numpy())
            all_fusion_preds.append(fusion_preds.cpu().numpy())
            all_targets.append(targets.numpy())
        
        # Concatenate all predictions and targets
        self.ensemble_preds = np.vstack(all_ensemble_preds)
        self.cnn_preds = np.vstack(all_cnn_preds)
        self.vit_preds = np.vstack(all_vit_preds)
        self.fusion_preds = np.vstack(all_fusion_preds)
        self.targets = np.vstack(all_targets)
        
        # Calculate all metrics
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ensemble_metrics': self._calculate_model_metrics(self.ensemble_preds, 'ensemble'),
            'cnn_metrics': self._calculate_model_metrics(self.cnn_preds, 'cnn'),
            'vit_metrics': self._calculate_model_metrics(self.vit_preds, 'vit'),
            'fusion_metrics': self._calculate_model_metrics(self.fusion_preds, 'fusion'),
            'model_weights': self._get_model_weights()
        }
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Save results
        self._save_results(results)
        
        logging.info("Evaluation completed successfully")
        return results
    
    def _calculate_model_metrics(self, predictions: np.ndarray, model_name: str) -> Dict:
        """Calculate comprehensive metrics for a model's predictions"""
        # Convert to binary predictions
        binary_preds = (predictions > 0.5).astype(int)
        
        # Calculate overall metrics
        overall_metrics = {
            'exact_match': np.mean(np.all(binary_preds == self.targets, axis=1)),
            'hamming_loss': np.mean(binary_preds != self.targets),
            'mean_average_precision': average_precision_score(
                self.targets, predictions, average='samples'
            )
        }
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            # Basic metrics
            cm = confusion_matrix(self.targets[:, i], binary_preds[:, i])
            tn, fp, fn, tp = cm.ravel()
            
            # Derived metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # ROC and PR curves
            fpr, tpr, _ = roc_curve(self.targets[:, i], predictions[:, i])
            precision_curve, recall_curve, _ = precision_recall_curve(
                self.targets[:, i], predictions[:, i]
            )
            
            per_class_metrics[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'roc_auc': float(auc(fpr, tpr)),
                'average_precision': float(average_precision_score(
                    self.targets[:, i], predictions[:, i]
                )),
                'confusion_matrix': cm.tolist(),
                'roc_curve': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                },
                'pr_curve': {
                    'precision': precision_curve.tolist(),
                    'recall': recall_curve.tolist()
                }
            }
        
        return {
            'overall_metrics': overall_metrics,
            'per_class_metrics': per_class_metrics
        }
    
    def _get_model_weights(self) -> Dict[str, float]:
        """Get current ensemble weights"""
        weights = self.model.ensemble_weights.softmax(dim=0).cpu().numpy()
        return {
            'cnn_weight': float(weights[0]),
            'vit_weight': float(weights[1])
        }
    
    def _generate_visualizations(self):
        """Generate all visualization plots"""
        viz_dir = self.save_dir / 'visualizations'
        
        # Generate various plots
        self._plot_roc_curves(viz_dir)
        self._plot_pr_curves(viz_dir)
        self._plot_confusion_matrices(viz_dir)
        self._plot_model_comparison(viz_dir)
        self._plot_prediction_distributions(viz_dir)
        self._plot_ensemble_weights(viz_dir)
    
    def _plot_roc_curves(self, viz_dir: Path):
        """Plot ROC curves for each class and model"""
        # Create subplots grid
        n_rows = (len(self.class_names) + 3) // 4
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.class_names):
            ax = axes[i]
            
            # Plot ROC curve for each model
            models = {
                'Ensemble': self.ensemble_preds,
                'CNN': self.cnn_preds,
                'ViT': self.vit_preds,
                'Fusion': self.fusion_preds
            }
            
            for model_name, preds in models.items():
                fpr, tpr, _ = roc_curve(self.targets[:, i], preds[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title(f'{class_name}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'roc_curves.png')
        plt.close()
    
    def _plot_pr_curves(self, viz_dir: Path):
        """Plot Precision-Recall curves for each class and model"""
        n_rows = (len(self.class_names) + 3) // 4
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.class_names):
            ax = axes[i]
            
            models = {
                'Ensemble': self.ensemble_preds,
                'CNN': self.cnn_preds,
                'ViT': self.vit_preds,
                'Fusion': self.fusion_preds
            }
            
            for model_name, preds in models.items():
                precision, recall, _ = precision_recall_curve(self.targets[:, i], preds[:, i])
                ap = average_precision_score(self.targets[:, i], preds[:, i])
                ax.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
            
            ax.set_title(f'{class_name}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.legend()
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'pr_curves.png')
        plt.close()
    
    def _plot_confusion_matrices(self, viz_dir: Path):
        """Plot confusion matrices for each class and model"""
        models = {
            'Ensemble': self.ensemble_preds,
            'CNN': self.cnn_preds,
            'ViT': self.vit_preds,
            'Fusion': self.fusion_preds
        }
        
        for model_name, preds in models.items():
            binary_preds = (preds > 0.5).astype(int)
            
            n_rows = (len(self.class_names) + 3) // 4
            fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
            axes = axes.ravel()
            
            for i, class_name in enumerate(self.class_names):
                cm = confusion_matrix(self.targets[:, i], binary_preds[:, i])
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=axes[i],
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive']
                )
                axes[i].set_title(f'{class_name}')
            
            # Remove empty subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            
            plt.suptitle(f'{model_name} Confusion Matrices')
            plt.tight_layout()
            plt.savefig(viz_dir / f'confusion_matrices_{model_name.lower()}.png')
            plt.close()
    
    def _plot_model_comparison(self, viz_dir: Path):
        """Plot model performance comparison"""
        models = {
            'Ensemble': self.ensemble_preds,
            'CNN': self.cnn_preds,
            'ViT': self.vit_preds,
            'Fusion': self.fusion_preds
        }
        
        # Calculate metrics for each model
        metrics_dict = {
            model_name: {
                'MAP': average_precision_score(self.targets, preds, average='samples'),
                'Exact Match': np.mean(np.all((preds > 0.5).astype(int) == self.targets, axis=1)),
                'Hamming Loss': np.mean((preds > 0.5).astype(int) != self.targets)
            }
            for model_name, preds in models.items()
        }
        
        # Create comparison plots
        metrics = ['MAP', 'Exact Match', 'Hamming Loss']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [metrics_dict[model][metric] for model in models.keys()]
            axes[i].bar(models.keys(), values)
            axes[i].set_title(metric)
            axes[i].set_ylim(0, 1)
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'model_comparison.png')
        plt.close()
    
    def _plot_prediction_distributions(self, viz_dir: Path):
        """Plot prediction score distributions"""
        n_rows = (len(self.class_names) + 3) // 4
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.class_names):
            ax = axes[i]
            
            # Get positive and negative samples
            pos_samples = self.ensemble_preds[self.targets[:, i] == 1, i]
            neg_samples = self.ensemble_preds[self.targets[:, i] == 0, i]
            
            # Plot distributions
            if len(pos_samples) > 0:
                ax.hist(pos_samples, bins=50, alpha=0.5, label='Positive', density=True)
            if len(neg_samples) > 0:
                ax.hist(neg_samples, bins=50, alpha=0.5, label='Negative', density=True)
            
            ax.set_title(f'{class_name}')
            ax.legend()
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.suptitle('Prediction Score Distributions')
        plt.tight_layout()
        plt.savefig(viz_dir / 'prediction_distributions.png')
        plt.close()
    
    def _plot_ensemble_weights(self, viz_dir: Path):
        """Plot ensemble weights"""
        weights = self.model.ensemble_weights.softmax(dim=0).cpu().numpy()
        
        plt.figure(figsize=(8, 6))
        plt.bar(['CNN', 'ViT'], weights)
        plt.title('Ensemble Model Weights')
        plt.ylabel('Weight')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(weights):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'ensemble_weights.png')
        plt.close()

    def _save_results(self, results: Dict):
        """Save evaluation results in multiple formats"""
        # Save raw results as JSON
        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save Excel report
        self._save_excel_report(results)
        
        # Save markdown report
        self._save_markdown_report(results)
        
        # Save predictions
        self._save_predictions()
    
    def _save_excel_report(self, results: Dict):
        """Save detailed Excel report with multiple sheets"""
        with pd.ExcelWriter(self.save_dir / 'evaluation_results.xlsx') as writer:
            # Overall metrics comparison sheet
            overall_metrics = {
                'Model': [],
                'Mean AP': [],
                'Exact Match': [],
                'Hamming Loss': []
            }
            
            for model in ['ensemble', 'cnn', 'vit', 'fusion']:
                metrics = results[f'{model}_metrics']['overall_metrics']
                overall_metrics['Model'].append(model.upper())
                overall_metrics['Mean AP'].append(metrics['mean_average_precision'])
                overall_metrics['Exact Match'].append(metrics['exact_match'])
                overall_metrics['Hamming Loss'].append(metrics['hamming_loss'])
            
            pd.DataFrame(overall_metrics).to_excel(writer, sheet_name='Overall Comparison', index=False)
            
            # Per-model per-class metrics sheets
            for model in ['ensemble', 'cnn', 'vit', 'fusion']:
                metrics = results[f'{model}_metrics']['per_class_metrics']
                
                # Create DataFrame for each metric type
                for metric in ['precision', 'recall', 'f1_score', 'average_precision']:
                    metric_data = {
                        class_name: metrics[class_name][metric]
                        for class_name in self.class_names
                    }
                    pd.Series(metric_data, name=metric).to_frame().to_excel(
                        writer,
                        sheet_name=f'{model.upper()} Metrics',
                        startrow=len(self.class_names) * list(metrics[self.class_names[0]].keys()).index(metric)
                    )
            
            # Ensemble weights sheet
            pd.Series(results['model_weights'], name='Weight').to_frame().to_excel(
                writer,
                sheet_name='Ensemble Weights'
            )
    
    def _save_markdown_report(self, results: Dict):
        """Generate and save markdown evaluation report"""
        report = f"# Ensemble Model Evaluation Report\n\n"
        report += f"Generated on: {results['timestamp']}\n\n"
        
        # Model Weights
        report += "## Ensemble Weights\n"
        for model, weight in results['model_weights'].items():
            report += f"- {model}: {weight:.4f}\n"
        
        # Overall Performance Comparison
        report += "\n## Overall Performance Comparison\n\n"
        report += "| Model | Mean AP | Exact Match | Hamming Loss |\n"
        report += "|-------|---------|--------------|---------------|\n"
        
        for model in ['ensemble', 'cnn', 'vit', 'fusion']:
            metrics = results[f'{model}_metrics']['overall_metrics']
            report += (f"| {model.upper()} | "
                      f"{metrics['mean_average_precision']:.4f} | "
                      f"{metrics['exact_match']:.4f} | "
                      f"{metrics['hamming_loss']:.4f} |\n")
        
        # Per-Class Performance
        report += "\n## Per-Class Performance\n\n"
        for class_name in self.class_names:
            report += f"\n### {class_name}\n\n"
            report += "| Model | Precision | Recall | F1 Score | AP |\n"
            report += "|-------|-----------|---------|----------|----|\n"
            
            for model in ['ensemble', 'cnn', 'vit', 'fusion']:
                metrics = results[f'{model}_metrics']['per_class_metrics'][class_name]
                report += (f"| {model.upper()} | "
                          f"{metrics['precision']:.4f} | "
                          f"{metrics['recall']:.4f} | "
                          f"{metrics['f1_score']:.4f} | "
                          f"{metrics['average_precision']:.4f} |\n")
        
        # Save report
        with open(self.save_dir / 'evaluation_report.md', 'w') as f:
            f.write(report)
    
    def _save_predictions(self):
        """Save model predictions for further analysis"""
        predictions = {
            'ensemble': self.ensemble_preds,
            'cnn': self.cnn_preds,
            'vit': self.vit_preds,
            'fusion': self.fusion_preds,
            'targets': self.targets
        }
        
        # Save as numpy arrays
        np.savez(
            self.save_dir / 'predictions' / 'predictions.npz',
            **predictions
        )
        
        # Save as CSV for easier analysis
        for model_name, preds in predictions.items():
            df = pd.DataFrame(
                preds,
                columns=self.class_names
            )
            df.to_csv(self.save_dir / 'predictions' / f'{model_name}_predictions.csv')

def run_evaluation(
    model_path: str,
    config_path: str,
    output_dir: str
) -> Dict:
    """Run complete evaluation pipeline"""
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data module
    from data_preprocessing import DatasetConfig, DataModule, ModelType
    data_config = DatasetConfig(config['data_path'])
    data_module = DataModule(
        data_config=data_config,
        model_type=ModelType.ENSEMBLE,
        batch_size=config['evaluation']['batch_size'],
        num_workers=config['evaluation']['num_workers']
    )
    
    # Create and load model
    from ensemble_model import create_ensemble_model
    model = create_ensemble_model(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator and run evaluation
    evaluator = EnsembleEvaluator(
        model=model,
        data_module=data_module,
        device=device,
        class_names=data_module.datasets[DatasetType.TEST].class_columns,
        save_dir=output_dir
    )
    
    results = evaluator.evaluate()
    logging.info(f"Evaluation completed! Results saved in {output_dir}")
    
    return results

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ensemble model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    results = run_evaluation(
        model_path=args.model_path,
        config_path=args.config,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()