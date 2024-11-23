import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import plotly.graph_objects as go

class ResultsAnalyzer:
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.results_dir = self.experiment_dir / 'analysis'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load evaluation results
        with open(self.experiment_dir / 'evaluation/evaluation_results.json', 'r') as f:
            self.evaluation_results = json.load(f)
    
    def analyze_class_performance(self) -> pd.DataFrame:
        """Analyze per-class performance metrics"""
        per_class_metrics = self.evaluation_results['per_class_metrics']
        metrics_df = pd.DataFrame(per_class_metrics).T
        
        # Sort by F1 score
        metrics_df = metrics_df.sort_values('f1_score', ascending=False)
        
        # Save to CSV
        metrics_df.to_csv(self.results_dir / 'class_performance.csv')
        
        # Create visualization
        fig = go.Figure()
        
        for metric in ['precision', 'recall', 'f1_score', 'auc']:
            fig.add_trace(go.Bar(
                name=metric.upper(),
                x=metrics_df.index,
                y=metrics_df[metric],
                text=metrics_df[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Per-Class Performance Metrics',
            barmode='group',
            xaxis_tickangle=-45,
            height=600
        )
        
        fig.write_html(self.results_dir / 'class_performance.html')
        return metrics_df
    
    def analyze_thresholds(self) -> pd.DataFrame:
        """Analyze optimal thresholds"""
        thresholds = self.evaluation_results['thresholds']
        thresholds_df = pd.DataFrame.from_dict(thresholds, orient='index', columns=['threshold'])
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        sns.barplot(data=thresholds_df, x=thresholds_df.index, y='threshold')
        plt.title('Optimal Classification Thresholds by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'thresholds.png')
        plt.close()
        
        return thresholds_df
    
    def analyze_confusion_matrices(self):
        """Analyze class-wise confusion matrices"""
        # Load and process confusion matrices
        with open(self.experiment_dir / 'evaluation/confusion_matrices.json', 'r') as f:
            confusion_matrices = json.load(f)
        
        # Calculate derived metrics
        derived_metrics = {}
        for class_name, cm in confusion_matrices.items():
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            
            # Calculate metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            derived_metrics[class_name] = {
                'specificity': specificity,
                'npv': npv
            }
        
        # Save derived metrics
        with open(self.results_dir / 'derived_metrics.json', 'w') as f:
            json.dump(derived_metrics, f, indent=4)
        
        return derived_metrics
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        overall_metrics = self.evaluation_results['overall_metrics']
        
        # Create summary markdown
        summary = f"""# Model Evaluation Summary

## Overall Performance
- Exact Match Accuracy: {overall_metrics['exact_match']:.4f}
- Hamming Loss: {overall_metrics['hamming_loss']:.4f}
- Average Precision: {overall_metrics['precision']:.4f}
- Average Recall: {overall_metrics['recall']:.4f}
- Average F1 Score: {overall_metrics['f1_score']:.4f}

## Class Performance Analysis
"""
        
        # Add class performance details
        class_metrics = pd.DataFrame(self.evaluation_results['per_class_metrics']).T
        class_metrics = class_metrics.sort_values('f1_score', ascending=False)
        
        summary += "\n### Top Performing Classes\n"
        for idx, (class_name, metrics) in enumerate(class_metrics.head().iterrows()):
            summary += f"{idx+1}. {class_name}\n"
            summary += f"   - F1 Score: {metrics['f1_score']:.4f}\n"
            summary += f"   - AUC: {metrics['auc']:.4f}\n"
        
        summary += "\n### Classes Needing Improvement\n"
        for idx, (class_name, metrics) in enumerate(class_metrics.tail().iterrows()):
            summary += f"{idx+1}. {class_name}\n"
            summary += f"   - F1 Score: {metrics['f1_score']:.4f}\n"
            summary += f"   - AUC: {metrics['auc']:.4f}\n"
        
        # Save summary
        with open(self.results_dir / 'summary_report.md', 'w') as f:
            f.write(summary)
    
    def run_complete_analysis(self):
        """Run all analyses"""
        # Analyze class performance
        class_performance = self.analyze_class_performance()
        
        # Analyze thresholds
        thresholds = self.analyze_thresholds()
        
        # Analyze confusion matrices
        derived_metrics = self.analyze_confusion_matrices()
        
        # Generate summary report
        self.generate_summary_report()
        
        return {
            'class_performance': class_performance,
            'thresholds': thresholds,
            'derived_metrics': derived_metrics
        }

def main():
    # Example usage
    experiment_dir = "experiments/vit_constellation_latest"
    analyzer = ResultsAnalyzer(experiment_dir)
    results = analyzer.run_complete_analysis()
    print(f"Analysis completed. Results saved in {analyzer.results_dir}")
