import torch
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import json

from data_preprocessing import DataModule, DatasetType

class EnsemblePipeline:
    """Main pipeline for ensemble model training and evaluation"""
    
    def __init__(
        self,
        config_path: str,
        experiment_name: Optional[str] = None
    ):
        self.config_path = config_path
        self.config = self._load_config()
        self.experiment_name = experiment_name or f"ensemble_{datetime.now():%Y%m%d_%H%M%S}"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
    
    def _load_config(self) -> Dict:
        """Load and validate configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self._validate_config(config)
        return config
    
    def _validate_config(self, config: Dict):
        """Validate configuration"""
        required_keys = [
            'data_path',
            'model',
            'training',
            'evaluation',
            'logging'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate model config
        model_required = ['num_classes', 'cnn_config', 'vit_config']
        for key in model_required:
            if key not in config['model']:
                raise ValueError(f"Missing required model config key: {key}")
    
    def setup_directories(self):
        """Setup experiment directories"""
        base_dir = Path(self.config['logging']['save_dir'])
        self.exp_dir = base_dir / self.experiment_name
        
        # Create subdirectories
        self.dirs = {
            'checkpoints': self.exp_dir / 'checkpoints',
            'evaluation': self.exp_dir / 'evaluation',
            'logs': self.exp_dir / 'logs',
            'visualizations': self.exp_dir / 'visualizations',
            'predictions': self.exp_dir / 'predictions'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.dirs['logs'] / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
    
    def run(self):
        """Run complete pipeline"""
        try:
            logging.info(f"Starting pipeline execution with experiment name: {self.experiment_name}")
            logging.info(f"Using device: {self.device}")
            
            # 1. Data Analysis
            logging.info("Running data analysis...")
            analysis_results = self.run_data_analysis()
            
            # 2. Data Preprocessing
            logging.info("Setting up data module...")
            data_module = self.setup_data_module()
            
            # 3. Create Model
            logging.info("Creating model...")
            model = self.create_model()
            
            # 4. Training
            logging.info("Starting training...")
            best_model_path = self.train_model(model, data_module)
            
            # 5. Evaluation
            logging.info("Running evaluation...")
            eval_results = self.evaluate_model(best_model_path, data_module)
            
            # 6. Save Results
            self.save_pipeline_results({
                'analysis_results': analysis_results,
                'evaluation_results': eval_results
            })
            
            logging.info(f"Pipeline completed successfully! Results saved in {self.exp_dir}")
            return {
                'status': 'success',
                'experiment_dir': str(self.exp_dir),
                'best_model_path': str(best_model_path)
            }
        
        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def run_data_analysis(self) -> Dict:
        """Run data analysis"""
        from data_preprocessing import DatasetConfig
        from data_analyser import DataAnalyzer
        
        data_config = DatasetConfig(self.config['data_path'])
        analyzer = DataAnalyzer(data_config)
        results = analyzer.analyze()
        
        # Save analysis results
        with open(self.dirs['logs'] / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results
    
    def setup_data_module(self) -> 'DataModule':
        """Setup data module"""
        from data_preprocessing import DatasetConfig, DataModule, ModelType
        
        data_config = DatasetConfig(self.config['data_path'])
        data_module = DataModule(
            data_config=data_config,
            model_type=ModelType.ENSEMBLE,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            image_size=self.config['model']['image_size']
        )
        
        return data_module
    
    def create_model(self) -> torch.nn.Module:
        """Create ensemble model"""
        from ensemble_model import create_ensemble_model
        model = create_ensemble_model(self.config['model'])
        return model.to(self.device)
    
    def train_model(
        self,
        model: torch.nn.Module,
        data_module: 'DataModule'
    ) -> Path:
        """Train model and return path to best checkpoint"""
        from ensemble_trainer import EnsembleTrainer
        
        trainer = EnsembleTrainer(
            model=model,
            data_module=data_module,
            config=self.config,
            device=self.device,
            save_dir=self.dirs['checkpoints'],
            experiment_name=self.experiment_name
        )
        
        trainer.train(num_epochs=self.config['training']['num_epochs'])
        return self.dirs['checkpoints'] / 'best_model.pt'
    
    def evaluate_model(
        self,
        model_path: Path,
        data_module: 'DataModule'
    ) -> Dict:
        """Evaluate trained model"""
        from evaluation import EnsembleEvaluator
        
        evaluator = EnsembleEvaluator(
            model_path=model_path,
            data_module=data_module,
            device=self.device,
            class_names=data_module.datasets[DatasetType.TEST].class_columns,
            save_dir=self.dirs['evaluation']
        )
        
        return evaluator.evaluate()
    
    def save_pipeline_results(self, results: Dict):
        """Save final pipeline results"""
        # Save complete results
        with open(self.exp_dir / 'pipeline_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Create summary markdown
        summary = self._create_summary(results)
        with open(self.exp_dir / 'summary.md', 'w') as f:
            f.write(summary)
    
    def _create_summary(self, results: Dict) -> str:
        """Create pipeline summary"""
        summary = f"# Ensemble Model Pipeline Summary\n\n"
        summary += f"Experiment Name: {self.experiment_name}\n"
        summary += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Data Analysis Summary
        summary += "## Data Analysis\n"
        analysis = results['analysis_results']
        summary += f"- Total Images: {analysis['total_images']}\n"
        summary += f"- Number of Classes: {analysis['num_classes']}\n"
        summary += f"- Average Labels per Image: {analysis['avg_labels_per_image']:.2f}\n\n"
        
        # Model Performance
        eval_results = results['evaluation_results']
        summary += "## Model Performance\n"
        summary += "\n### Ensemble Model\n"
        metrics = eval_results['ensemble_metrics']['overall_metrics']
        summary += f"- Mean Average Precision: {metrics['mean_average_precision']:.4f}\n"
        summary += f"- Exact Match Accuracy: {metrics['exact_match']:.4f}\n"
        summary += f"- Hamming Loss: {metrics['hamming_loss']:.4f}\n\n"
        
        # Model Weights
        weights = eval_results['model_weights']
        summary += "### Ensemble Weights\n"
        summary += f"- CNN Weight: {weights['cnn_weight']:.4f}\n"
        summary += f"- ViT Weight: {weights['vit_weight']:.4f}\n\n"
        
        # Directory Structure
        summary += "## Output Directory Structure\n"
        for dir_name, dir_path in self.dirs.items():
            summary += f"- {dir_name}: {dir_path.relative_to(self.exp_dir)}\n"
        
        return summary

def run_pipeline(config_path: str, experiment_name: Optional[str] = None) -> Dict:
    """Run complete pipeline"""
    pipeline = EnsemblePipeline(
        config_path=config_path,
        experiment_name=experiment_name
    )
    return pipeline.run()

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ensemble model pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--name', type=str, help='Experiment name')
    
    args = parser.parse_args()
    
    results = run_pipeline(
        config_path=args.config,
        experiment_name=args.name
    )
    
    print(f"\nPipeline completed successfully!")
    print(f"Results saved in: {results['experiment_dir']}")
    print(f"Best model saved as: {results['best_model_path']}")

if __name__ == "__main__":
    main()