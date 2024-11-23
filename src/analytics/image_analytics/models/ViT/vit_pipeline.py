import torch
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / 'pipeline.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_config(config_path: str) -> Dict:
    """Load and validate configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_keys = [
        'data_path', 'model_name', 'num_labels', 'batch_size',
        'learning_rate', 'num_epochs'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    return config

def run_pipeline(config_path: str):
    """Run complete training pipeline"""
    # Load configuration
    config = load_config(config_path)
    
    # Setup paths and logging
    base_path = Path(config['data_path'])
    experiment_name = f"vit_constellation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path('experiments') / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(output_dir)
    logging.info("Starting pipeline execution")
    
    try:
        # 1. Run EDA
        logging.info("Running Exploratory Data Analysis...")
        from EDA import ConstellationEDA
        eda = ConstellationEDA(
            data_dir=str(base_path),
            csv_file='train/_classes.csv'
        )
        eda_report = eda.generate_report()
        
        # Update config with EDA recommendations
        config.update(eda_report['recommendations'])
        
        train_csv = base_path / 'train' / '_classes.csv'
        train_img_dir = base_path / 'train' / 'images'
        val_csv = base_path / 'valid' / '_classes.csv'
        val_img_dir = base_path / 'valid' / 'images'
        test_csv = base_path / 'test' / '_classes.csv'
        test_img_dir = base_path / 'test' / 'images'

        # 2. Setup Data Module
        logging.info("Setting up data preprocessing...")
        from data_preprocessing import DataModule
        data_module = DataModule(
            train_csv=train_csv,
            train_img_dir=train_img_dir,
            val_csv=val_csv,
            val_img_dir=val_img_dir,
            batch_size=config['batch_size'],
            model_name=config['model_name']
        )
        data_module.setup()
        train_loader, val_loader = data_module.get_dataloaders()
        
        # 3. Create Model
        logging.info("Creating model...")
        from vit_model import create_model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model(config)
        
        # 4. Training
        logging.info("Starting training...")
        from vit_trainer import Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config={**config, 'save_dir': str(output_dir / 'checkpoints')},
            device=device
        )
        trainer.train()
        
        # 5. Evaluation
        logging.info("Running evaluation...")
        from evaluation import ModelEvaluator
        
        # Load best model
        best_model = create_model(config)
        checkpoint = torch.load(
            output_dir / 'checkpoints/best_model.pt',
            map_location=device
        )
        best_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create test dataloader
        test_module = DataModule(
            train_csv=str(train_csv),  # Not used but needed for initialization
            train_img_dir=str(train_img_dir),  # Not used but needed for initialization
            val_csv=str(test_csv),
            val_img_dir=str(test_img_dir),
            batch_size=config['batch_size'],
            model_name=config['model_name']
        )
        test_module.setup()
        _, test_loader = test_module.get_dataloaders()
        
        # Run evaluation
        evaluator = ModelEvaluator(
            model=best_model,
            test_loader=test_loader,
            class_names=data_module.get_class_names(),
            device=device,
            output_dir=str(output_dir / 'evaluation')
        )
        evaluation_results = evaluator.evaluate()
        
        logging.info("Pipeline completed successfully")
        logging.info(f"Results saved in {output_dir}")
        
        return {
            'output_dir': output_dir,
            'evaluation_results': evaluation_results,
            'eda_report': eda_report
        }
    
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    config_path = "config.yaml"
    run_pipeline(config_path)