import torch
import logging
from pathlib import Path
import yaml
from datetime import datetime
from typing import Dict

from data_preprocessing import DatasetType

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file with type conversion"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert numeric values to proper types
    training = config.get('training', {})
    training['learning_rate'] = float(training.get('learning_rate', 1e-4))
    training['weight_decay'] = float(training.get('weight_decay', 1e-5))
    training['min_lr'] = float(training.get('min_lr', 1e-6))
    training['max_lr'] = float(training.get('max_lr', 1e-3))
    training['warmup_start_lr'] = float(training.get('warmup_start_lr', 1e-7))
    training['plateau_factor'] = float(training.get('plateau_factor', 0.5))
    
    return config

def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / 'pipeline.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_pipeline():
    """Main pipeline execution function"""
    # Load configuration
    config_path = r'src\analytics\image_analytics\models\CNN\config.yaml'
    config = load_config(config_path)
    
    # Setup paths
    base_path = Path(config['data_path'])
    experiment_name = f"constellation_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(config['log_dir']) / experiment_name
    
    # Setup logging
    setup_logging(log_dir)
    logging.info("Starting pipeline execution")
    
    try:
        # 1. Data Analysis
        logging.info("Running data analysis...")
        from data_analysis import analyze_dataset, print_preprocessing_recommendations
        analysis_results = analyze_dataset(
            csv_file=base_path / "train/_classes.csv",
            img_dir=base_path / "train/images"
        )
        print_preprocessing_recommendations(analysis_results)
        
        # 2. Data Preprocessing
        logging.info("Setting up data module...")
        from data_preprocessing import DatasetConfig, DataModule, ModelType
        data_config = DatasetConfig(base_path)
        data_module = DataModule(
            data_config=data_config,
            model_type=ModelType.CNN,
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers']
        )
        
        # 3. Model Creation
        logging.info("Creating model...")
        from CNN import create_model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model(
            model_type='cnn',
            num_classes=16,
            pretrained=True,
            backbone=config['model']['backbone'],
            dropout_rate=config['model']['dropout_rate']
        )
        model = model.to(device)
        
        # 4. Training
        logging.info("Starting training...")
        from training import Trainer
        trainer = Trainer(
            model=model,
            data_module=data_module,
            config=config['training'],
            device=device,
            experiment_name=experiment_name,
            use_wandb=config['logging']['use_wandb']
        )
        trainer.train(num_epochs=config['training']['num_epochs'])
        
        # 5. Visualization and Analysis
        logging.info("Running visualization pipeline...")
        from vis_analysis_pipeline import VisualizationManager
        viz_manager = VisualizationManager(
            save_dir=log_dir / 'visualizations',
            class_names=data_module.datasets[DatasetType.TRAIN].class_columns
        )
        
        # Get validation predictions for visualization
        val_loader = data_module.get_dataloader(DatasetType.VALID)
        all_preds = []
        all_targets = []
        all_images = []
        
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                outputs = model(images)
                predictions = torch.sigmoid(outputs)
                all_preds.append(predictions.cpu())
                all_targets.append(targets)
                all_images.append(images.cpu())
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_images = torch.cat(all_images)
        
        # Create visualizations
        viz_manager.visualize_predictions(all_images[:16], all_preds[:16], all_targets[:16])
        viz_manager.plot_roc_curves(all_preds.numpy(), all_targets.numpy())
        viz_manager.plot_precision_recall_curves(all_preds.numpy(), all_targets.numpy())
        
        logging.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_pipeline()