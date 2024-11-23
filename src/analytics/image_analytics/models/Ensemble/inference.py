import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import logging
from typing import Dict, List, Union, Tuple, Optional
import json
import time
import yaml
from tqdm import tqdm
import pandas as pd

from data_preprocessing import ModelType

class EnsemblePredictor:
    """Inference class for ensemble constellation classification"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing
        self._setup_preprocessing()
        
        # Load class names
        self.class_names = self._load_class_names()
        
        logging.info(f"Initialized ensemble model on {device}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load ensemble model"""
        from ensemble_model import create_ensemble_model
        
        # Create model architecture
        model = create_ensemble_model(self.config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def _setup_preprocessing(self):
        """Setup preprocessing components"""
        from data_preprocessing import DataModule, DatasetConfig, ModelType
        
        # Create minimal data module for transforms
        data_config = DatasetConfig(self.config['data_path'])
        self.data_module = DataModule(
            data_config=data_config,
            model_type=ModelType.ENSEMBLE,
            batch_size=1,  # Not used for inference
            image_size=self.config['model']['image_size']
        )
    
    def _load_class_names(self) -> List[str]:
        """Load constellation class names"""
        train_csv = pd.read_csv(
            Path(self.config['data_path']) / 'train/_classes.csv'
        )
        return [col for col in train_csv.columns if col != 'filename']
    
    def preprocess_image(
        self,
        image: Union[str, np.ndarray, Path, Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess image for inference"""
        # Convert to PIL Image if necessary
        if isinstance(image, (str, Path)):
            image = Image.open(str(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")
        
        # Apply CNN transforms
        cnn_transform = self.data_module.transforms[ModelType.CNN]
        cnn_input = cnn_transform(image=np.array(image))['image']
        
        # Apply ViT transforms
        vit_transform = self.data_module.transforms[ModelType.VIT]
        vit_input = vit_transform(image=np.array(image))['image']
        
        # Add batch dimension if necessary
        if cnn_input.ndim == 3:
            cnn_input = cnn_input.unsqueeze(0)
        if vit_input.ndim == 3:
            vit_input = vit_input.unsqueeze(0)
        
        return cnn_input.to(self.device), vit_input.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, np.ndarray, Path, Image.Image],
        conf_thresh: float = 0.5,
        return_all_predictions: bool = False,
        return_visualization: bool = False
    ) -> Union[Dict, Tuple[Dict, np.ndarray]]:
        """Run inference on single image"""
        # Preprocess image
        cnn_input, vit_input = self.preprocess_image(image)
        
        # Run inference
        outputs = self.model(
            cnn_input=cnn_input,
            vit_input=vit_input,
            return_features=False
        )
        
        # Process outputs
        predictions = self._process_outputs(outputs, conf_thresh)
        
        if return_all_predictions:
            predictions.update(self._process_all_outputs(outputs, conf_thresh))
        
        # Create visualization if requested
        visualization = None
        if return_visualization:
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
            visualization = self.visualize_predictions(image, predictions)
        
        return (predictions, visualization) if return_visualization else predictions
    
    def _process_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        conf_thresh: float
    ) -> Dict:
        """Process model outputs into predictions"""
        # Apply sigmoid to ensemble logits
        probs = torch.sigmoid(outputs['ensemble_logits'])[0]
        
        # Get predictions above threshold
        predictions = []
        for idx, (prob, class_name) in enumerate(zip(probs, self.class_names)):
            if prob > conf_thresh:
                predictions.append({
                    'class': class_name,
                    'confidence': float(prob),
                    'binary_prediction': 1
                })
            else:
                predictions.append({
                    'class': class_name,
                    'confidence': float(prob),
                    'binary_prediction': 0
                })
        
        # Get ensemble weights
        weights = self.model.ensemble_weights.softmax(dim=0)
        
        return {
            'predictions': predictions,
            'ensemble_weights': {
                'cnn_weight': float(weights[0]),
                'vit_weight': float(weights[1])
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _process_all_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        conf_thresh: float
    ) -> Dict:
        """Process outputs from all models"""
        model_outputs = {}
        
        for model_name in ['cnn', 'vit', 'fusion']:
            probs = torch.sigmoid(outputs[f'{model_name}_logits'])[0]
            predictions = []
            
            for idx, (prob, class_name) in enumerate(zip(probs, self.class_names)):
                predictions.append({
                    'class': class_name,
                    'confidence': float(prob),
                    'binary_prediction': 1 if prob > conf_thresh else 0
                })
            
            model_outputs[f'{model_name}_predictions'] = predictions
        
        return model_outputs
    
    def visualize_predictions(
        self,
        image: np.ndarray,
        predictions: Dict,
        show_all_models: bool = False
    ) -> np.ndarray:
        """Visualize predictions on image"""
        image_vis = image.copy()
        
        # Add ensemble predictions
        y_offset = 30
        cv2.putText(
            image_vis,
            "Ensemble Predictions:",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        y_offset += 25
        
        for pred in predictions['predictions']:
            if pred['binary_prediction'] == 1:
                text = f"{pred['class']}: {pred['confidence']:.2f}"
                cv2.putText(
                    image_vis,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                y_offset += 25
        
        # Add model weights
        y_offset += 10
        weights = predictions['ensemble_weights']
        cv2.putText(
            image_vis,
            f"CNN Weight: {weights['cnn_weight']:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )
        y_offset += 25
        cv2.putText(
            image_vis,
            f"ViT Weight: {weights['vit_weight']:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )
        
        # Add predictions from other models if requested
        if show_all_models and 'cnn_predictions' in predictions:
            for model in ['cnn', 'vit', 'fusion']:
                y_offset += 35
                cv2.putText(
                    image_vis,
                    f"{model.upper()} Predictions:",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                y_offset += 25
                
                for pred in predictions[f'{model}_predictions']:
                    if pred['binary_prediction'] == 1:
                        text = f"{pred['class']}: {pred['confidence']:.2f}"
                        cv2.putText(
                            image_vis,
                            text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2
                        )
                        y_offset += 25
        
        return image_vis
    
    def process_batch(
        self,
        image_paths: List[str],
        conf_thresh: float = 0.5,
        batch_size: int = 32
    ) -> List[Dict]:
        """Process a batch of images"""
        all_predictions = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Process each image
            for path in tqdm(batch_paths, desc=f"Processing batch {i//batch_size + 1}"):
                predictions = self.predict(path, conf_thresh=conf_thresh)
                predictions['image_path'] = str(path)
                all_predictions.append(predictions)
        
        return all_predictions
    
    def process_video(
        self,
        video_path: str,
        output_path: str,
        conf_thresh: float = 0.5,
        skip_frames: int = 0
    ) -> Dict:
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        processing_times = []
        all_predictions = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if skip_frames > 0 and frame_count % skip_frames != 0:
                    continue
                
                # Process frame
                start_time = time.time()
                predictions, frame_vis = self.predict(
                    frame,
                    conf_thresh=conf_thresh,
                    return_visualization=True
                )
                processing_time = time.time() - start_time
                
                # Track results
                processing_times.append(processing_time)
                all_predictions.append(predictions)
                
                # Write frame
                out.write(frame_vis)
                
                # Print progress
                if frame_count % 100 == 0:
                    avg_time = np.mean(processing_times[-100:])
                    current_fps = 1.0 / avg_time
                    print(f"Processed {frame_count} frames, "
                          f"Average time: {avg_time:.3f}s, "
                          f"FPS: {current_fps:.2f}")
        
        finally:
            cap.release()
            out.release()
        
        return {
            'frames_processed': frame_count,
            'average_processing_time': np.mean(processing_times),
            'fps': 1.0 / np.mean(processing_times),
            'predictions': all_predictions
        }

def main():
    """Example usage of ensemble predictor"""
    # Initialize predictor
    predictor = EnsemblePredictor(
        model_path="models/ensemble/best_model.pt",
        config_path="config/ensemble_config.yaml"
    )
    
    # Test single image prediction
    image_path = "test_images/constellation.jpg"
    predictions, visualization = predictor.predict(
        image_path,
        conf_thresh=0.5,
        return_all_predictions=True,
        return_visualization=True
    )
    
    # Save results
    output_dir = Path("results/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(
        str(output_dir / "prediction.jpg"),
        visualization
    )
    
    with open(output_dir / "predictions.json", 'w') as f:
        json.dump(predictions, f, indent=4)
    
    # Test batch processing
    image_paths = list(Path("test_images").glob("*.jpg"))
    batch_predictions = predictor.process_batch(
        image_paths,
        conf_thresh=0.5,
        batch_size=32
    )
    
    with open(output_dir / "batch_predictions.json", 'w') as f:
        json.dump(batch_predictions, f, indent=4)
    
    # Test video processing
    video_results = predictor.process_video(
        video_path="test_videos/night_sky.mp4",
        output_path=str(output_dir / "output_video.mp4"),
        conf_thresh=0.5,
        skip_frames=2
    )
    
    with open(output_dir / "video_results.json", 'w') as f:
        json.dump(video_results, f, indent=4)

if __name__ == "__main__":
    main()