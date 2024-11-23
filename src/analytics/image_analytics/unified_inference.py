import json
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Union
import yaml
from ultralytics import YOLO
import logging
from datetime import datetime
import pandas as pd

class ConstellationPredictor:
    """Unified inference class for all constellation detection models"""
    
    def __init__(
        self,
        model_type: str,  # 'cnn', 'vit', 'ensemble', or 'yolo'
        model_path: str,
        config_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model_type = model_type.lower()
        self.device = device
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        # Load class names
        self.class_names = self._load_class_names()
        
        logging.info(f"Initialized {model_type} model on {device}")
    
    def _load_model(self, model_path: str):
        """Load appropriate model based on type"""
        if self.model_type == 'yolo':
            return YOLO(model_path)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if self.model_type == 'cnn':
            from models.CNN.CNN import create_model
            model = create_model(
                model_type='cnn',
                num_classes=self.config['num_classes'],
                backbone=self.config['model']['backbone']
            )
        
        elif self.model_type == 'vit':
            from models.ViT.vit_model import create_model
            model = create_model(self.config)
        
        elif self.model_type == 'ensemble':
            from models.Ensemble.ensemble_model import create_ensemble_model
            model = create_ensemble_model(self.config)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def _get_transforms(self):
        """Get appropriate transforms based on model type"""
        if self.model_type == 'yolo':
            return None  # YOLO handles its own preprocessing
        
        # Standard transforms for CNN/ViT/Ensemble
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_class_names(self) -> List[str]:
        """Load class names based on model type"""
        if self.model_type == 'yolo':
            with open(self.config['data_path'] + '/dataset.yaml', 'r') as f:
                dataset_config = yaml.safe_load(f)
                return dataset_config['names']
        else:
            # For CNN/ViT/Ensemble
            df = pd.read_csv(Path(self.config['data_path']) / 'train/_classes.csv')
            return [col for col in df.columns if col != 'filename']
    
    def preprocess_image(
        self,
        image: Union[str, np.ndarray, Path, Image.Image]
    ) -> torch.Tensor:
        """Preprocess image for inference"""
        if self.model_type == 'yolo':
            # YOLO handles preprocessing internally
            return image
        
        # Convert to PIL Image if necessary
        if isinstance(image, (str, Path)):
            image = Image.open(str(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply transforms
        img_tensor = self.transform(image)
        
        # Add batch dimension
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, np.ndarray, Path, Image.Image],
        conf_thresh: float = 0.5,
        return_visualization: bool = False
    ) -> Union[Dict, Tuple[Dict, np.ndarray]]:
        """Run inference on image"""
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        if self.model_type == 'yolo':
            # YOLO inference
            results = self.model(processed_image, conf=conf_thresh)
            predictions = self._process_yolo_results(results[0])
        else:
            # CNN/ViT/Ensemble inference
            outputs = self.model(processed_image)
            predictions = self._process_classification_results(outputs, conf_thresh)
        
        # Create visualization if requested
        visualization = None
        if return_visualization:
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
            visualization = self.visualize_predictions(image, predictions)
        
        return (predictions, visualization) if return_visualization else predictions
    
    def _process_yolo_results(self, results) -> Dict:
        """Process YOLO detection results"""
        detections = []
        
        for box in results.boxes:
            if box.conf.item() > self.config['model']['conf_thresh']:
                detection = {
                    'class': self.class_names[int(box.cls.item())],
                    'confidence': float(box.conf.item()),
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
        
        return {
            'detections': detections,
            'model_type': 'yolo',
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_classification_results(
        self,
        outputs: torch.Tensor,
        conf_thresh: float
    ) -> Dict:
        """Process classification model results"""
        # Apply sigmoid for multi-label output
        probs = torch.sigmoid(outputs)[0]
        
        # Get predictions above threshold
        predictions = []
        for idx, (prob, class_name) in enumerate(zip(probs, self.class_names)):
            if prob > conf_thresh:
                predictions.append({
                    'class': class_name,
                    'confidence': float(prob)
                })
        
        return {
            'predictions': predictions,
            'model_type': self.model_type,
            'raw_probabilities': probs.cpu().numpy().tolist(),
            'timestamp': datetime.now().isoformat()
        }
    
    def visualize_predictions(
        self,
        image: np.ndarray,
        predictions: Dict
    ) -> np.ndarray:
        """Visualize predictions on image"""
        image_vis = image.copy()
        
        if predictions['model_type'] == 'yolo':
            # Visualize YOLO detections
            for detection in predictions['detections']:
                bbox = detection['bbox']
                label = f"{detection['class']}: {detection['confidence']:.2f}"
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image_vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
        else:
            # Visualize classification predictions
            y_offset = 30
            for pred in predictions['predictions']:
                label = f"{pred['class']}: {pred['confidence']:.2f}"
                cv2.putText(
                    image_vis, label, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                y_offset += 20
        
        return image_vis
    
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
                    fps = 1.0 / avg_time
                    print(f"Processed {frame_count} frames, "
                          f"Average time: {avg_time:.3f}s, "
                          f"FPS: {fps:.2f}")
        
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
    """Example usage for all model types"""
    # Paths
    base_path = Path("models")
    
    # Test images
    image_path = "test_images/constellation.jpg"
    video_path = "test_videos/night_sky.mp4"
    
    # Test all model types
    model_configs = [
        {
            'type': 'cnn',
            'model_path': base_path / 'cnn/best_model.pt',
            'config_path': base_path / 'cnn/config.yaml'
        },
        {
            'type': 'vit',
            'model_path': base_path / 'vit/best_model.pt',
            'config_path': base_path / 'vit/config.yaml'
        },
        {
            'type': 'ensemble',
            'model_path': base_path / 'ensemble/best_model.pt',
            'config_path': base_path / 'ensemble/config.yaml'
        },
        {
            'type': 'yolo',
            'model_path': base_path / 'yolo/best.pt',
            'config_path': base_path / 'yolo/config.yaml'
        }
    ]
    
    for config in model_configs:
        print(f"\nTesting {config['type'].upper()} model:")
        
        # Initialize predictor
        predictor = ConstellationPredictor(
            model_type=config['type'],
            model_path=config['model_path'],
            config_path=config['config_path']
        )
        
        # Test image prediction
        predictions, visualization = predictor.predict(
            image_path,
            conf_thresh=0.5,
            return_visualization=True
        )
        
        # Save results
        output_dir = Path(f"results/{config['type']}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(
            str(output_dir / "prediction.jpg"),
            visualization
        )
        
        with open(output_dir / "predictions.json", 'w') as f:
            json.dump(predictions, f, indent=4)
        
        # Test video processing
        video_results = predictor.process_video(
            video_path=video_path,
            output_path=str(output_dir / "output_video.mp4"),
            conf_thresh=0.5,
            skip_frames=2
        )
        
        print(f"Video processing results for {config['type']}:")
        print(f"Average FPS: {video_results['fps']:.2f}")
        print(f"Frames processed: {video_results['frames_processed']}")

if __name__ == "__main__":
    main()