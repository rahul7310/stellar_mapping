import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import json
import logging
import time
import pandas as pd

class CNNPredictor:
    """Inference class for CNN-based constellation classification"""
    
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
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        # Load class names
        self.class_names = self._load_class_names()
        
        logging.info(f"Initialized CNN model on {device}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self, model_path: str):
        """Load CNN model"""
        from CNN import create_model
        
        # Create model architecture
        model = create_model(
            model_type='cnn',
            num_classes=len(self.class_names),
            backbone=self.config['model']['backbone'],
            dropout_rate=self.config['model']['dropout_rate']
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def _get_transforms(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_class_names(self) -> List[str]:
        """Load constellation class names"""
        df = pd.read_csv(Path(self.config['data_path']) / 'train/_classes.csv')
        return [col for col in df.columns if col != 'filename']
    
    def preprocess_image(
        self,
        image: Union[str, np.ndarray, Path, Image.Image]
    ) -> torch.Tensor:
        """Preprocess image for inference"""
        # Convert to PIL Image if necessary
        if isinstance(image, (str, Path)):
            image = Image.open(str(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")
        
        # Apply transforms
        img_tensor = self.transform(image)
        
        # Add batch dimension if necessary
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
        img_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.model(img_tensor)
        predictions = self._process_outputs(outputs, conf_thresh)
        
        # Create visualization if requested
        visualization = None
        if return_visualization:
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
            visualization = self.visualize_predictions(image, predictions)
        
        return (predictions, visualization) if return_visualization else predictions
    
    def _process_outputs(
        self,
        outputs: torch.Tensor,
        conf_thresh: float
    ) -> Dict:
        """Process model outputs into predictions"""
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
            'raw_probabilities': probs.cpu().numpy().tolist(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def visualize_predictions(
        self,
        image: np.ndarray,
        predictions: Dict
    ) -> np.ndarray:
        """Visualize predictions on image"""
        image_vis = image.copy()
        
        # Add predictions text
        y_offset = 30
        for pred in predictions['predictions']:
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
            batch_tensors = []
            
            # Preprocess batch
            for path in batch_paths:
                tensor = self.preprocess_image(path)
                batch_tensors.append(tensor)
            
            # Stack tensors and run inference
            batch_input = torch.cat(batch_tensors)
            with torch.no_grad():
                outputs = self.model(batch_input)
            
            # Process predictions
            for j, output in enumerate(outputs):
                predictions = self._process_outputs(
                    output.unsqueeze(0),
                    conf_thresh
                )
                predictions['image_path'] = str(batch_paths[j])
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
    """Example usage of CNN predictor"""
    # Initialize predictor
    predictor = CNNPredictor(
        model_path="models/cnn/best_model.pt",
        config_path="models/cnn/config.yaml"
    )
    
    # Test single image prediction
    image_path = "test_images/constellation.jpg"
    predictions, visualization = predictor.predict(
        image_path,
        conf_thresh=0.5,
        return_visualization=True
    )
    
    # Save results
    output_dir = Path("results/cnn")
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