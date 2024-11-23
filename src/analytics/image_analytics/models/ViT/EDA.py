import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from typing import Dict

class ConstellationEDA:
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        self.df = pd.read_csv(self.split_dir / '_classes.csv')
        self.img_dir = self.split_dir / 'images'
        self.data_dir = Path(data_dir)
        self.class_columns = [col for col in self.df.columns if col != 'filename']
        
    def analyze_class_distribution(self) -> Dict:
        """Analyze class distribution and imbalance"""
        class_dist = self.df[self.class_columns].sum().sort_values(ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=class_dist.index, y=class_dist.values)
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        
        return {
            'distribution': class_dist.to_dict(),
            'total_samples': len(self.df),
            'avg_labels_per_image': self.df[self.class_columns].sum(axis=1).mean(),
            'max_imbalance_ratio': class_dist.max() / class_dist.min()
        }
    
    def analyze_image_properties(self) -> Dict:
        """Analyze image properties"""
        image_stats = {'widths': [], 'heights': [], 'channels': [], 'sizes': []}
        
        print(f"Data directory: {self.data_dir}")
        print(f"Number of image files in DataFrame: {len(self.df)}")
        
        for idx, row in self.df.iterrows():
            img_path = self.data_dir / 'train' / 'images' / row['filename']
            if not img_path.exists():
                print(f"Image not found: {img_path}")
                continue
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    image_stats['widths'].append(w)
                    image_stats['heights'].append(h)
                    image_stats['channels'].append(len(img.getbands()))
                    image_stats['sizes'].append(w * h)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return {
            'width_stats': {
                'mean': np.mean(image_stats['widths']),
                'std': np.std(image_stats['widths']),
                'min': min(image_stats['widths']),
                'max': max(image_stats['widths'])
            },
            'height_stats': {
                'mean': np.mean(image_stats['heights']),
                'std': np.std(image_stats['heights']),
                'min': min(image_stats['heights']),
                'max': max(image_stats['heights'])
            },
            'channel_stats': {
                'modes': np.unique(image_stats['channels'], return_counts=True)
            }
        }
    
    def analyze_label_correlations(self) -> pd.DataFrame:
        """Analyze correlations between labels"""
        corr_matrix = self.df[self.class_columns].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Label Correlations')
        plt.tight_layout()
        plt.savefig('label_correlations.png')
        
        return corr_matrix
    
    def generate_report(self) -> Dict:
        """Generate comprehensive EDA report"""
        class_stats = self.analyze_class_distribution()
        image_stats = self.analyze_image_properties()
        corr_matrix = self.analyze_label_correlations()
        
        report = {
            'dataset_stats': {
                'total_images': len(self.df),
                'num_classes': len(self.class_columns),
                'class_names': self.class_columns
            },
            'class_distribution': class_stats,
            'image_properties': image_stats,
            'recommendations': {
                'target_size': (224, 224),  # Standard ViT input size
                'augmentations': [
                    'random_resize_crop',
                    'random_horizontal_flip',
                    'color_jitter',
                    'normalize'
                ],
                'class_weights': self._calculate_class_weights()
            }
        }
        
        return report
    
    def _calculate_class_weights(self) -> Dict:
        """Calculate class weights for handling imbalance"""
        class_counts = self.df[self.class_columns].sum()
        total_samples = len(self.df)
        weights = total_samples / (len(self.class_columns) * class_counts)
        return {col: float(weight) for col, weight in zip(self.class_columns, weights)}
    