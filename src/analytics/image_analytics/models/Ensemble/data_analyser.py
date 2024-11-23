import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Dict
import logging
from collections import defaultdict
import yaml
from data_preprocessing import DatasetConfig, DatasetType
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict

class DataAnalyzer:
    """Analyzer for constellation dataset"""
    
    def __init__(self, data_config: DatasetConfig):
        self.data_config = data_config
        self.splits = [DatasetType.TRAIN, DatasetType.VALID, DatasetType.TEST]
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Load data
        self.data = self._load_data()
        self.class_names = [col for col in self.data[DatasetType.TRAIN].columns if col != 'filename']
    
    def _load_data(self) -> Dict[DatasetType, pd.DataFrame]:
        """Load data for all splits"""
        data = {}
        for split in self.splits:
            img_dir, csv_path = self.data_config.get_split_paths(split)
            if csv_path.exists():
                data[split] = pd.read_csv(csv_path)
                logging.info(f"Loaded {split.value} data: {len(data[split])} samples")
        return data
    
    def analyze(self) -> Dict:
        """Run complete dataset analysis"""
        logging.info("Starting dataset analysis...")
        
        results = {
            'basic_stats': self._analyze_basic_stats(),
            'class_distribution': self._analyze_class_distribution(),
            'label_correlations': self._analyze_label_correlations(),
            'image_properties': self._analyze_image_properties(),
            'recommendations': self._generate_recommendations()
        }
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        logging.info("Analysis completed")
        return results
    
    def _analyze_basic_stats(self) -> Dict:
        """Analyze basic dataset statistics"""
        stats = {
            'total_images': sum(len(df) for df in self.data.values()),
            'num_classes': len(self.class_names),
            'split_sizes': {split.value: len(df) for split, df in self.data.items()}
        }
        
        # Calculate labels per image
        train_labels_per_image = self.data[DatasetType.TRAIN][self.class_names].sum(axis=1)
        stats.update({
            'avg_labels_per_image': float(train_labels_per_image.mean()),
            'min_labels_per_image': int(train_labels_per_image.min()),
            'max_labels_per_image': int(train_labels_per_image.max())
        })
        
        return stats
    
    def _analyze_class_distribution(self) -> Dict:
        """Analyze class distribution and imbalance"""
        distribution = {}
        
        for split, df in self.data.items():
            # Class counts
            class_counts = df[self.class_names].sum()
            
            # Class ratios
            max_count = class_counts.max()
            class_ratios = max_count / class_counts
            
            distribution[split.value] = {
                'class_counts': class_counts.to_dict(),
                'class_ratios': class_ratios.to_dict(),
                'imbalance_ratio': float(max_count / class_counts.min())
            }
        
        return distribution
    
    def _analyze_label_correlations(self) -> Dict:
        """Analyze correlations between labels"""
        train_df = self.data[DatasetType.TRAIN]
        
        # Calculate correlation matrix
        corr_matrix = train_df[self.class_names].corr()
        
        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(self.class_names)):
            for j in range(i+1, len(self.class_names)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:  # Threshold for "high" correlation
                    high_correlations.append({
                        'class1': self.class_names[i],
                        'class2': self.class_names[j],
                        'correlation': float(corr)
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations
        }
    
    def _analyze_image_properties(self) -> Dict:
        """Analyze image properties"""
        properties = defaultdict(list)
        
        # Sample images for analysis
        train_df = self.data[DatasetType.TRAIN]
        sample_size = min(100, len(train_df))
        sample_indices = np.random.choice(len(train_df), sample_size, replace=False)
        
        train_img_dir = self.data_config.splits[DatasetType.TRAIN] / 'images'
        
        for idx in sample_indices:
            img_path = train_img_dir / train_df.iloc[idx]['filename']
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    properties['widths'].append(width)
                    properties['heights'].append(height)
                    properties['aspect_ratios'].append(width/height)
                    properties['formats'].append(img.format)
                    if img.mode == 'RGB':
                        img_array = np.array(img)
                        properties['mean_intensities'].append(img_array.mean(axis=(0,1)).tolist())
                        properties['std_intensities'].append(img_array.std(axis=(0,1)).tolist())
            except Exception as e:
                logging.warning(f"Error processing image {img_path}: {str(e)}")
        
        return {
            'width_stats': {
                'mean': float(np.mean(properties['widths'])),
                'std': float(np.std(properties['widths'])),
                'min': float(np.min(properties['widths'])),
                'max': float(np.max(properties['widths']))
            },
            'height_stats': {
                'mean': float(np.mean(properties['heights'])),
                'std': float(np.std(properties['heights'])),
                'min': float(np.min(properties['heights'])),
                'max': float(np.max(properties['heights']))
            },
            'aspect_ratio_stats': {
                'mean': float(np.mean(properties['aspect_ratios'])),
                'std': float(np.std(properties['aspect_ratios']))
            },
            'format_distribution': dict(pd.Series(properties['formats']).value_counts()),
            'intensity_stats': {
                'mean_rgb': np.mean(properties['mean_intensities'], axis=0).tolist(),
                'std_rgb': np.mean(properties['std_intensities'], axis=0).tolist()
            }
        }
    
    def _generate_recommendations(self) -> Dict:
        """Generate recommendations based on analysis"""
        train_df = self.data[DatasetType.TRAIN]
        class_counts = train_df[self.class_names].sum()
        max_count = class_counts.max()
        
        return {
            'image_size': {
                'target_size': 224,
                'reasoning': "Standard input size compatible with both CNN and ViT architectures"
            },
            'class_weights': {
                name: float(max_count / count) for name, count in class_counts.items()
            },
            'augmentation': {
                'recommended_techniques': [
                    'random_horizontal_flip',
                    'random_rotation',
                    'random_resized_crop',
                    'color_jitter',
                    'gaussian_noise'
                ],
                'intensity': 'moderate' if class_counts.min() > 100 else 'strong'
            }
        }
    
    def _generate_visualizations(self, results: Dict):
        """Generate analysis visualizations"""
        # Create visualizations directory if it doesn't exist
        viz_dir = Path('analysis_visualizations')
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Class Distribution
        plt.figure(figsize=(15, 6))
        train_counts = pd.Series(results['class_distribution'][DatasetType.TRAIN.value]['class_counts'])
        sns.barplot(x=train_counts.index, y=train_counts.values)
        plt.title('Class Distribution in Training Set')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(viz_dir / 'class_distribution.png')
        plt.close()
        
        # 2. Correlation Matrix
        plt.figure(figsize=(12, 10))
        corr_matrix = pd.DataFrame(results['label_correlations']['correlation_matrix'])
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Label Correlation Matrix')
        plt.tight_layout()
        plt.savefig(viz_dir / 'label_correlations.png')
        plt.close()
    
    def save_analysis(self, output_dir: str):
        """Save analysis results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        results = self.analyze()
        
        # Save results as YAML
        with open(output_dir / 'analysis_results.yaml', 'w') as f:
            yaml.dump(results, f)
        
        # Create markdown report
        report = self._create_markdown_report(results)
        with open(output_dir / 'analysis_report.md', 'w') as f:
            f.write(report)
    
    def _create_markdown_report(self, results: Dict) -> str:
        """Create markdown report from analysis results"""
        report = "# Constellation Dataset Analysis Report\n\n"
        
        # Basic Statistics
        report += "## Basic Statistics\n"
        basic_stats = results['basic_stats']
        report += f"- Total Images: {basic_stats['total_images']}\n"
        report += f"- Number of Classes: {basic_stats['num_classes']}\n"
        report += f"- Average Labels per Image: {basic_stats['avg_labels_per_image']:.2f}\n"
        report += "\nSplit Sizes:\n"
        for split, size in basic_stats['split_sizes'].items():
            report += f"- {split}: {size}\n"
        
        # Class Distribution
        report += "\n## Class Distribution\n"
        train_dist = results['class_distribution']['train']
        report += "\nClass counts in training set:\n"
        for class_name, count in train_dist['class_counts'].items():
            report += f"- {class_name}: {count}\n"
        
        # Correlations
        report += "\n## Label Correlations\n"
        high_corr = results['label_correlations']['high_correlations']
        if high_corr:
            report += "\nHighly correlated label pairs:\n"
            for pair in high_corr:
                report += f"- {pair['class1']} & {pair['class2']}: {pair['correlation']:.3f}\n"
        
        # Image Properties
        report += "\n## Image Properties\n"
        img_props = results['image_properties']
        report += f"\nImage dimensions (mean ± std):\n"
        report += f"- Width: {img_props['width_stats']['mean']:.1f} ± {img_props['width_stats']['std']:.1f}\n"
        report += f"- Height: {img_props['height_stats']['mean']:.1f} ± {img_props['height_stats']['std']:.1f}\n"
        
        # Recommendations
        report += "\n## Recommendations\n"
        recs = results['recommendations']
        report += f"\nImage Size: {recs['image_size']['target_size']}x{recs['image_size']['target_size']}\n"
        report += f"Recommended batch size: {recs['batch_size']['recommended']}\n"
        report += "\nRecommended augmentation techniques:\n"
        for tech in recs['augmentation']['recommended_techniques']:
            report += f"- {tech}\n"
        
        return report

def main():
    """Example usage"""
    from data_preprocessing import DatasetConfig
    
    # Setup paths
    data_path = "data/constellation_dataset_1"
    output_dir = "analysis_results"
    
    # Create data config
    data_config = DatasetConfig(data_path)
    
    # Create analyzer and run analysis
    analyzer = DataAnalyzer(data_config)
    analyzer.save_analysis(output_dir)
    print(f"Analysis results saved to {output_dir}")

if __name__ == "__main__":
    main()