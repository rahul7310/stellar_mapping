import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from torchvision import transforms
from CNN import ConstellationDataset

def analyze_dataset(csv_file, img_dir): 
    """
    Perform comprehensive analysis of the constellation dataset
    """
    # Read the dataset
    df = pd.read_csv(csv_file)
    
    # Get class columns (all except filename)
    class_columns = [col for col in df.columns if col != 'filename']
    
    # 1. Basic Dataset Statistics
    print("\n=== Dataset Statistics ===")
    total_images = len(df)
    total_labels = df[class_columns].sum().sum()
    avg_labels_per_image = df[class_columns].sum(axis=1).mean()
    
    print(f"Total number of images: {total_images}")
    print(f"Total number of labels: {int(total_labels)}")
    print(f"Average labels per image: {avg_labels_per_image:.2f}")
    
    # 2. Class Distribution
    class_distribution = df[class_columns].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(15, 6))
    sns.barplot(x=class_distribution.index, y=class_distribution.values)
    plt.title('Class Distribution in Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Occurrences')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()
    
    # 3. Calculate class imbalance ratios
    print("\n=== Class Imbalance Analysis ===")
    max_class_count = class_distribution.max()
    class_ratios = max_class_count / class_distribution
    print("\nClass imbalance ratios (relative to most frequent class):")
    for class_name, ratio in class_ratios.items():
        print(f"{class_name}: {ratio:.2f}:1")
    
    # 4. Co-occurrence Analysis
    cooccurrence = df[class_columns].T.dot(df[class_columns])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cooccurrence, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Class Co-occurrence Matrix')
    plt.tight_layout()
    plt.savefig('cooccurrence_matrix.png')
    plt.close()
    
    # 5. Multi-label Statistics
    labels_per_image = df[class_columns].sum(axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=labels_per_image, bins=range(int(labels_per_image.max()) + 2))
    plt.title('Distribution of Labels per Image')
    plt.xlabel('Number of Labels')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('labels_per_image_dist.png')
    plt.close()
    
    # 6. Sample Image Analysis
    dataset = ConstellationDataset(csv_file, img_dir, transform=None)
    
    # Analyze image properties
    print("\n=== Image Properties Analysis ===")
    widths = []
    heights = []
    aspects = []
    
    for idx in range(min(100, len(dataset))):  # Sample first 100 images
        img_path = os.path.join(img_dir, df.iloc[idx]['filename'])
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspects.append(w/h)
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            continue
    
    print(f"\nImage Dimensions Summary:")
    print(f"Width  - Mean: {np.mean(widths):.1f}, Min: {min(widths)}, Max: {max(widths)}")
    print(f"Height - Mean: {np.mean(heights):.1f}, Min: {min(heights)}, Max: {max(heights)}")
    print(f"Aspect Ratio - Mean: {np.mean(aspects):.2f}, Min: {min(aspects):.2f}, Max: {max(aspects):.2f}")
    
    # 7. Visualize Augmentation Effects
    if len(dataset) > 0:
        img, _ = dataset[0]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Original
        axes[0].imshow(img)
        axes[0].set_title('Original')
        
        # Apply different augmentations
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomResizedCrop(224),
            transforms.RandomVerticalFlip(p=1)
        ]
        
        for i, transform in enumerate(aug_transforms, 1):
            aug_img = transform(img)
            axes[i].imshow(aug_img)
            axes[i].set_title(f'Augmentation {i}')
        
        plt.tight_layout()
        plt.savefig('augmentation_examples.png')
        plt.close()
    
    return {
        'total_images': total_images,
        'class_distribution': class_distribution,
        'class_ratios': class_ratios,
        'avg_labels_per_image': avg_labels_per_image
    }

def print_preprocessing_recommendations(analysis_results):
    """
    Print recommendations based on the analysis
    """
    print("\n=== Preprocessing Recommendations ===")
    
    # Class imbalance recommendations
    max_ratio = analysis_results['class_ratios'].max()
    if max_ratio > 10:
        print("\n1. Class Imbalance Handling:")
        print("- Implement weighted loss function")
        print("- Consider oversampling minority classes")
        print("- Use class-specific data augmentation")
    
    # Multi-label specific recommendations
    if analysis_results['avg_labels_per_image'] > 1:
        print("\n2. Multi-label Specific:")
        print("- Use BCEWithLogitsLoss for training")
        print("- Consider label correlation in loss function")
        print("- Implement proper multi-label evaluation metrics")
    
    # Data augmentation recommendations
    print("\n3. Recommended Augmentation Techniques:")
    print("- Random horizontal flips (already implemented)")
    print("- Random rotation (±10 degrees)")
    print("- Color jittering")
    print("- Random crops")
    print("- Consider adding Gaussian noise")
    
    # General preprocessing recommendations
    print("\n4. General Preprocessing:")
    print("- Normalize images using ImageNet statistics")
    print("- Resize images to consistent size (224x224)")
    print("- Consider using center crop for validation")
