import React from 'react';

const ModelVisualizations = ({ modelType }) => {
  const visualizations = {
    cnn: [
      { src: "./CNN_train_stats.png", caption: "Training and Validation Loss", description: "Visualization of model convergence showing training and validation loss curves over epochs" },
      { src: "./CNN_PR.png", caption: "Precision Recall Curve", description: "Precision-Recall curve demonstrating model performance" },
      { src: "./CNN_ROC_AUC.png", caption: "ROC Curves per Class", description: "ROC curves demonstrating classification performance for each constellation" }
    ],
    vit: [
      { src: "./ViT_metrics.png", caption: "Training Metrics Over Time", description: "Comprehensive view of accuracy, loss, and learning rate throughout training" },
      { src: "./ViT_ROC.png", caption: "ROC Curves", description: "ROC curves showing classification performance for each constellation" },
    ],
    ensemble: [
      { src: "./Ensemble_model_weights.png", caption: "Model Weights Distribution", description: "Distribution of learned weights across ensemble components" },
      { src: "./MAP_Exact_Hamming.png", caption: "MAP and Exact Hamming Loss", description: "Comparison of ensemble performance metrics" },
      { src: "./Ensemble_metrics.png", caption: "Performance Metrics", description: "Comprehensive performance metrics for ensemble predictions" },
      { src: "./Ensemble_metrics_2.png", caption: "Training Metrics", description: "Training metrics showing convergence of ensemble model" },
      { src: "./Ensemble_metrics_3.png", caption: "Validation Metrics", description: "Validation metrics showing convergence of ensemble model" }
    ],
    efficientnet: [
      { src: "./EfficientNet_metrics.png", caption: "Training Metrics", description: "Training metrics showing convergence of EfficientNet model" },
      { src: "./EfficientNet_metrics_2.png", caption: "Validation Metrics", description: "Validation metrics showing convergence of EfficientNet model" },
      { src: "./EfficientPR.png", caption: "Precision Recall Curve", description: "Precision-Recall curve demonstrating model performance" },
      { src: "./ROC.png", caption: "ROC Curves", description: "ROC curves showing classification performance for each constellation" },
    ]
  };

  return (
    <div className="mt-8">
      <h3 className="text-2xl font-bold mb-6" style={{ color: "yellow" }}>Model Visualizations</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {visualizations[modelType].map((viz, index) => (
          <div key={index} className="bg-gray-900 rounded-lg overflow-hidden border border-yellow-400">
            <div className="p-4">
            <h4 className="mt-4 text-lg font-semibold text-yellow-400">{viz.caption}</h4>
            <p className="mt-2 text-gray-300 text-sm">{viz.description}</p>
              <img
                src={viz.src}
                alt={viz.caption}
                className="w-full h-48 object-cover rounded"
              />
              
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModelVisualizations;