import React, { useState } from 'react';
import ModelVisualizations from './ModelVisualizations';

const ModelSection = ({ title, children }) => (
  <div className="mb-8">
    <h3 className="text-2xl font-bold mb-4" style={{ color: "yellow" }}>{title}</h3>
    {children}
  </div>
);

const Card = ({ className, children }) => (
  <div className={`p-6 rounded-lg border border-yellow-400 ${className}`}>
    {children}
  </div>
);

const ModelsImplemented = () => {
  const [activeTab, setActiveTab] = useState("cnn");
  
  const models = [
    {
      id: "cnn",
      name: "CNN with Spatial Attention",
      description: "A CNN architecture leveraging transfer learning and incorporating spatial attention for improved feature extraction from astronomical imagery.",
      architecture: {
        title: "Architecture Details",
        sections: [
          {
            title: "Feature Extraction",
            points: [
              "ResNet50 backbone pre-trained on ImageNet",
              "Modified final layers for constellation-specific features",
              "Preservation of both low-level and high-level astronomical patterns"
            ]
          },
          {
            title: "Spatial Attention Module",
            points: [
              "Global average pooling for feature aggregation",
              "7×7 convolutional layer for spatial attention weights",
              "Batch normalization and sigmoid activation",
              "Element-wise multiplication for feature refinement"
            ]
          },
          {
            title: "Classification Head",
            points: [
              "Initial dropout (p=0.5) for regularization",
              "Linear transformation: R^d → R^512 with ReLU activation",
              "Batch normalization for training stability",
              "Secondary dropout (p=0.25)",
              "Dimension reduction: R^512 → R^256",
              "Final projection: R^256 → R^16 for constellation classes"
            ]
          }
        ]
      },
      training: {
        title: "Training Protocol",
        points: [
          "Binary cross-entropy loss for multi-label classification",
          "ImageNet weight initialization",
          "End-to-end fine-tuning of all layers",
          "Global average pooling before classification"
        ]
      },
      evaluation: {
        title: "Model Evaluation",
        metrics: {
          "Mean Average Precision (mAP)": "0.8321",
          "Micro F1 Score": "0.7605",
          "Micro-precision": "0.9801",
          "Micro-recall": "0.6213"
        },
        highlights: [
          {
            title: "Perfect Performance",
            constellations: ["Gemini (F1: 1.000, AP: 1.000)", "Scorpius (F1: 1.000, AP: 1.000)", "Bootes (F1: 0.9851, AP: 1.000)"]
          },
          {
            title: "Challenging Cases",
            constellations: ["Cassiopeia (F1: 0.000, AP: 0.703)", "Ursa Major (F1: 0.0833, AP: 0.937)", "Cygnus (F1: 0.6154, AP: 0.791)"]
          }
        ]
      }
    },
    {
      id: "vit",
      name: "Vision Transformer (ViT)",
      description: "A modified Vision Transformer architecture optimized for astronomical imagery processing with patch-based approach and specialized attention mechanisms.",
      architecture: {
        title: "Architecture Details",
        sections: [
          {
            title: "Input Processing",
            points: [
              "Image division into N = (H×W)/P² non-overlapping patches",
              "Patch size P=16 for optimal feature capture",
              "Linear projection to embedding dimension D=768",
              "Learnable position embeddings for spatial information"
            ]
          },
          {
            title: "Transformer Encoder",
            points: [
              "12 transformer blocks with multi-head attention",
              "Layer normalization and MLP blocks",
              "Self-attention mechanism for global feature capture",
              "Residual connections for gradient flow"
            ]
          }
        ]
      },
      evaluation: {
        title: "Model Evaluation",
        metrics: {
          "Exact Match Ratio": "0.953",
          "Hamming Loss": "0.030",
          "Precision": "0.880",
          "Recall": "0.914",
          "F1 Score": "0.923"
        },
        performance_tiers: [
          {
            tier: "Outstanding (AUC > 0.95)",
            constellations: ["Canis Major (0.998)", "Gemini (0.998)", "Sagittarius (0.997)"]
          },
          {
            tier: "Strong (AUC 0.90-0.95)",
            constellations: ["Taurus (0.969)", "Orion (0.921)", "Leo (0.912)"]
          },
          {
            tier: "Moderate (AUC < 0.90)",
            constellations: ["Lyra (0.874)", "Cygnus (0.798)", "Pleiades (0.780)"]
          }
        ]
      }
    },
    {
      id: "ensemble",
      name: "Ensemble Architecture",
      description: "A novel ensemble combining CNNs and Vision Transformers with a dual-path fusion mechanism and learnable weights.",
      architecture: {
        title: "Architecture Details",
        sections: [
          {
            title: "Feature Extraction & Projection",
            points: [
              "CNN backbone produces 2048-dimensional feature vectors",
              "ViT backbone generates 768-dimensional features",
              "Dimension reduction through fully connected layers",
              "Common 512-dimensional representation for fusion"
            ]
          },
          {
            title: "Dual-path Fusion Mechanism",
            points: [
              "Decision-level fusion with learnable weights w = [w₁, w₂]",
              "Feature-level fusion through concatenation",
              "Progressive dimensionality reduction: 1024 → 512 → 256 → N",
              "Adaptive model contribution weights"
            ]
          },
          {
            title: "Enhanced Robustness",
            points: [
              "Dual-path architecture for redundancy",
              "Feature-level fusion for pattern relationships",
              "Decision-level fusion for semantic information",
              "Hierarchical regularization strategy"
            ]
          }
        ]
      },
      training: {
        title: "Training Protocol",
        points: [
          "Three separate optimizers for CNN, ViT, and fusion components",
          "AdamW optimizer with individually tuned learning rates",
          "Gradient clipping for stable training",
          "Combined loss function with configurable weights",
          "Early stopping based on validation loss",
          "Model checkpointing for best validation mAP"
        ]
      },
      evaluation: {
        title: "Model Evaluation",
        metrics: {
          "Mean AP": "0.8308",
          "Exact Match": "0.1880",
          "Hamming Loss": "0.0820",
          "CNN Individual mAP": "0.7739",
          "ViT Individual mAP": "0.7991",
          "Fusion mAP": "0.8666"
        },
        highlights: [
          {
            title: "Component Performance",
            constellations: [
              "Ensemble outperforms individual models by 3-9%",
              "Feature-level fusion shows 5% improvement over decision-level fusion",
              "Adaptive weights demonstrate optimal combining of CNN and ViT strengths"
            ]
          },
          {
            title: "Per-class Performance",
            constellations: [
              "Scorpius (F1: 1.000)",
              "Bootes (F1: 0.985)",
              "Canis Minor (Best improvement over individual models)"
            ]
          }
        ]
      }
    },
    {
      id: "efficientnet",
      name: "EfficientNet Architecture",
      description: "A modified EfficientNet-B0 architecture adapted for multi-label constellation detection, leveraging compound scaling properties.",
      architecture: {
        title: "Architecture Details",
        sections: [
          {
            title: "Base Architecture",
            points: [
              "EfficientNet-B0 backbone with compound scaling",
              "Balanced network depth, width, and resolution scaling",
              "Mobile Inverted Bottleneck Convolution (MBConv) blocks",
              "Squeeze-and-excitation optimization"
            ]
          },
          {
            title: "Classification Head",
            points: [
              "Three-stage reduction pipeline: D → 512 → 256 → N",
              "Batch normalization at each stage",
              "ReLU activation functions",
              "Dropout (p=0.3) for regularization",
              "He initialization for all linear layers"
            ]
          },
          {
            title: "Memory Optimizations",
            points: [
              "Batch size optimization: 32 samples per device",
              "Gradient accumulation when necessary",
              "Efficient memory utilization strategies",
              "Mixed precision training support"
            ]
          }
        ]
      },
      training: {
        title: "Training Protocol",
        points: [
          "AdamW optimizer with weight decay correction",
          "Initial learning rate: η = 10⁻³",
          "Weight decay: λ = 0.01",
          "ReduceLROnPlateau schedule with 0.5 reduction factor",
          "10% warmup period of total steps",
          "Gradient clipping at ||g|| = 1.0"
        ]
      },
      evaluation: {
        title: "Model Performance",
        metrics: {
          "Accuracy": "0.8545",
          "F1 Score": "0.8234",
          "Memory Efficiency": "32% reduction",
          "Training Time": "40% faster than baseline",
          "Model Size": "6.5M parameters"
        },
        performance_tiers: [
          {
            tier: "Computation Efficiency",
            constellations: [
              "40% reduction in FLOPs compared to baseline",
              "32% memory usage reduction",
              "2.3x faster inference time"
            ]
          },
          {
            tier: "Model Effectiveness",
            constellations: [
              "Comparable accuracy to larger models",
              "Better performance on resource-constrained devices",
              "Efficient scaling across different compute budgets"
            ]
          }
        ]
      }
    }
  ];

  return (
    <div style={{ 
      fontFamily: 'SpaceGrotesk-VariableFont_wght',
      backgroundColor: 'black',
      color: 'white',
      width: '100%',
      padding: '20px'
    }}>
      <div style={{ padding: '20px' }}>
        <h1 style={{ 
          color: "yellow", 
          fontSize: "2.5rem", 
          marginBottom: "2rem",
          fontWeight: "bold"
        }}>Models Implemented</h1>
        
        {/* Tabs */}
        <div style={{
          display: 'flex',
          marginBottom: '24px',
          borderBottom: '1px solid yellow',
          // overflowX: 'auto',
          whiteSpace: 'nowrap'
        }}>
          {models.map(model => (
            <button
              key={model.id}
              onClick={() => setActiveTab(model.id)}
              style={{
                padding: '12px 24px',
                color: activeTab === model.id ? 'yellow' : 'white',
                borderBottom: activeTab === model.id ? '2px solid yellow' : 'none',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                fontSize: '1rem',
                fontFamily: 'SpaceGrotesk-VariableFont_wght'
              }}
            >
              {model.name}
            </button>
          ))}
        </div>

        {/* Content */}
        <div>
          {models.map(model => (
            activeTab === model.id && (
              <div key={model.id}>
                <Card className="bg-black">
                  <h2 className="text-2xl mb-4" style={{ color: "yellow" }}>{model.name}</h2>
                  <p style={{ color: "rgb(209, 213, 219)", marginBottom: "24px" }}>{model.description}</p>

                  {/* Architecture Section */}
                  <ModelSection title={model.architecture.title}>
                    <div style={{ 
                      display: 'grid', 
                      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                      gap: '24px'
                    }}>
                      {model.architecture.sections.map((section, idx) => (
                        <Card key={idx} className="bg-gray-900">
                          <h4 className="text-lg mb-4" style={{ color: "yellow" }}>{section.title}</h4>
                          <ul style={{ listStyleType: 'disc', paddingLeft: '20px' }}>
                            {section.points.map((point, pidx) => (
                              <li key={pidx} style={{ color: "rgb(209, 213, 219)", marginBottom: "8px" }}>{point}</li>
                            ))}
                          </ul>
                        </Card>
                      ))}
                    </div>
                  </ModelSection>

                  {/* Training Protocol Section */}
                  {model.training && (
                    <ModelSection title="Training Protocol">
                      <Card className="bg-gray-900">
                        <ul style={{ listStyleType: 'disc', paddingLeft: '20px' }}>
                          {model.training.points.map((point, idx) => (
                            <li key={idx} style={{ color: "rgb(209, 213, 219)", marginBottom: "8px" }}>{point}</li>
                          ))}
                        </ul>
                      </Card>
                    </ModelSection>
                  )}

                  {/* Evaluation Section */}
                  <ModelSection title="Model Evaluation">
                    <div style={{ 
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                      gap: '24px'
                    }}>
                      {/* Metrics */}
                      <Card className="bg-gray-900">
                        <h4 className="text-lg mb-4" style={{ color: "yellow" }}>Performance Metrics</h4>
                        <div style={{ 
                          display: 'grid',
                          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                          gap: '16px'
                        }}>
                          {Object.entries(model.evaluation.metrics).map(([key, value]) => (
                            <div key={key} style={{
                              padding: '16px',
                              backgroundColor: 'black',
                              borderRadius: '8px',
                              border: '1px solid yellow',
                              textAlign: 'center'
                            }}>
                              <p style={{ color: "rgb(156, 163, 175)" }}>{key}</p>
                              <p style={{ color: "yellow", fontSize: "1.5rem", fontWeight: "bold" }}>{value}</p>
                            </div>
                          ))}
                        </div>
                      </Card>

                      {/* Performance Analysis */}
                      <Card className="bg-gray-900">
                        <h4 className="text-lg mb-4" style={{ color: "yellow" }}>Performance Analysis</h4>
                        {model.evaluation.highlights ? (
                          model.evaluation.highlights.map((highlight, idx) => (
                            <div key={idx} style={{ marginBottom: '16px' }}>
                              <h5 style={{ color: "yellow", marginBottom: '8px' }}>{highlight.title}</h5>
                              <ul style={{ listStyleType: 'disc', paddingLeft: '20px' }}>
                                {highlight.constellations.map((const_name, cidx) => (
                                  <li key={cidx} style={{ color: "rgb(209, 213, 219)", marginBottom: "4px" }}>{const_name}</li>
                                ))}
                              </ul>
                            </div>
                          ))
                        ) : (
                          model.evaluation.performance_tiers.map((tier, idx) => (
                            <div key={idx} style={{ marginBottom: '16px' }}>
                              <h5 style={{ color: "yellow", marginBottom: '8px' }}>{tier.tier}</h5>
                              <ul style={{ listStyleType: 'disc', paddingLeft: '20px' }}>
                                {tier.constellations.map((const_name, cidx) => (
                                  <li key={cidx} style={{ color: "rgb(209, 213, 219)", marginBottom: "4px" }}>{const_name}</li>
                                ))}
                              </ul>
                            </div>
                          ))
                        )}
                      </Card>
                    </div>
                  </ModelSection>
                  <ModelVisualizations modelType={model.id} />
                </Card>
              </div>
            )
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelsImplemented;
