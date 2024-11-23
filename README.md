# Stellar_Mapping: Constellation Detection & Classification

An interactive web application for detecting and classifying constellations in astronomical images using deep learning.

## Overview
This project aims to develop an accurate constellation detection system using various deep learning architectures including CNN with spatial attention, Vision Transformers (ViT), and ensemble methods. The application features interactive 3D visualizations and comprehensive data exploration tools.

## Features
- Comprehensive data exploration interface
- Multiple deep learning models for constellation detection
- Detailed visualization of astronomical data
- Real-time constellation classification ( coming soon ! )

## Team Members
1. Rahul Prasanna - Computing and Applied Math, Software Engineering expertise
2. Sivakumar Ramakrishnan - Deep Learning Researcher, NLP and Computer Vision specialist
3. Sai Nandini Tata - AI Developer, Machine Learning and Data Science expert

## Project Setup

### Prerequisites
- Node.js (v14 or higher)
- npm (v6 or higher)
- Modern web browser with WebGL support

### Installation
1. Clone the repository
```bash
git clone https://github.com/rahul7310/stellar_mapping.git
cd stellar_mapping
```

2. Install dependencies
```bash
npm install
```

3. Start the development server
```bash
npm run dev
```

4. Open your browser and navigate to `http://localhost:5173` (or the port shown in your terminal)

## Datasets
The project uses multiple datasets that are available through Google Drive due to their size:

1. Roboflow Constellation Images (2000+ labeled images)
   - [Download Link](https://drive.google.com/file/d/1OhX9GxI18xPdX0BZKs0R3MphjSFVA2UA/view?usp=sharing)
   - Contains annotated constellation images with star labels
   - Various quality and resolution samples

2. Stellarium Images
   - [Download Link](https://drive.google.com/file/d/1Gt02x2PDeaprRq5HHhDSVeTJIE7iidoJ/view?usp=drive_link)
   - High-quality constellation mappings
   - Generated using custom Stellarium scripts

## Technologies Used
- React + Vite for frontend development
- Three.js for 3D visualizations
- TensorFlow/PyTorch for model development
- React Three Fiber for 3D rendering
- Tailwind CSS for styling

## Project Structure
```
stellar_mapping/
├── src/
│   ├── Components/       # React components
│   ├── Experiences/     # 3D visualization components
│   ├── Models/          # 3D model components
│   └── assets/          # Static assets
├── public/
│   ├── models/          # 3D model files
│   ├── textures/        # Texture files
│   └── icons/           # Icon files
└── datasets/            # Dataset samples and documentation
```

## Models Implemented
1. CNN with Spatial Attention
2. Vision Transformer (ViT)
3. Ensemble Architecture (CNN + ViT)
4. EfficientNet Architecture

**Trained Models and Evaluation link** 
- [link](https://drive.google.com/drive/folders/1wNStLgkJYmBll8tsK3iv_dr7QcVyNklY?usp=sharing) 
*Note: Only CU Boulder staff and students can access the link for now


## Attributions
- Moon 3D Model: Poly by Google [CC-BY](https://creativecommons.org/licenses/by/3.0/) via [Poly Pizza](https://poly.pizza/m/9OPocAqXM0u)
- Constellation Dataset: [Roboflow Universe](https://universe.roboflow.com/ws-qwbuh/constellation-dsphi)
- Stellarium Software: For constellation mapping generation

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact
For any questions or collaboration opportunities, please reach out to team members through their respective profiles:
- [Rahul Prasanna](https://github.com/rahul7310)
- [Sivakumar Ramakrishnan](https://github.com/arsive02)
- [Sai Nandini Tata](https://github.com/nandinitata)
