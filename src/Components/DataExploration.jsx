// import { IpynbRenderer } from "react-ipynb-renderer";
// import { IpynbRenderer } from "react-ipynb-renderer";

// import "react-ipynb-renderer/dist/styles/monokai.css";
// import ipynb from "../analytics/image_analytics/data_preprocessing.json";
// import ipynb from "../../datasets/constellation_templates/"
import { ImageGallery } from "react-image-grid-gallery";

import Andromeda from "../../datasets/constellation_templates/Andromeda.jpg";
import Aries from "../../datasets/constellation_templates/Aries.jpg";
import Bootes from "../../datasets/constellation_templates/Bootes.jpg";
import Gemini from "../../datasets/constellation_templates/Gemini.jpg";
import Leo from "../../datasets/constellation_templates/Leo.jpg";
import Orion from "../../datasets/constellation_templates/Orion.jpg";

import Andromeda_stellarium from "../../datasets/stellarium_constellations/sample/Andromeda001.png";
import Aries_stellarium from "../../datasets/stellarium_constellations/sample/Aries001.png";
import Bootes_stellarium from "../../datasets/stellarium_constellations/sample/Bootes001.png";
import Gemini_stellarium from "../../datasets/stellarium_constellations/sample/Gemini001.png";
import Leo_stellarium from "../../datasets/stellarium_constellations/sample/Leo001.png";
import Orion_stellarium from "../../datasets/stellarium_constellations/sample/Orion001.png";

import img1 from "../../datasets/roboflow_constellation_images/sample/2022-01-08-00-00-00-n_png_jpg.rf.5ad2e498f3937e5fc7f746ff4954afca.jpg";
import img2 from "../../datasets/roboflow_constellation_images/sample/2022-01-10-00-00-00-s_png_jpg.rf.9949ec204bd5b8af72cf2cb22a3402b0.jpg";
import img3 from "../../datasets/roboflow_constellation_images/sample/2022-01-15-00-00-00-s_png_jpg.rf.1c04821ef762c0af4d5bf8772e6ac08d.jpg";
import img4 from "../../datasets/roboflow_constellation_images/sample/2022-01-22-00-00-00-s_png_jpg.rf.37e31378be869cbc5da12083a9452f1c.jpg";
import img5 from "../../datasets/roboflow_constellation_images/sample/2022-01-30-00-00-00-s_png_jpg.rf.b5eee58c97e33ef80b95c30a8aad3d23.jpg";
import img6 from "../../datasets/roboflow_constellation_images/sample/2022-01-30-00-00-00-s_png_jpg.rf.b5eee58c97e33ef80b95c30a8aad3d23.jpg";

import React from 'react';
import styled from "styled-components";

// Custom Card Components
const Card = ({ children, className = "" }) => (
  <div className={`bg-gray-900 border border-yellow-400 rounded-lg overflow-hidden ${className}`}>
    {children}
  </div>
);

const CardHeader = ({ children, className = "" }) => (
  <div className={`p-4 border-b border-yellow-400 text-yellow-300 font-semibold ${className}`}>
    {children}
  </div>
);

const CardContent = ({ children, className = "" }) => (
  <div className={`p-4 text-gray-200 ${className}`}>
    {children}
  </div>
);

const HeadingContainer = styled.div`
  margin-bottom: 15px;
  margin-top: 15px;
  font-size: large;
`;

export default function DataExploration() {
  const template_images_list = [
    { src: Andromeda },
    { src: Aries },
    { src: Bootes },
    { src: Gemini },
    { src: Leo },
    { src: Orion },
  ];

  const stellarium_images_list = [
    { src: Andromeda_stellarium },
    { src: Aries_stellarium },
    { src: Bootes_stellarium },
    { src: Gemini_stellarium },
    { src: Leo_stellarium },
    { src: Orion_stellarium },
  ];

  const roboflow_images_list = [
    { src: img1 },
    { src: img2 },
    { src: img3 },
    { src: img4 },
    { src: img5 },
    { src: img6 },
  ];

  return (
    <div style={{
      position: "absolute",
      top: 0,
      left: 0,
      width: "100%",
      marginTop: "550px",
    }}>
      <div style={{
        fontFamily: "SpaceGrotesk-VariableFont_wght",
        color: "ivory",
        margin: "20px",
        marginTop: "50px",
      }}>
        <h2 style={{ margin: "10px", color: "yellow" }}>DATA COLLECTION</h2>
        <div style={{ margin: "10px" }}>
          Data was collected from multiple sources using various methods as
          detailed below:
          <ul>
            <li>
              <HeadingContainer>Constellation template Images</HeadingContainer>
              <p>
                These were collected by writing a{" "}
                <a
                  style={{ color: "salmon" }}
                  target="_blank"
                  href="https://github.com/rahul7310/stellar_mapping/blob/main/data_collection/get_template_images.py"
                >
                  python script
                </a>{" "}
                to scrape the data from two different websites,{" "}
                <a
                  style={{ color: "salmon" }}
                  target="_blank"
                  href="https://starchild.gsfc.nasa.gov/docs/StarChild/questions/88constellations.html"
                >
                  starchild.gsfc.nasa.gov
                </a>{" "}
                was scraped to get a list of the names of all 88 officially
                recognised constellations. The list was then used to scrape
                the images from{" "}
                <a
                  style={{ color: "salmon" }}
                  target="_blank"
                  href="https://astronomyonline.org/Observation/Constellations.asp?Cate=Observation&SubCate=MP07&SubCate2=MP0801"
                >
                  AstronomyOnline.org
                </a>
                <HeadingContainer>Sample Images Collected</HeadingContainer>
                <ImageGallery
                  imagesInfoArray={template_images_list}
                  columnCount={"auto"}
                  columnWidth={400}
                  gapSize={24}
                />
              </p>
            </li>
            <li>
              <HeadingContainer>Constellation mappings using the Stellarium app</HeadingContainer>
              <p>
                Captured images of each of the constellations using a custom{" "}
                <a
                  style={{ color: "salmon" }}
                  target="_blank"
                  href="https://github.com/rahul7310/stellar_mapping/blob/main/data_collection/get_stellarium_images.ssc"
                >
                  Stellarium script
                </a>
              </p>
            </li>
            <HeadingContainer>Sample Images Collected</HeadingContainer>
            <ImageGallery
              imagesInfoArray={stellarium_images_list}
              columnCount={"auto"}
              columnWidth={400}
              gapSize={24}
              showThumbnails={true}
            />

            <li>
              <HeadingContainer>Roboflow Universe Dataset</HeadingContainer>
              <p>
                We utilized the Constellation Dataset from{" "}
                <a
                  style={{ color: "salmon" }}
                  target="_blank"
                  href="https://universe.roboflow.com/ws-qwbuh/constellation-dsphi"
                >
                  Roboflow Universe,
                </a>{" "}
                which contains approximately 2,000 labeled images of various
                constellations. These images include annotations for the stars
                that comprise each constellation. The images are of uneven
                quality and resolution, mimicking how real-world conditions
                could affect visibility and recognition.
              </p>
            </li>
            <HeadingContainer>Sample Images Collected</HeadingContainer>
            <ImageGallery
              imagesInfoArray={roboflow_images_list}
              columnCount={"auto"}
              columnWidth={400}
              gapSize={24}
              showThumbnails={true}
            />

            <li>
              <HeadingContainer>Instructables Star Recognition</HeadingContainer>
              The Instructables{" "}
              <a
                style={{ color: "salmon" }}
                target="_blank"
                href="https://www.instructables.com/Star-Recognition-Using-Computer-Vision-OpenCV/"
              >
                article
              </a>{" "}
              on Star Recognition Using Computer Vision provided practical
              insights into using OpenCV for star recognition.
            </li>
          </ul>

          <h2 style={{ color: "yellow" }}>DATA PREPARATION AND ANALYSIS </h2>
          <div>
            <h3 style={{ color: "yellow", marginTop: "20px" }}>2.1 Dataset Characteristics</h3>
            <Card>
              <CardContent>
                <ul className="list-disc pl-6 space-y-2">
                  <li>Total images: 1,641 night sky images</li>
                  <li>Number of classes: 16 different constellation classes</li>
                  <li>Labels per image: Average 2.99 labels</li>
                  <li>Total annotations: 4,909 constellation annotations</li>
                  <li>Image dimensions: 640×640 pixels (1:1 aspect ratio)</li>
                </ul>
              </CardContent>
            </Card>

            <h3 style={{ color: "yellow", marginTop: "20px" }}>2.2 Class Distribution Analysis</h3>
            <Card>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="text-yellow-200 mb-2">Frequently Observed (IR &lt; 1.30):</h4>
                    <ul className="list-disc pl-6">
                      <li>Cassiopeia (IR: 1.00)</li>
                      <li>Pleiades (IR: 1.16)</li>
                      <li>Ursa Major (IR: 1.21)</li>
                      <li>Cygnus (IR: 1.27)</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-yellow-200 mb-2">Moderately Represented:</h4>
                    <ul className="list-disc pl-6">
                      <li>Lyra (IR: 1.28)</li>
                      <li>Moon (IR: 1.36)</li>
                      <li>Orion (IR: 1.46)</li>
                      <li>Bootes (IR: 1.96)</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-yellow-200 mb-2">Under-represented:</h4>
                    <ul className="list-disc pl-6">
                      <li>Gemini (IR: 2.05)</li>
                      <li>Leo (IR: 2.31)</li>
                      <li>Canis Major (IR: 2.56)</li>
                      <li>Sagittarius (IR: 2.81)</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>

            <h3 style={{ color: "yellow", marginTop: "20px" }}>2.3 Data Preprocessing</h3>
            <Card>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <h4 className="text-yellow-200 mb-2">Spatial Augmentations:</h4>
                    <ul className="list-disc pl-6">
                      <li>Random rotation (±15 degrees, p=0.5)</li>
                      <li>Random 90-degree rotation (p=0.3)</li>
                      <li>Random horizontal flip</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-yellow-200 mb-2">Intensity Augmentations:</h4>
                    <div>
                      <p>Stream 1 (p=0.3):</p>
                      <ul className="list-disc pl-6">
                        <li>Gaussian noise (variance: 10.0-50.0, p=0.5)</li>
                        <li>Gaussian blur (kernel: 3-5 pixels, p=0.5)</li>
                      </ul>
                      <p className="mt-2">Stream 2 (p=0.3):</p>
                      <ul className="list-disc pl-6">
                        <li>Brightness/contrast adjustment (±20%, p=0.5)</li>
                        <li>Gamma correction (range: 80-120, p=0.5)</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
                <CardHeader>Augmentation Examples</CardHeader>
                <CardContent>
                  <p className="mb-4">
                    Visual demonstration of our augmentation pipeline showing how a single image is 
                    transformed using different augmentation techniques. Examples include geometric 
                    transformations (rotation, flips) and intensity adjustments (brightness, contrast), 
                    which help improve model robustness.
                  </p>
                  <div className="flex justify-center">
                    <img 
                      src="./Augmentation.png" 
                      style={{ maxWidth: '600px', height: 'auto' }} 
                      alt="Augmentation examples"
                    />
                  </div>
                </CardContent>
              </Card>

              

            <h3 style={{ color: "yellow", marginTop: "20px" }}>2.4 Data Visualization</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Visualization cards */}
              <Card>
                <CardHeader>Image Count Distribution</CardHeader>
                <CardContent>
                  <p className="mb-4">
                    Bar plot showing the distribution of images across different constellation classes. 
                    Cassiopeia dominates with approximately 500 instances, while Sagittarius represents 
                    the minority class with around 180 instances. This visualization helps identify class 
                    imbalance issues in the dataset.
                  </p>
                  <div className="flex justify-center">
                    <img 
                      src="./image_count_distribution.png" 
                      style={{ maxWidth: '600px', height: 'auto' }} 
                      alt="Image count distribution"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>Pixel Intensity Analysis</CardHeader>
                <CardContent>
                  <p className="mb-4">
                    Evaluating the average and variability of pixel intensities for each class can reveal 
                    differences in brightness or contrast between classes. This information can inform 
                    preprocessing strategies and help optimize model training. The analysis helps identify 
                    any systematic differences in image characteristics across constellation classes.
                  </p>
                  <div className="flex justify-center">
                    <img 
                      src="./MeanPixelIntensity.png" 
                      style={{ maxWidth: '600px', height: 'auto' }} 
                      alt="Pixel intensity analysis"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>Constellation Mapping</CardHeader>
                <CardContent>
                  <p className="mb-4">
                    The constellation map provides a visual representation of the relationships between 
                    different constellations. By visualizing the proximity and overlap of constellations, 
                    we can gain insights into their spatial distribution and potential correlations. This 
                    visualization can be useful for understanding the overall structure of the celestial 
                    sphere and identifying potential challenges in constellation recognition tasks, such 
                    as overlapping star patterns or ambiguous boundaries.
                  </p>
                  <div className="flex justify-center">
                    <img 
                      src="./ConstellationMapping.png" 
                      style={{ maxWidth: '600px', height: 'auto' }} 
                      alt="Constellation mapping"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>Constellation Boundaries</CardHeader>
                <CardContent>
                  <p className="mb-4">
                    This visualization provides a detailed view of the 88 officially recognized constellations, 
                    outlining their boundaries and highlighting their relative positions in the night sky. 
                    By clearly delineating the boundaries, this map helps to distinguish between different 
                    constellations and avoid confusion, especially in areas where star patterns overlap. 
                    This visualization is a valuable tool for stargazers and astronomers alike, aiding in 
                    the identification and study of celestial objects.
                  </p>
                  <div className="flex justify-center">
                    <img 
                      src="./ConstellationBoundaries.png" 
                      style={{ maxWidth: '600px', height: 'auto' }} 
                      alt="Constellation boundaries"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>Color Channel Distribution</CardHeader>
                <CardContent>
                  <p className="mb-4">
                    Three histograms showing the distribution of average pixel values for red, green, 
                    and blue channels. These distributions help understand color biases in the dataset 
                    and guide preprocessing decisions for color normalization.
                  </p>
                  <div className="flex flex-col items-center space-y-4">
                    <img 
                      src="./color_channel_distribution_red.png" 
                      style={{ maxWidth: '600px', height: 'auto' }} 
                      alt="Red channel distribution"
                    />
                    <img 
                      src="./color_channel_distribution_blue.png" 
                      style={{ maxWidth: '600px', height: 'auto' }} 
                      alt="Blue channel distribution"
                    />
                    <img 
                      src="./color_channel_distribution_green.png" 
                      style={{ maxWidth: '600px', height: 'auto' }} 
                      alt="Green channel distribution"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>PCA Visualization</CardHeader>
                <CardContent>
                  <p className="mb-4">
                    A scatter plot of images projected onto their first two principal components. 
                    This visualization reveals clusters or patterns in the dataset that might not 
                    be apparent from other plots, helping understand the overall structure and 
                    variability of the data.
                  </p>
                  <div className="flex justify-center">
                    <img 
                      src="./image_pca_visualization.png" 
                      style={{ maxWidth: '600px', height: 'auto' }} 
                      alt="PCA visualization"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>Image Statistics</CardHeader>
                <CardContent>
                  <p className="mb-4">
                    Distribution plots showing the mean, standard deviation, skewness, and kurtosis 
                    of pixel values across all images. These statistics help identify outliers and 
                    understand the overall characteristics of the images, guiding preprocessing decisions.
                  </p>
                  <div className="flex flex-col items-center space-y-4">
                    <img 
                      src="./image_statistics_distribution1.png" 
                      style={{ maxWidth: '600px', height: 'auto' }} 
                      alt="Image statistics 1"
                    />
                    <img 
                      src="./image_statistics_distribution2.png" 
                      style={{ maxWidth: '600px', height: 'auto' }} 
                      alt="Image statistics 2"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>Image Complexity</CardHeader>
                <CardContent>
                  <p className="mb-4">
                    A histogram showing the distribution of image complexity, measured by the standard 
                    deviation of pixel values. This helps identify if there's a good mix of simple and 
                    complex images in the dataset and can guide decisions on data augmentation or 
                    model complexity.
                  </p>
                  <div className="flex justify-center">
                    <img 
                      src="./image_complexity.png" 
                      style={{ maxWidth: '600px', height: 'auto' }} 
                      alt="Image complexity"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>Distribution of Labels</CardHeader>
                <CardContent>
                  <p className="mb-4">
                    Bar plot showing the distribution of labels across different constellation classes.
                    The plot helps identify the most common and rarest constellations in the dataset,
                    which can guide data preprocessing and model training strategies.
                  </p>
                  <div className="flex justify-center">
                    <img

                      src="./label_distribution.png"
                      style={{ maxWidth: '600px', height: 'auto' }}
                      alt="Label distribution"
                    />
                  </div>
                </CardContent>
              </Card>

              <h3 style={{ color: "yellow", marginTop: "20px" }}>2.5 Co-occurance Analysis</h3>
            <Card>
                <CardHeader>Correlation Analysis</CardHeader>
                <CardContent>
                  <p className="mb-4">
                    Heatmap showing co-occurrence patterns between different constellations. Strong 
                    correlations exist between certain constellation pairs like Canis Major and Canis 
                    Minor (178 co-occurrences), Gemini and Canis Minor (233 co-occurrences), and Lyra 
                    and Cygnus (288 co-occurrences). This information helps understand natural groupings 
                    and seasonal patterns in constellation visibility.
                  </p>
                  <div className="flex flex-col items-center space-y-4">
                    <div className="flex justify-center">
                      <img 
                        src="./Co-occurence_1.png" 
                        style={{ maxWidth: '600px', height: 'auto' }} 
                        alt="Correlation matrix"
                      />
                    </div>
                    <div className="flex justify-center">
                      <img 
                        src="./Label_correlation.png" 
                        style={{ maxWidth: '600px', height: 'auto' }} 
                        alt="Correlation matrix"
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            <h3 style={{ color: "yellow", marginTop: "20px" }}>2.6 Dataset Splitting</h3>
            <Card>
              <CardContent>
                <p>The dataset is stratified into:</p>
                <ul className="list-disc pl-6 mt-2">
                  <li>Training set: 70%</li>
                  <li>Validation set: 15%</li>
                  <li>Test set: 15%</li>
                </ul>
                <p className="mt-4">Using iterative stratification strategy considering constellation co-occurrence patterns.</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}