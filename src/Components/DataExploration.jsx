// import { IpynbRenderer } from "react-ipynb-renderer";
// import { IpynbRenderer } from "react-ipynb-renderer";

// import "react-ipynb-renderer/dist/styles/monokai.css";
// import ipynb from "../analytics/image_analytics/data_preprocessing.json";
// import ipynb from "../../datasets/constellation_templates/"
import { ImageGallery } from "react-image-grid-gallery";
import styled from "styled-components";

import Andromeda from "../../datasets/constellation_templates/Andromeda.jpg";
import Aries from "../../datasets/constellation_templates/Aries.jpg";
import Bootes from "../../datasets/constellation_templates/Bootes.jpg";
import Gemini from "../../datasets/constellation_templates/Gemini.jpg";
import Leo from "../../datasets/constellation_templates/Leo.jpg";
import Orion from "../../datasets/constellation_templates/Orion.jpg";

import Andromeda_stellarium from "../../datasets/stellarium_constellations/Andromeda001.png";
import Aries_stellarium from "../../datasets/stellarium_constellations/Aries001.png";
import Bootes_stellarium from "../../datasets/stellarium_constellations/Bootes001.png";
import Gemini_stellarium from "../../datasets/stellarium_constellations/Gemini001.png";
import Leo_stellarium from "../../datasets/stellarium_constellations/Leo001.png";
import Orion_stellarium from "../../datasets/stellarium_constellations/Orion001.png";

import img1 from "../../datasets/roboflow_constellation_images/train/2022-01-08-00-00-00-n_png_jpg.rf.5ad2e498f3937e5fc7f746ff4954afca.jpg";
import img2 from "../../datasets/roboflow_constellation_images/train/2022-01-10-00-00-00-s_png_jpg.rf.9949ec204bd5b8af72cf2cb22a3402b0.jpg";
import img3 from "../../datasets/roboflow_constellation_images/train/2022-01-15-00-00-00-s_png_jpg.rf.1c04821ef762c0af4d5bf8772e6ac08d.jpg";
import img4 from "../../datasets/roboflow_constellation_images/train/2022-01-22-00-00-00-s_png_jpg.rf.37e31378be869cbc5da12083a9452f1c.jpg";
import img5 from "../../datasets/roboflow_constellation_images/train/2022-01-30-00-00-00-s_png_jpg.rf.b5eee58c97e33ef80b95c30a8aad3d23.jpg";
import img6 from "../../datasets/roboflow_constellation_images/train/2022-01-30-00-00-00-s_png_jpg.rf.b5eee58c97e33ef80b95c30a8aad3d23.jpg";

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
    <>
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          marginTop: "50px",
        }}
      >
        <div
          style={{
            fontFamily: "SpaceGrotesk-VariableFont_wght",
            color: "ivory",
            margin: "20px",
            marginTop: "50px",
          }}
        >
          <h2 style={{ margin: "10px", color: "yellow" }}>Data Collection</h2>
          <div style={{ margin: "10px" }}>
            Data was collected from multiple sources using various methods as
            detailed below,
            <ul>
              <li>
                <HeadingContainer>
                  Constellation template Images
                </HeadingContainer>
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
                <HeadingContainer>
                  Constellation mappings using the Stellarium app
                </HeadingContainer>
                <p>
                  Captured images of each of the constellations using a custom{" "}
                  <a
                    style={{ color: "salmon" }}
                    target="_blank"
                    href="https://github.com/rahul7310/stellar_mapping/blob/main/data_collection/get_stellarium_images.ssc"
                  >
                    Stellarium script
                  </a>{" "}
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
                <HeadingContainer>
                  Instructables Star Recognition
                </HeadingContainer>
                The Instructables{" "}
                <a
                  target="_blank"
                  href="https://www.instructables.com/Star-Recognition-Using-Computer-Vision-OpenCV/"
                >
                  article
                </a>{" "}
                on Star Recognition Using Computer Vision provided practical
                insights into using OpenCV for star recognition. While this
                resource doesn't offer a specific dataset, it provides
                guidelines and methodologies relevant to image processing, which
                were crucial for our data preparation steps.
              </li>
            </ul>
            <h2 style={{ color: "yellow" }}>Data Preparation</h2>
            <div>
              In this project, we performed several data preprocessing steps to
              prepare our dataset for training a model to identify
              constellations using astronomical images. The following outlines
              the key steps involved in our preprocessing pipeline, along with
              code snippets and explanations. Here's a breakdown of each step:
              <ul>
                <li>
                  <b>Resize: transforms.Resize((128, 128))</b>
                  <br></br>
                  Resizes the input images to a fixed size of 128x128 pixels.
                  This ensures uniformity in input dimensions, which is crucial
                  for batch processing in neural networks.
                </li>
                <br></br>
                <li>
                  <b>
                    Random Horizontal Flip: transforms.RandomHorizontalFlip()
                  </b>
                  <br></br>
                  Randomly flips the image horizontally with a probability of
                  0.5. This augmentation helps the model become invariant to
                  horizontal orientation, which can improve generalization.
                </li>
                <br></br>
                <li>
                  <b>Random Rotation: transforms.RandomRotation(15)</b>
                  <br></br>
                  Randomly rotates the image by up to 15 degrees in either
                  direction. This adds rotational variability to the dataset,
                  helping the model learn features that are less sensitive to
                  orientation.
                </li>
                <br></br>
                <li>
                  <b>Convert to Tensor: transforms.ToTensor()</b>
                  <br></br>
                  Converts the image (likely in PIL format or a NumPy array) to
                  a PyTorch tensor. This transformation also scales pixel values
                  to the range [0, 1].
                </li>
                <br></br>
                <li>
                  <b>Cutout: Cutout(n_holes=1, length=16)</b>
                  <br></br>
                  This custom transformation randomly selects a region (16x16
                  pixels) in the image and sets its values to zero. This
                  simulates occlusion, encouraging the model to focus on the
                  remaining visible features and improving robustness.
                </li>
                <br></br>
                <li>
                  <b>Normalize: transforms.Normalize((0.5,), (0.5,))</b>
                  <br></br>
                  Normalizes the tensor image to have a mean of 0.5 and a
                  standard deviation of 0.5 for each channel. This is important
                  for speeding up convergence during training and can help the
                  model perform better by ensuring that inputs are centered
                  around zero.
                </li>
                <br></br>
                Together, these preprocessing steps enhance the dataset through
                augmentation (increasing diversity), ensure consistent input
                sizes, convert images to the required format, and normalize
                pixel values for effective model training. This pipeline aims to
                improve model robustness and generalization by exposing it to
                varied versions of the input images. We load an example image
                and apply the defined transformations. The original and
                transformed images are displayed side by side for visual
                comparison. This helps to illustrate the effect of our
                preprocessing pipeline.
                <br></br>
                <br></br>
                The following contains a sample from the dataset that shows how
                images are, before and after preprocessing:
                <br></br>
                <br></br>
                {/* <img src="./image_processing_sample.png"></img> */}
                <ImageGallery
                  imagesInfoArray={[{ src: "./image_processing_sample.png" }]}
                  columnCount={"auto"}
                  columnWidth={600}
                  gapSize={24}
                  showThumbnails={true}
                />
              </ul>
              <h2 style={{ color: "yellow" }}>Data Visualization</h2>
              <ul>
                <li>
                  <HeadingContainer>
                    Image Count Distribution by Class
                  </HeadingContainer>
                  Bar plot of distribution of images across different classes.
                </li>
                <br></br>
                <ImageGallery
                  imagesInfoArray={[{ src: "./image_count_distribution.png" }]}
                  columnCount={"auto"}
                  columnWidth={600}
                  gapSize={24}
                  showThumbnails={true}
                />

                <li>
                  <HeadingContainer>
                    Color Channel Distribution
                  </HeadingContainer>
                  Three histograms showing the distribution of average pixel
                  values for red, green, and blue channels.
                </li>
                <br></br>
                <ImageGallery
                  imagesInfoArray={[
                    { src: "./color_channel_distribution_red.png" },
                    { src: "./color_channel_distribution_blue.png" },
                    { src: "./color_channel_distribution_green.png" },
                  ]}
                  columnCount={"auto"}
                  columnWidth={600}
                  gapSize={24}
                  showThumbnails={true}
                />
                <li>
                  <HeadingContainer>
                    PCA Visualization of Images
                  </HeadingContainer>
                  A scatter plot of images projected onto their first two
                  principal components.
                </li>
                <br></br>
                <ImageGallery
                  imagesInfoArray={[{ src: "./image_pca_visualization.png" }]}
                  columnCount={"auto"}
                  columnWidth={600}
                  gapSize={24}
                  showThumbnails={true}
                />

                <li>
                  <HeadingContainer>
                    Image Statistics Distributions
                  </HeadingContainer>
                  Four histograms showing the distribution of mean, standard
                  deviation, skewness, and kurtosis of pixel values across all
                  images.
                </li>
                <br></br>
                <ImageGallery
                  imagesInfoArray={[
                    { src: "./image_statistics_distribution1.png" },
                    { src: "./image_statistics_distribution2.png" },
                  ]}
                  columnCount={"auto"}
                  columnWidth={600}
                  gapSize={24}
                  showThumbnails={true}
                />
                <li>
                  <HeadingContainer>
                    Distribution of Image Complexity
                  </HeadingContainer>
                  A histogram of image complexity, measured by the standard
                  deviation of pixel values.
                </li>
                <br></br>

                <ImageGallery
                  imagesInfoArray={[{ src: "./image_complexity.png" }]}
                  columnCount={"auto"}
                  columnWidth={600}
                  gapSize={24}
                  showThumbnails={true}
                />
              </ul>
            </div>
          </div>
        </div>
        <div style={{ margin: "20px" }}>
          {/* <IpynbRenderer ipynb={ipynb} /> */}
        </div>
      </div>
    </>
  );
}

const HeadingContainer = styled.div`
  margin-bottom: 15px;
  margin-top: 15px;
  font-size: large;
`;
