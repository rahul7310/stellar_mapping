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
          <h2>Data Collection</h2>
          <div>
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
                    target="_blank"
                    href="https://github.com/rahul7310/stellar_mapping/blob/main/data_collection/get_template_images.py"
                  >
                    python script
                  </a>{" "}
                  to scrape the data from two different websites,{" "}
                  <a
                    target="_blank"
                    href="https://starchild.gsfc.nasa.gov/docs/StarChild/questions/88constellations.html"
                  >
                    starchild.gsfc.nasa.gov
                  </a>{" "}
                  was scraped to get a list of the names of all 88 officially
                  recognised constellations. The list was then used to scrape
                  the images from{" "}
                  <a
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
                  We utilized the Constellation Dataset from Roboflow Universe,
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
            </ul>
            <h2>Data Preparation information</h2>
            <div>
              In this project, we performed several data preprocessing steps to
              prepare our dataset for training a model to identify
              constellations using astronomical images. The following outlines
              the key steps involved in our preprocessing pipeline, along with
              code snippets and explanations.
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
