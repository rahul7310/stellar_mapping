// import { IpynbRenderer } from "react-ipynb-renderer";
// import { IpynbRenderer } from "react-ipynb-renderer";

// import "react-ipynb-renderer/dist/styles/monokai.css";
// import ipynb from "../analytics/image_analytics/data_preprocessing.json";

export default function DataExploration() {
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
          <h3>Data Collection</h3>
          <p>
            Data was collected from multiple sources using various methods as
            detailed below,
            <ul>
              <li>
                <h4>Constellation template Images</h4>
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
                  <p>Sample Images Collected</p>
                </p>
              </li>
              <li>
                <h4>Constellation mappings using the Stellarium app</h4>
                <p>
                  Captured images of each of the constellations using a custom
                  <a
                    target="_blank"
                    href="https://github.com/rahul7310/stellar_mapping/blob/main/data_collection/get_stellarium_images.ssc"
                  >
                    Stellarium script
                  </a>
                  .
                </p>
              </li>
              <p>Sample Images Collected</p>

              <li>
                <h4>Large constellation data set</h4>
                <p>
                  Found an existing collection of night sky images classified
                  into constellations.
                </p>
              </li>
              <p>Sample Images Collected</p>
            </ul>
          </p>
        </div>
        <div style={{ margin: "20px" }}>
          {/* <IpynbRenderer ipynb={ipynb} /> */}
        </div>
      </div>
    </>
  );
}
