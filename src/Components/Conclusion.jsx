import { ImageGallery } from "react-image-grid-gallery";
import styled from "styled-components";

export default function Conclusion() {
  return (
    <>
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          marginTop: "350px",
        }}
      >
        <div
          style={{
            fontFamily: "SpaceGrotesk-VariableFont_wght",
            color: "ivory",
            margin: "20px",
            marginTop: "20px",
          }}
        >
          <p>
            We embarked on this project in an attempt to make astronomy more
            accessible to the public through Data Science. Constellations, rich
            in both cultural and scientific significance, emerged as the ideal
            subject. Not only are they universally recognized, but they also
            serve as an excellent entry point for people of all ages to delve
            into the wonders of space.
          </p>

          <p>
            We set out to develop a user-friendly tool that could analyze a
            photo of the night sky and identify the constellations present.
            Making astronomy more accessible by empowering individuals and
            inspire them to explore the cosmos. Fostering a new generation of
            stargazers while also contributing and driving innovation in the
            data science – astronomy community.
          </p>

          <p>
            As we delved into this project, we learned a great deal about both
            data science and astronomy. Our journey through this project was
            akin to a space exploration of its own. Like any exploration, we
            encountered unexpected challenges. Initially, we aimed to identify
            all 88 constellations, but due to data constraints, we had to focus
            on the 16 most popular constellations. Despite these hurdles, we
            were successful in training four models with impressive
            performances.
          </p>

          <p>
            This project has further reinforced our belief in the potential of
            data science in bridging the gap between complex scientific fields
            and the general public. It also has the potential to serve as a
            valuable educational tool for people of all ages, especially in
            collaboration with planetariums and science centers. It also has the
            potential to aid in researching more complex concepts in astronomy
            and can have a larger impact on the broader field of citizen
            science.
          </p>

          <h3 style={{ color: "yellow" }}>Future Work and Improvements</h3>
          <ul>
            <li>
              <HeadingContainer>Data Collection</HeadingContainer>
              <p>
                To enhance the model's capabilities, we aim to expand the
                dataset to include all 88 officially recognized constellations.
                By incorporating images from various seasons, locations, and
                atmospheric conditions, we can improve the model's robustness
                and accuracy. Additionally, integrating the model with
                astronomical databases will provide access to valuable
                contextual information.
              </p>
            </li>

            <li>
              <HeadingContainer>Annotation Improvements</HeadingContainer>
              <p>
                We plan to refine our model's output by incorporating bounding
                boxes around identified constellations. This visual enhancement
                will improve the clarity and accuracy of the annotations. In
                addition, we aim to develop techniques to visualize the
                constellation patterns, further enriching the user experience.
              </p>
            </li>

            <li>
              <HeadingContainer>Chatbot</HeadingContainer>
              <p>
                Recognizing the significant educational potential of this
                project, we envision adding a chatbot that can provide users
                with detailed information about the constellations they have
                identified. This interactive element would enhance user
                experience and not only provide valuable information but also
                encourage further exploration and learning.
              </p>
            </li>

            <li>
              <HeadingContainer>Real time detection</HeadingContainer>
              <p>
                To elevate the user experience to new heights, we envision
                incorporating real-time constellation recognition. By utilizing
                AR technology, users could point their device at the night sky
                and instantly identify constellations in their field of view.
                This feature would provide a truly interactive and immersive
                astronomical experience.
              </p>
            </li>
          </ul>
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
