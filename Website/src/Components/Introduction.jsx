export default function Introduction() {
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
            marginTop: "400px",
          }}
        >
          <h3 style={{ color: "yellow" }}>The nature of the topic</h3>
          <p>
            This project combines astronomy and technology. It's an attempt to
            create a system that can automatically recognize and label star
            patterns in images taken of the sky and give some extra information
            abt it. It's specifically useful for people who don't have a lot of
            constellation knowledge, to understand and appreciate the beauty of
            the stars. As more discoveries of celestial objects are being made
            on a regular basis due to large telescopes capturing clear and high
            definition images of deep space giving more interesting observations
            of our universe, the study of constellations is also gaining some
            progress in astronomical data analysis. As of today, The
            International Astronomical Union (IAU) recognizes 88 constellations.
            Among the 88 recognized constellations, there are various themes and
            representations. For instance, 42 of them depict animals, 29
            represent inanimate objects, and 17 illustrate humans or
            mythological characters. This diversity reflects humanity's
            long-standing fascination with the stars and their stories.
          </p>
          <h3 style={{ color: "yellow" }}>Why it is important?</h3>
          <p>
            Today, most people lack knowledge about constellations - the number
            of constellations, their types, their history, or even their
            spelling. This may be due to limited content on the subject, less
            time spent observing the beauty of the stars, or simply pollution
            making it difficult for people to spot them. It's important for
            everyone to learn about constellations as they have been used for
            centuries by sailors and travelers to navigate the earth, prediction
            of weather and climate changes. Even today, knowing how to identify
            constellations can help us find our way at night and help in
            predicting seasonal changes as well. These constellations are part
            of our cultural heritage and are featured in myths across different
            civilizations. By making it easier to identify these star patterns,
            we can connect more people with the wonders of astronomy and
            encourage interest in science and exploration.
          </p>
          <h3 style={{ color: "yellow" }}>Who is affected?</h3>
          <p>
            This project will benefit many people, from amateur astronomers to
            students and educators. For amateur astronomers, having a tool that
            quickly identifies constellations can enhance their stargazing
            experience. Students can use such a system in classrooms or during
            field trips to learn about space in a fun and interactive way.
          </p>
          <h3 style={{ color: "yellow" }}>
            What has been done so far and what gaps remain?
          </h3>
          <p>
            Despite the progress made, there are still gaps that need to be
            filled in this field. We need better ways to handle situations where
            only part of a constellation is visible or when stars are faint due
            to light pollution. Additionally, we should develop systems that can
            adapt to different cameras or telescopes used by stargazers around
            the world. As interest in space exploration continues to grow,
            having reliable tools for identifying constellations will become
            increasingly important.
          </p>
          <h3 style={{ color: "yellow" }}>Questions related to Dataset?</h3>
          <p>
            On a yearly basis, the astronomical catalogs are being released that
            contain huge lists of astronomical objects and that also includes
            the stars that make up the constellations. One example being Sloan
            Digital Sky Survey (SDSS) [1]that contains a repository of deep sky
            images in multiple formats and releases the new dataset with latest
            detected celestial objects on a yearly basis.
            <li>
              How we are going to classify constellation just by looking at the
              stars
            </li>
            <li>
              How we are going to predict multiple constellations in a single
              image
            </li>
            <li>
              How we’re going to use augmentation methods to improve the dataset
            </li>
            <li>
              How we are going to improve the quality of the images (data
              preprocessing)
            </li>
            <li>Which websites are we planning to get datasets from?</li>
            <li>
              Can the system accurately determine the user's geographical
              location based on the visible constellations?
            </li>
            <li>
              How do star formation rates vary across different regions of the
              galaxy?
            </li>
            <li>
              How do changes in the positions of stars affect our understanding
              of gravitational influences in space?
            </li>
            <li>
              Can we develop tools to help amateur astronomers contribute
              meaningfully to data collection?
            </li>
            <li>
              What role do constellations play in cultural astronomy across
              different societies?
            </li>
            <li>
              How can public engagement with astronomy be improved through data
              visualization?
            </li>
          </p>
        </div>
      </div>
    </>
  );
}
