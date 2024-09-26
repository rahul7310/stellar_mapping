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
            Constellations are groups of stars that form recognizable patterns
            or shapes in the night sky. They serve as celestial landmarks,
            helping us locate and understand the positions of stars and other
            astronomical objects. Constellations have been a huge part of human
            history from being used as navigational tools by sailors to being
            linked to seasons and agricultural cycles. They even have a place in
            the myths and lores of many cultures across the world. Currently,
            the International Astronomical Union (IAU) officially recognizes 88
            constellations. Among the 88 recognized constellations, there are
            various themes and representations. For instance, 42 of them depict
            animals, 29 represent inanimate objects, and 17 illustrate humans or
            mythological characters. Through this project we aim to leverage
            data science techniques to build a model that can accurately
            identify constellations in astronomical images.
          </p>
          <div></div>
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
            Accurate identification of constellations can help in tracking the
            movements of stars and planets. This can in turn help in the
            advancement of Astronomy as a whole. It can be extremely valuable in
            enhancing educational programs and outreach efforts of planetariums
            and science centers and helping them make astronomy more accessible
            to the general public. Furthermore, it could assist professional
            astronomers in their research by quickly identifying celestial
            objects. The technology might also be adapted for use in satellite
            navigation systems, improving their accuracy and reliability. By
            making astronomy more accessible to everyone, this project has the
            potential to inspire a genereation of future scientists and space
            explorers.
          </p>
          <h3 style={{ color: "yellow" }}>
            What has been done so far and what gaps remain?
          </h3>
          <p>
            Researchers have made significant strides in developing intelligent
            constellation detection algorithms and star recognition systems.
            These advanced technologies effectively detect stars, produce
            orientation data, and label celestial patterns with impressive
            accuracy. Utilizing computer vision techniques, particularly the
            OpenCV library and trained HAAR cascades, these systems can
            recognize specific star patterns. They have demonstrated a
            remarkable 90% success rate in correctly identifying sources and
            mapping constellations. The applications of these systems extend to
            various amateur astronomy projects, enhancing the experience for
            novice stargazers. Additionally, they hold potential for telescope
            orientation and automatic image classification, streamlining data
            analysis. Some systems developed at institutions like MIT Haystack
            Observatory study star-forming sites in our galaxy and nearby
            galaxies using interferometric arrays. Other projects aim to detect
            hydrogen signatures from the formation of the first stars and
            galaxies. The development of imaging algorithms for radio
            interferometry has also been essential in creating astronomical
            images from data collected by projects like the Event Horizon
            Telescope. Despite these advancements, challenges still exist that
            need attention. For example, better methods are needed to handle
            situations where only part of a constellation is visible or when
            stars are faint due to light pollution. Additionally, it is crucial
            to develop systems that can adapt to different cameras or telescopes
            used by stargazers worldwide. As interest in space exploration
          </p>
          <h3 style={{ color: "yellow" }}>Questions related to Dataset?</h3>
          <p>
            On a yearly basis, the astronomical catalogs are being released that
            contain huge lists of astronomical objects and that also includes
            the stars that make up the constellations. One example being Sloan
            Digital Sky Survey (SDSS) that contains a repository of deep sky
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
