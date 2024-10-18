import styled from "styled-components";

export default function Team() {
  return (
    <>
      <h1
        style={{
          color: "ivory",
          display: "flex",
          justifyContent: "center",
          marginTop: "100px",
          marginBottom: "75px",
          fontFamily: "SpaceGrotesk-VariableFont_wght",
        }}
      >
        The Team
      </h1>
      <TeamContainer>
        <BioCard>
          <img
            style={{
              width: "200px",
              height: "200px",
              borderRadius: "50%",
              objectFit: "cover",
            }}
            src="./Nandini_photo.jpg"
            alt=" Nandini"
            className="profile-picture"
          />
          <h4 style={{ color: "ivory" }}>Tata Sai Nandini</h4>
          <p style={{ marginTop: "10px", textAlign: "center", color: "white" }}>
            Tata Sai Nandini is an AI developer with two years of experience,
            holding an undergraduate degree from Anna University, Chennai. Her
            passion for AI ignited during a Coursera course in the lockdown,
            leading her to participate in hackathons and organize workshops on
            Machine Learning and Python. Proficient in SQL and certified in
            Power BI, she shares her knowledge on Hashnode. Nandini's technical
            skills include Python, C++, and various machine learning frameworks.
            Beyond tech, she is a trained Carnatic singer, a regional-level
            chess player, and an amateur guitarist.
          </p>
          <a
          href="https://www.linkedin.com/in/tatasainandini/"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "lightblue", marginTop: "10px" }}
        >
          LinkedIn
        </a>
        </BioCard>
        <BioCard>
          <img
            style={{
              width: "200px",
              height: "200px",
              borderRadius: "50%",
              objectFit: "cover",
            }}
            src="./Siva_photo.jpg"
            alt="Siva"
            className="profile-picture"
          />
          <h4 style={{ color: "ivory" }}>Sivakumar</h4>
          <p style={{ marginTop: "10px", textAlign: "center", color: "white" }}>
            Exploring and learning new stuffs! Physics is my favourite, Math is
            my queen, Programming since 2018. Having work experience in the
            domain of applied Deep Learning. As a part of the Natural Language
            Processing Research team at ZOHO corporation, I have developed
            RAG-based chatbot that is now scaled for millions of users. From
            predicting whether a customer requires assistance, detecting
            phishing mails, to analyzing root cause of customer's negative
            remarks and thereby reducing customer churn by addressing the
            relevant issue, I have worked on many other generative AI tasks like
            Content creation, Email reply generation, FAQ generation and so on.
            I was also a part of the research work involving multimodality and
            Vision Language Models.
          </p>
          <a
          href="https://www.linkedin.com/in/siva-kumar-5b2527190/"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "lightblue", marginTop: "10px" }}
        >
          LinkedIn
        </a>
        </BioCard>
        <BioCard>
          <img
            style={{
              width: "200px",
              height: "200px",
              borderRadius: "50%",
              objectFit: "cover",
            }}
            src="./Rahul_photo.jpg"
            alt="Rahul"
            className="profile-picture"
          />
          <h4 style={{ color: "ivory" }}>Rahul Prasanna</h4>
          <p
            style={{
              marginTop: "10px",
              textAlign: "center",
              color: "white",
              width: "100%",
            }}
          >
            Bachelors in Computing and Applied math, 4 years of experience
            working as a Software Engineer across industries. Passionate about
            Math and a keen interest in data science. Enjoy building things.
          </p>
          <a
          href="https://github.com/rahul7310"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "lightblue", marginTop: "10px" }}
        >
          GitHub
        </a>
        </BioCard>
      </TeamContainer>
    </>
  );
}
const TeamContainer = styled.div`
  font-family: "SpaceGrotesk-VariableFont_wght";
  @media (max-width: 768px) {
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  @media (min-width: 769px) {
    // Styles for tablets and desktops
    display: flex;
  }
`;

const BioCard = styled.div`
  @media (max-width: 768px) {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
  }
  @media (min-width: 769px) {
    // Styles for tablets and desktops
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    width: 30%;
  }
`;