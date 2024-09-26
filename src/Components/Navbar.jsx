export default function Navbar({ setSelectedTab, selectedTab }) {
  console.log();
  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
      }}
    >
      <div
        style={{
          margin: "20px",
          display: "flex",
          color: "ivory",
          justifyContent: "space-evenly",
        }}
      >
        <div
          style={{
            cursor: "pointer",
            border:
              selectedTab === "Introduction" ? "solid 1px yellow" : "none",

            borderRadius: "5px",
          }}
          onClick={() => setSelectedTab("Introduction")}
        >
          <div style={{ margin: "5px" }}>Introduction</div>
        </div>
        <div
          style={{
            cursor: "pointer",
            border:
              selectedTab === "Data Exploration" ? "solid 1px yellow" : "none",
            borderRadius: "5px",
          }}
          onClick={() => setSelectedTab("Data Exploration")}
        >
          <div style={{ margin: "5px" }}>Data Exploration</div>
        </div>
        <div
          style={{
            cursor: "pointer",
            border:
              selectedTab === "Models Implemented"
                ? "solid 1px yellow"
                : "none",
            borderRadius: "5px",
          }}
          onClick={() => setSelectedTab("Models Implemented")}
        >
          <div style={{ margin: "5px" }}>Models Implemented</div>
        </div>
        <div
          style={{
            cursor: "pointer",
            border: selectedTab === "Conclusion" ? "solid 1px yellow" : "none",
            borderRadius: "5px",
          }}
          onClick={() => setSelectedTab("Conclusion")}
        >
          <div style={{ margin: "5px" }}>Conclusion</div>
        </div>
        <div
          style={{
            cursor: "pointer",
            border: selectedTab === "Team" ? "solid 1px yellow" : "none",
            borderRadius: "5px",
          }}
          onClick={() => setSelectedTab("Team")}
        >
          <div style={{ margin: "5px" }}>Team</div>
        </div>
      </div>
    </div>
  );
}
