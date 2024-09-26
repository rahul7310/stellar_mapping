import { Canvas } from "@react-three/fiber";
import Navbar from "./Components/Navbar";
import Experience from "./Experience";
import { useEffect, useState } from "react";
import Introduction from "./Components/Introduction";
import Team from "./Components/Team";

export default function App() {
  const [selectedTab, setSelectedTab] = useState("Introduction");

  useEffect(() => {
    console.log(selectedTab);
  }, [selectedTab]);
  return (
    <>
      <Navbar setSelectedTab={setSelectedTab} selectedTab={selectedTab} />
      {selectedTab === "Introduction" ? (
        <>
          <div
            style={{
              width: "100%",
              height: "100%",
              marginTop: "50px",
              overflow: "auto",
            }}
          >
            <Canvas
              camera={{
                fov: 45,
                near: 0.1,
                far: 2000,
                position: [-3, 1.5, 4],
              }}
            >
              <Experience />
            </Canvas>
            <Introduction />
          </div>
        </>
      ) : selectedTab === "Team" ? (
        <Team />
      ) : (
        <></>
      )}
    </>
  );
}
