import { Canvas } from "@react-three/fiber";
import Navbar from "./Components/Navbar";
import MoonExperience from "./Experiences/MoonExperience";
import { useEffect, useState } from "react";
import Introduction from "./Components/Introduction";
import Team from "./Components/Team";
import DataExploration from "./Components/DataExploration";
import ModelsImplemented from "./Components/ModelsImplemented";
import { Stars } from "@react-three/drei";
import RocketShipExperience from "./Experiences/RocketShipExperience";
import StarsAnimated from "./Experiences/StarFall";
import RoverExperience from "./Experiences/RoverExperience";
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
              <MoonExperience />
            </Canvas>
            <Introduction />
          </div>
        </>
      ) : selectedTab === "Team" ? (
        <Team />
      ) : selectedTab === "Data Exploration" ? (
        <div
          style={{
            width: "100%",
            height: "50%",
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
            <RocketShipExperience />
          </Canvas>

          <DataExploration />
        </div>
      ) : selectedTab === "Model Implemented" ? (
        <div
          style={{
            width: "100%",
            height: "100%",
            marginTop: "50px",
            // overflow: "auto",
          }}
        >
          <div style={{ height: "50vh" }}>
            <Canvas
              shadows
              camera={{
                fov: 45,
                near: 0.1,
                far: 2000,
                position: [-3, 1.5, 4],
              }}
            >
              <RoverExperience />
            </Canvas>
          </div>
      
          <div> {/* Content will follow the Canvas */}
            <ModelsImplemented />
          </div>
        </div>
      ) : (
        <></>
      )}
    </>
  );
}
