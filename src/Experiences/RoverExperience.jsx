import { OrbitControls, Stars, useGLTF } from "@react-three/drei";
import { useFrame, useLoader } from "@react-three/fiber";
import { useRef } from "react";
import { TextureLoader } from "three/src/Three.js";
import RocketShip from "../Models/RocketShip";
import SpaceRover from "../Models/SpaceRover";

export default function RoverExperience() {
  const surfaceTexture = useLoader(
    TextureLoader,
    "/textures/Surface_texture_2.jpg"
  );
  const surfaceTextureNormal = useLoader(
    TextureLoader,
    "/textures/Surface_texture_2_normal.jpg"
  );

  const sphereRef = useRef();

  useFrame(() => {
    if (sphereRef.current) {
      sphereRef.current.position.z -= 0.002;
    }
  });
  //   const rover = useGLTF("/models/Rocketship.glb");

  //   console.log(rover);
  const directionalLight = useRef();

  const ForeignPlanet = () => {
    return (
      <mesh
        ref={sphereRef}
        rotation={[Math.PI * 1.5, Math.PI * 2, Math.PI * 1.75]}
        receiveShadow
        position={[0, 0, 0]}
      >
        <planeGeometry receiveShadow args={[60, 30]} scale={1} />
        {/* <sphereGeometry receiveShadow args={[1, 256, 256]} /> */}
        <meshStandardMaterial
          map={surfaceTexture}
          normalMap={surfaceTextureNormal}
        />
      </mesh>
    );
  };

  return (
    <>
      <ambientLight intensity={0.7} />

      <OrbitControls />

      <directionalLight
        ref={directionalLight}
        castShadow
        position={[4, 2, 3]}
        intensity={4}
      />

      <ForeignPlanet />
      <SpaceRover />
      <Stars
        radius={100}
        depth={50}
        count={1500}
        factor={6}
        saturation={0}
        fade
      />
      {/* <primitive
        castShadow
        object={rover.scene}
        scale={0.75}
        rotation-y={0.5}
        position-y={0.6}
      ></primitive> */}
      {/* <mesh position-y={0} receiveShadow rotation-x={-Math.PI * 0.5} scale={10}>
        <planeGeometry />
        <meshStandardMaterial
          map={surfaceTexture}
          normalMap={surfaceTextureNormal}
          color={"black"}
        />
      </mesh> */}
    </>
  );
}
