import {
  OrbitControls,
  PointMaterial,
  Points,
  Stage,
  Stars,
  useAnimations,
  useGLTF,
  useHelper,
} from "@react-three/drei";
import { useFrame, useLoader } from "@react-three/fiber";
import { Suspense, useEffect, useRef } from "react";
import { AnimationMixer, AnimationAction } from "three";
import { TextureLoader } from "three/src/Three.js";

import { useControls } from "leva";
import * as THREE from "three";

import Robot from "../Models/Robot";
import RocketShip from "../Models/RocketShip";
import StarFall from "./StarFall";
import Loading from "../Components/Loading";

export default function RocketShipExperience() {
  // const rocket = useGLTF("Rocket.glb");
  // const fox = useGLTF("Fox.glb");
  const robot = useGLTF("/models/Robot.glb");
  // console.log(robot);
  // const animations = useAnimations(rocket.animations, rocket.scene);
  // const animations_fox = useAnimations(fox.animations, fox.scene);
  const animations_robot = useAnimations(robot.animations, robot.scene);
  // console.log(animations_robot);

  // const marsTexture = useLoader(TextureLoader, 'Material_1900.png');

  //Controls
  // const controls = useControls({ position: -2 });

  const cube = useRef();
  const directionalLight = useRef();
  const pointsRef = useRef();

  const count = 1000; // Number of stars

  // useHelper(directionalLight, THREE.DirectionalLightHelper, 1);

  // const { rotation_x, rotation_y, rotation_z } = useControls({
  //   rotation_x: 0.5,
  //   rotation_y: 0.5,
  //   rotation_z: 0.5,
  // });

  const rocketShip = useRef();
  const starRef = useRef();
  useFrame(() => {
    if (rocketShip.current) {
      rocketShip.current.rotation.y += 0.003;
    }
    if (starRef.current) {
      if (starRef.current.position.y > -180) {
        starRef.current.position.y -= 0.1;
      } else {
        starRef.current.position.y = 0;
      }
    }
  });

  return (
    <>
      <Suspense fallback={<Loading />}>
        <ambientLight intensity={0.3} />

        <directionalLight
          ref={directionalLight}
          castShadow
          position={[1, 2, 3]}
          intensity={3}
        />

        <StarFall />

        <group position={[0, -1.5, 0]} rotation-z={Math.PI * -0.15}>
          <mesh ref={rocketShip}>
            {/* <Robot scale={0.37} position-x={1} /> */}
            <RocketShip scale={4} />
          </mesh>
        </group>
      </Suspense>
    </>
  );
}
