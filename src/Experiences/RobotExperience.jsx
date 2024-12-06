import { SpotLight, useGLTF } from "@react-three/drei";
import Robot from "../Models/Robot";
import { useEffect, useRef, useState } from "react";

import * as THREE from "three";
import { useFrame } from "@react-three/fiber";

export default function RobotExperience() {
  return (
    <>
      <ambientLight intensity={1} />
      <directionalLight castShadow position={[4, 2, 3]} intensity={2} />

      <Robot />
    </>
  );
}
