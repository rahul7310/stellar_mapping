import { Points, Point, useTexture, OrbitControls } from "@react-three/drei";
import { useControls } from "leva";
import React, { useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";

const particleTextures = "textures/star_texture.png";

function Particles() {
  //   const { count, size, positionFactor, textureType, rotationSpeed } =
  //     useControls({
  //       textureType: { value: 3, min: 0, max: 12, step: 1 },
  //       count: { value: 1000, min: 1, max: 10000 },
  //       size: { value: 2, min: 1, max: 20 },
  //       positionFactor: { value: 120, min: 5, max: 200 },
  //       rotationSpeed: 0.1,
  //       position_y: 0,
  //     });
  const count = 1000;
  const size = 3;
  const particleTexture = useTexture(particleTextures);
  const particlesRef = useRef();

  useFrame(() => {
    if (particlesRef.current) {
      if (particlesRef.current.position.y > -70) {
        particlesRef.current.position.y -= 0.01;
      } else {
        particlesRef.current.position.y = 50;
      }
    }
  });
  return (
    <Points ref={particlesRef} limit={10000} position={[0, 50, 0]}>
      <pointsMaterial
        size={size}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        sizeAttenuation
        vertexColors
        map={particleTexture}
        alphaMap={particleTexture}
      />
      {Array.from({ length: count }).map((_, i) => (
        <Point
          key={i}
          position={[
            (0.5 - Math.random()) * 120,
            (0.5 - Math.random()) * 120,
            (0.5 - Math.random()) * 120,
          ]}
          color="white"
        />
      ))}
    </Points>
  );
}

export default function StarFall() {
  return <Particles />;
}
