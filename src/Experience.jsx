import { Stars, useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useRef } from "react";

export default function Experience() {
  const moon = useGLTF("Moon.glb");

  const moonRef = useRef();

  useFrame(() => {
    if (moonRef.current) {
      moonRef.current.rotation.y += 0.005;
    }
  });

  return (
    <>
      <ambientLight intensity={3} />
      <rectAreaLight
        width={2.5}
        height={1.65}
        intensity={55}
        color={"#ff6900"}
        rotation={[-0.1, Math.PI, 0]}
        position={[-1, 1.25, -1.15]}
      />
      <mesh ref={moonRef}>
        <primitive
          object={moon.scene}
          scale={0.015}
          rotation-y={0.5}
          position-y={0.6}
        ></primitive>
      </mesh>

      <Stars
        radius={100}
        depth={50}
        count={15000}
        factor={6}
        saturation={0}
        fade
      />
    </>
  );
}
