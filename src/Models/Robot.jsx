/*
Auto-generated by: https://github.com/pmndrs/gltfjsx
*/

import React, { useEffect, useRef } from "react";
import { useGLTF, useAnimations } from "@react-three/drei";

export default function Robot(props) {
  const group = useRef();
  const { nodes, materials, animations } = useGLTF("/Robot.glb");
  const { actions } = useAnimations(animations, group);

  useEffect(() => {
    const action = actions.Robot_Idle;
    action.play();
  }, []);
  //   console.log(actions);

  return (
    <group ref={group} {...props} dispose={null}>
      <group name="Scene">
        <group name="RobotArmature">
          <group name="HandL">
            <skinnedMesh
              name="Cylinder022"
              geometry={nodes.Cylinder022.geometry}
              material={materials.Main}
              skeleton={nodes.Cylinder022.skeleton}
            />
            <skinnedMesh
              name="Cylinder022_1"
              geometry={nodes.Cylinder022_1.geometry}
              material={materials.Grey}
              skeleton={nodes.Cylinder022_1.skeleton}
            />
          </group>
          <group name="HandR">
            <skinnedMesh
              name="Cylinder015"
              geometry={nodes.Cylinder015.geometry}
              material={materials.Main}
              skeleton={nodes.Cylinder015.skeleton}
            />
            <skinnedMesh
              name="Cylinder015_1"
              geometry={nodes.Cylinder015_1.geometry}
              material={materials.Grey}
              skeleton={nodes.Cylinder015_1.skeleton}
            />
          </group>
          <primitive castShadow object={nodes.Bone} />
        </group>
      </group>
    </group>
  );
}

useGLTF.preload("/Robot.glb");