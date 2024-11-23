import { Html, useProgress } from "@react-three/drei";

export default function Loading() {
  const { progress } = useProgress();
  return (
    <Html center>
      <div style={{ color: "white", fontSize: "1.5em" }}>
        Loading... {progress.toFixed(2)}%
      </div>
    </Html>
  );
}
