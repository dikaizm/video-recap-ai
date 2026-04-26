import React from "react";
import { interpolate, useCurrentFrame } from "remotion";

type Props = {
  dialogue?: string;
  displayFrames: number;
};

export const TextOverlay: React.FC<Props> = ({ dialogue, displayFrames }) => {
  const frame = useCurrentFrame();

  const opacity = interpolate(
    frame,
    [0, 12, displayFrames - 12, displayFrames],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <>
      {dialogue && (
        <div
          style={{
            position: "absolute",
            bottom: 32,
            left: "10%",
            right: "10%",
            background: "rgba(0,0,0,0.6)",
            borderRadius: 4,
            padding: "8px 16px",
            color: "#ffd700",
            fontFamily: "Arial, sans-serif",
            fontSize: 18,
            textAlign: "center",
            opacity,
          }}
        >
          {dialogue}
        </div>
      )}
    </>
  );
};
