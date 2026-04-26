import React from "react";
import {
  AbsoluteFill,
  Audio,
  OffthreadVideo,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { type Scene } from "./schemas";

type Props = Scene & { videoSrc: string };

export const SceneComponent: React.FC<Props> = ({
  startSec,
  displayFrames,
  voiceoverPath,
  videoSrc,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scale = spring({
    frame,
    fps,
    from: 1.0,
    to: 1.06,
    config: { damping: 200 },
    durationInFrames: displayFrames,
  });

  const startFrame = Math.round(startSec * fps);

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      <AbsoluteFill
        style={{
          transform: `scale(${scale})`,
          transformOrigin: "center center",
        }}
      >
        <OffthreadVideo
          src={staticFile(videoSrc)}
          startFrom={startFrame}
          endAt={startFrame + displayFrames}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
          muted
        />
      </AbsoluteFill>

      {voiceoverPath && <Audio src={staticFile(voiceoverPath)} />}
    </AbsoluteFill>
  );
};
