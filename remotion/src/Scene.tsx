import React from "react";
import {
  AbsoluteFill,
  Audio,
  OffthreadVideo,
  Series,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { type Scene, type SceneSegment } from "./schemas";

type Props = Scene & { videoSrc: string };

const VideoSegment: React.FC<{
  startSec: number;
  displayFrames: number;
  videoSrc: string;
}> = ({ startSec, displayFrames, videoSrc }) => {
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
  );
};

export const SceneComponent: React.FC<Props> = ({
  startSec,
  displayFrames,
  segments,
  voiceoverPath,
  videoSrc,
}) => {
  const segs: SceneSegment[] =
    segments && segments.length > 0
      ? segments
      : [{ startSec, displayFrames }];

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      <Series>
        {segs.map((seg, i) => (
          <Series.Sequence
            key={i}
            durationInFrames={Math.max(1, seg.displayFrames)}
            layout="none"
          >
            <VideoSegment
              startSec={seg.startSec}
              displayFrames={Math.max(1, seg.displayFrames)}
              videoSrc={videoSrc}
            />
          </Series.Sequence>
        ))}
      </Series>

      {voiceoverPath && <Audio src={staticFile(voiceoverPath)} />}
    </AbsoluteFill>
  );
};
