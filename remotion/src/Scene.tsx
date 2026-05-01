import React from "react";
import {
  AbsoluteFill,
  Audio,
  Img,
  interpolate,
  OffthreadVideo,
  Series,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { type Scene, type SceneSegment } from "./schemas";
import { GreetingCard } from "./GreetingCard";

type Props = Scene & { videoSrc: string; isFirstScene?: boolean };

const LogoOverlay: React.FC<{ isFirstScene?: boolean }> = ({ isFirstScene }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Only show logo on first scene
  if (!isFirstScene) {
    return null;
  }

  // Fade out after 15 seconds (only for first scene)
  const fadeOutStartFrame = 15 * fps; // 15 seconds
  const fadeOutEndFrame = fadeOutStartFrame + fps; // Fade over 1 second

  const opacity = interpolate(
    frame,
    [fadeOutStartFrame, fadeOutEndFrame],
    [0.95, 0],
    {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }
  );

  // Hide completely after fade
  if (frame > fadeOutEndFrame) {
    return null;
  }

  return (
    <div
      style={{
        position: "absolute",
        top: 24,
        left: 24,
        zIndex: 100,
        opacity,
      }}
    >
      <Img
        src={staticFile("Logo_PremiereRoll_circle.png")}
        style={{
          width: 140,
          height: "auto",
        }}
      />
    </div>
  );
};

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
  isFirstScene,
  isGreeting,
  channelName,
}) => {
  // Greeting beat: branded title card, no source video
  if (isGreeting) {
    return (
      <GreetingCard
        channelName={channelName ?? "Premiere Roll"}
        voiceoverPath={voiceoverPath}
        displayFrames={displayFrames}
      />
    );
  }

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

      {/* PR Logo overlay - top left corner */}
      <LogoOverlay isFirstScene={isFirstScene} />
    </AbsoluteFill>
  );
};
