import React from "react";
import {
  AbsoluteFill,
  Audio,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

interface GreetingCardProps {
  channelName: string;
  voiceoverPath?: string;
  displayFrames: number;
}

export const GreetingCard: React.FC<GreetingCardProps> = ({
  channelName,
  voiceoverPath,
  displayFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Channel name slides up and fades in
  const nameProgress = spring({
    frame,
    fps,
    from: 0,
    to: 1,
    config: { damping: 18, stiffness: 60 },
    durationInFrames: Math.round(fps * 0.9),
  });

  // Tagline fades in slightly after the name
  const tagDelay = Math.round(fps * 0.4);
  const tagProgress = spring({
    frame: Math.max(0, frame - tagDelay),
    fps,
    from: 0,
    to: 1,
    config: { damping: 22, stiffness: 55 },
    durationInFrames: Math.round(fps * 0.8),
  });

  // Subtle underline width expansion
  const lineWidth = interpolate(nameProgress, [0, 1], [0, 100]);

  // Fade out near end
  const fadeOutStart = displayFrames - Math.round(fps * 0.6);
  const opacity = interpolate(
    frame,
    [fadeOutStart, displayFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const nameY = interpolate(nameProgress, [0, 1], [32, 0]);
  const nameOpacity = nameProgress;
  const tagOpacity = tagProgress;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#0c0c0c",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        opacity,
      }}
    >
      {/* Subtle top accent line */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 3,
          background: "linear-gradient(90deg, transparent 0%, #c9a84c 40%, #e8c96a 60%, transparent 100%)",
          opacity: nameProgress,
        }}
      />

      {/* Channel name */}
      <div
        style={{
          transform: `translateY(${nameY}px)`,
          opacity: nameOpacity,
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontFamily: "'Georgia', 'Times New Roman', serif",
            fontSize: 72,
            fontWeight: 700,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "#f0e8d0",
            textShadow: "0 2px 24px rgba(200,168,76,0.35)",
          }}
        >
          {channelName.toUpperCase()}
        </div>

        {/* Gold underline */}
        <div
          style={{
            margin: "10px auto 0",
            height: 2,
            width: `${lineWidth}%`,
            background: "linear-gradient(90deg, transparent, #c9a84c, transparent)",
          }}
        />
      </div>

      {/* Tagline */}
      <div
        style={{
          marginTop: 24,
          opacity: tagOpacity,
          fontFamily: "'Georgia', 'Times New Roman', serif",
          fontSize: 22,
          letterSpacing: "0.12em",
          color: "#a09070",
          fontStyle: "italic",
        }}
      >
        every story, perfectly framed
      </div>

      {/* Subtle bottom accent */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: 3,
          background: "linear-gradient(90deg, transparent 0%, #c9a84c 40%, #e8c96a 60%, transparent 100%)",
          opacity: nameProgress,
        }}
      />

      {voiceoverPath && <Audio src={staticFile(voiceoverPath)} />}
    </AbsoluteFill>
  );
};
