import React from "react";
import { Composition } from "remotion";
import { MovieRecap } from "./MovieRecap";
import { StoryboardSchema, type Storyboard } from "./schemas";

const DEFAULT_FPS = 30;

const emptyStoryboard: Storyboard = {
  videoPath: "video.mp4",
  fps: DEFAULT_FPS,
  recapRatio: 0.15,
  scenes: [],
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="MovieRecap"
      component={MovieRecap}
      durationInFrames={1}
      fps={DEFAULT_FPS}
      width={1920}
      height={1080}
      defaultProps={{ storyboard: emptyStoryboard }}
      calculateMetadata={({ props }: { props: { storyboard: Storyboard } }) => {
        const storyboard = StoryboardSchema.parse(props.storyboard);
        const totalFrames = storyboard.scenes.reduce(
          (acc, s) => acc + s.displayFrames,
          0
        );
        return {
          durationInFrames: totalFrames || 1,
          fps: storyboard.fps,
        };
      }}
    />
  );
};
