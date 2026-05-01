import React from "react";
import { Composition } from "remotion";
import { MovieRecap } from "./MovieRecap";
import { StoryboardSchema, type Storyboard } from "./schemas";
import storyboardRaw from "./latest_storyboard.json";

const DEFAULT_FPS = 30;

const defaultStoryboard: Storyboard = StoryboardSchema.parse(storyboardRaw) as Storyboard;

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="MovieRecap"
      component={MovieRecap}
      durationInFrames={1}
      fps={DEFAULT_FPS}
      width={1920}
      height={1080}
      defaultProps={{ storyboard: defaultStoryboard }}
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
