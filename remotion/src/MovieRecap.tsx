import React from "react";
import { Series } from "remotion";
import { type Storyboard } from "./schemas";
import { SceneComponent } from "./Scene";

type Props = {
  storyboard: Storyboard;
};

export const MovieRecap: React.FC<Props> = ({ storyboard }) => {
  return (
    <Series>
      {storyboard.scenes.map((scene) => (
        <Series.Sequence
          key={scene.window}
          durationInFrames={scene.displayFrames}
          layout="none"
        >
          <SceneComponent {...scene} videoSrc={storyboard.videoPath} />
        </Series.Sequence>
      ))}
    </Series>
  );
};
