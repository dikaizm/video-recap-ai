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
      {storyboard.scenes.map((scene, index) => (
        <Series.Sequence
          key={scene.window}
          durationInFrames={scene.displayFrames}
          layout="none"
        >
          <SceneComponent
            {...scene}
            videoSrc={storyboard.videoPath}
            isFirstScene={index === 0}
          />
        </Series.Sequence>
      ))}
    </Series>
  );
};
