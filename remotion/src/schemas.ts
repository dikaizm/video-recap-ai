import { z } from "zod";

export const SceneSegmentSchema = z.object({
  startSec: z.number().nonnegative(),
  displayFrames: z.number().int().positive(),
  sourceWindow: z.number().int().optional(),
  phrase: z.string().optional(),
});

export const SceneSchema = z.object({
  window: z.number().int().positive(),
  startSec: z.number().nonnegative(),
  endSec: z.number().positive(),
  durationInFrames: z.number().int().positive(),
  displayFrames: z.number().int().positive(),
  povText: z.string(),
  dialogue: z.string(),
  voiceoverPath: z.string().optional(),
  startFmt: z.string(),
  segments: z.array(SceneSegmentSchema).optional(),
  isIntro: z.boolean().optional(),
});

export const StoryboardSchema = z.object({
  videoPath: z.string(),
  fps: z.number().int().positive().default(30),
  recapRatio: z.number().positive().default(0.15),
  scenes: z.array(SceneSchema),
});

export type SceneSegment = z.infer<typeof SceneSegmentSchema>;
export type Scene = z.infer<typeof SceneSchema>;
export type Storyboard = z.infer<typeof StoryboardSchema>;
