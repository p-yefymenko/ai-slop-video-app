/** Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show. A chatbot can write this file from these comments. */

export type ShowCharacter = {
  /** Face and body for the identity still. Scene stills attach that PNG, not this text. */
  promptBlock: string;
};

/** Short environment clause for the Qwen edit — where they are. Not a camera. */
export type ShowLocation = {
  promptBlock: string;
};

/** Templates sent to Qwen/LTX. Include the `{placeholders}` or the rest of this JSON is ignored. */
export type ShowPrompts = {
  /** Identity still on a blank canvas. `{characterPromptBlock}` */
  characterImage: string;
  /**
   * Qwen-Edit-2511: name Picture 1 / Picture 2, who is where, facing whom, in this place.
   * `{characterIds}` `{locationPromptBlock}` `{imagePrompt}`
   */
  sceneStill: string;
  /**
   * LTX I2V: from this image, what happens — action, camera, audio. Do not redescribe the frame.
   * `{videoPrompt}`
   */
  sceneVideo: string;
};

export type ScriptScene = {
  sceneNumber: number;
  locationId: string;
  /** At most two. Qwen Picture 1, then Picture 2. */
  characterIds: string[];
  /** Who is where, facing whom, wardrobe. Identity is the PNG. */
  imagePrompt: string;
  /**
   * What happens next: action, camera, audio.
   * Spoken lines in quotes with voice quality; one acting beat (pause, gaze, breath) between phrases.
   * Audio at the end. Do not redescribe the start frame.
   */
  videoPrompt: string;
  durationSeconds: number;
};

export type ShowEpisode = {
  episodeNumber: number;
  title: string;
  isFree: boolean;
  coinCost: number;
  scenes: ScriptScene[];
};

export type ShowScript = {
  id: string;
  title: string;
  characters: Record<string, ShowCharacter>;
  locations: Record<string, ShowLocation>;
  prompts: ShowPrompts;
  episodes: ShowEpisode[];
};
