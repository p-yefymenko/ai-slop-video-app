/** Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show. A chatbot can write this file from these comments. */

export type ShowCharacter = {
  /** Face and body landmarks for the identity still. Scene stills use that PNG, not this text. */
  promptBlock: string;
};

/** Materials and light next to the people. Not a viewpoint. Not a space you look down. */
export type ShowLocation = {
  promptBlock: string;
};

/** Templates sent to Qwen/LTX. Include the `{placeholders}` or the rest of this JSON is ignored. */
export type ShowPrompts = {
  /** Identity still on a blank canvas. `{characterPromptBlock}` */
  characterImage: string;
  /** Identity PNGs in `characterIds` order. `{characterIds}` `{locationPromptBlock}` `{imagePrompt}` */
  sceneStill: string;
  /** Motion and spoken lines only. The start PNG is the lock. `{videoPrompt}` */
  sceneVideo: string;
};

export type ScriptScene = {
  sceneNumber: number;
  locationId: string;
  /**
   * At most two. Faces fill the frame at identity-still scale, Picture 1 then Picture 2.
   * Far, tiny, or over-the-shoulder is a different scene.
   */
  characterIds: string[];
  /**
   * This camera, wardrobe, blocking. People waist-up to head-and-shoulders.
   * The set is immediately around them. No room-wide, no far-end camera, no crop tighter than the portraits.
   */
  imagePrompt: string;
  /** Camera move, action, spoken lines. Do not redescribe faces or the set. */
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
