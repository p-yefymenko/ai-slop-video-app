/** Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show. A chatbot can write this file from these comments. */

export type ShowCharacter = {
  /** Face and body landmarks for the identity still. Scene stills use that PNG, not this text. */
  promptBlock: string;
};

/** Materials and light on the people. Not a camera looking at a building or a room. */
export type ShowLocation = {
  promptBlock: string;
};

/** Templates sent to Qwen/LTX. Include the `{placeholders}` or the rest of this JSON is ignored. */
export type ShowPrompts = {
  /** Identity still on a blank canvas. `{characterPromptBlock}` */
  characterImage: string;
  /** Identity PNGs in `characterIds` order. `{characterIds}` `{locationPromptBlock}` `{imagePrompt}` */
  sceneStill: string;
  /** Same camera as the start PNG. Small motion. Spoken lines with how they sound. `{videoPrompt}` */
  sceneVideo: string;
};

export type ScriptScene = {
  sceneNumber: number;
  locationId: string;
  /**
   * At most two. Each appears once. Far, tiny, or over-the-shoulder is another scene.
   */
  characterIds: string[];
  /**
   * This camera, wardrobe, blocking, where they look. Faces at identity-still scale. One pose per listed person.
   * Set behind them. Nothing between the camera and a listed face.
   */
  imagePrompt: string;
  /**
   * Same camera as the start frame. Small motion. Spoken lines include how the voice sounds.
   * Do not restage, do not redescribe faces or the set, do not invent people who are not in the still.
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
