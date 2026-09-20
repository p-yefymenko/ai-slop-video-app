/** Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show. A chatbot can write this file from these comments. */

export type ShowCharacter = {
  /** Face and body landmarks only. Used for the identity still, not injected into scene stills or video. */
  promptBlock: string;
};

/** Shared set bible. Text only. Injected once into each scene still. Do not repeat this in imagePrompt. */
export type ShowLocation = {
  promptBlock: string;
};

/** Templates sent to Qwen/LTX. Include the `{placeholders}` or the rest of this JSON is ignored. */
export type ShowPrompts = {
  /** Identity still. `{characterPromptBlock}` */
  characterImage: string;
  /** Picture 1 is blank. `{locationPromptBlock}` `{imagePrompt}` — do not also paste character bibles. */
  sceneStill: string;
  /** Motion and spoken lines only. The start PNG is the lock. `{videoPrompt}` */
  sceneVideo: string;
};

export type ScriptScene = {
  sceneNumber: number;
  locationId: string;
  /**
   * Faces that fill this still at the same size. At most two.
   * Not everyone in the scene: a speck, a blur, or someone over a shoulder is not a characterId — cut to another scene.
   */
  characterIds: string[];
  /**
   * This camera, wardrobe, blocking. Do not restate the location promptBlock or character bibles.
   * One camera, one scale.
   */
  imagePrompt: string;
  /** Camera move, action, spoken lines. Do not redescribe faces or the room. */
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
