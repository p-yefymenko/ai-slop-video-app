/** Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show. A chatbot can write this file from these comments. */

export type ShowCharacter = {
  /** Face and body landmarks only. Used for the identity still. Scene stills get that PNG, not this text. */
  promptBlock: string;
};

/** Shared set bible. Text only. Injected once into each scene still. Do not repeat this in imagePrompt. */
export type ShowLocation = {
  promptBlock: string;
};

/** Templates sent to Qwen/LTX. Include the `{placeholders}` or the rest of this JSON is ignored. */
export type ShowPrompts = {
  /** Identity still on a blank canvas. `{characterPromptBlock}` */
  characterImage: string;
  /**
   * Picture 1 (and Picture 2) are the identity stills in `characterIds` order.
   * `{characterIds}` `{locationPromptBlock}` `{imagePrompt}` — do not paste character bibles.
   */
  sceneStill: string;
  /** Motion and spoken lines only. The start PNG is the lock. `{videoPrompt}` */
  sceneVideo: string;
};

export type ScriptScene = {
  sceneNumber: number;
  locationId: string;
  /**
   * Faces that fill this still at the same size. At most two. Order is Qwen Picture 1, Picture 2.
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
