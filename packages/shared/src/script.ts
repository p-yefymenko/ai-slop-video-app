/** Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show. A chatbot can write this file from these comments. */

export type ShowCharacter = {
  promptBlock: string;
  imagePrompt: string;
};

/** Shared set bible. Text only — do not generate or attach a location PNG. */
export type ShowLocation = {
  promptBlock: string;
  preserve: string;
};

/** Templates sent to Qwen/LTX. Include the `{placeholders}` or the rest of this JSON is ignored. */
export type ShowPrompts = {
  /** `{characterPromptBlock}` `{imagePrompt}` */
  characterImage: string;
  /** Picture 1 is blank; no portraits or location stills. `{characterPromptBlocks}` `{locationPromptBlock}` `{imagePrompt}` */
  sceneStill: string;
  /** `{characterPromptBlocks}` `{preserve}` `{videoPrompt}` */
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
   * One camera, one scale. Wardrobe and blocking for characterIds only.
   * Illegal: tiny/far background, over-the-shoulder, hand in the foreground, close-up of one person while listing two.
   */
  imagePrompt: string;
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
