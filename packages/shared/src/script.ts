/** Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show. */

export type ShowCharacter = {
  promptBlock: string;
  imagePrompt: string;
};

export type LocationCharacter = {
  characterIds: string[];
  promptBlock: string;
  preserve: string;
  platePrompt: string;
};

/** Templates sent to Qwen/LTX. Include the `{placeholders}` or the rest of this JSON is ignored. */
export type ShowPrompts = {
  /** `{characterPromptBlock}` `{imagePrompt}` */
  characterImage: string;
  /** `{characterPromptBlocks}` `{locationPromptBlock}` `{platePrompt}` */
  locationCharacter: string;
  /** `{characterPromptBlocks}` `{preserve}` `{imagePrompt}` */
  sceneStill: string;
  /** `{characterPromptBlocks}` `{preserve}` `{videoPrompt}` */
  sceneVideo: string;
};

export type ScriptScene = {
  sceneNumber: number;
  locationCharacterId: string;
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
  locationCharacters: Record<string, LocationCharacter>;
  prompts: ShowPrompts;
  episodes: ShowEpisode[];
};
