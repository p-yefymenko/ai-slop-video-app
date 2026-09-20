/** Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show. */

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
  /** At most two people on camera. Extra portrait photos make Qwen collage. */
  characterIds: string[];
  /** Full shot: camera, wardrobe, blocking. People and room are generated together from text. */
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
