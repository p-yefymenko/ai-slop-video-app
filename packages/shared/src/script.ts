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

export type ShowPrompts = {
  characterImage: string;
  locationCharacter: string;
  sceneStill: string;
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
