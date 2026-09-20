/** Authoring JSON for `content-pipeline/scripts_input/<id>.json`. One file per show. A chatbot can write this file from these comments. */

export type ShowCharacter = {
  /** Face and body landmarks only. Used for the identity still. Scene stills get that PNG, not this text. */
  promptBlock: string;
};

/**
 * Materials, architecture, and light of ONE place the camera can stand.
 * Not a viewpoint, not a building tour, not "in the distance", not guests looking at something off-frame.
 * Altar, aisle, and pews are three locations. A paragraph that names all of them is already a camera
 * (stock 9:16 wedding = people at the entrance, congregation facing an empty altar).
 */
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
   * Instruction must keep attached faces at portrait scale and forbid collage.
   */
  sceneStill: string;
  /** Motion and spoken lines only. The start PNG is the lock. `{videoPrompt}` */
  sceneVideo: string;
};

export type ScriptScene = {
  sceneNumber: number;
  locationId: string;
  /**
   * Faces that fill this still at roughly the same size as the identity portraits. At most two.
   * Order is Qwen Picture 1, Picture 2.
   * A person who is far, tiny, over a shoulder, or "too small to identify" is not listed — cut to another scene.
   * Qwen cannot copy a face that is a speck, and cannot invent a tighter crop than the portrait.
   */
  characterIds: string[];
  /**
   * This camera, wardrobe, blocking. Do not restate the location promptBlock or character bibles.
   * Named people are waist-up to head-and-shoulders, matching the identity stills.
   * Forbidden: a wide of the whole room, a shot from the far end, an extreme close-up tighter than the portraits.
   * "At the altar" means the altar is immediately behind or beside them, filling the background — not seen from the church door.
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
