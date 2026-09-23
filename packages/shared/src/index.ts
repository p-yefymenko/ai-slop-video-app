export type Series = {
  id: string;
  title: string;
  description: string;
  coverImageUrl: string;
  isPublished: boolean;
};

export type Episode = {
  id: string;
  seriesId: string;
  order: number;
  title: string;
  videoUrl: string | null;
  thumbnailUrl: string | null;
  coinCost: number;
  isFree: boolean;
  locked: boolean;
};

export type User = {
  id: string;
  googlePlayAccountId: string | null;
  deviceId: string | null;
  coinBalance: number;
  createdAt: string;
};

export type Purchase = {
  id: string;
  userId: string;
  playOrderId: string;
  coinsGranted: number;
  amountPaid: number | null;
  verifiedAt: string | null;
  status: "pending" | "verified" | "failed";
};

export type Wallet = {
  userId: string;
  coinBalance: number;
};

export type PlayResponse = {
  url: string;
  expiresAt: number;
};

export type UnlockedEpisode = {
  userId: string;
  episodeId: string;
  unlockedAt: string;
};

export type CoinPackage = {
  productId: string;
  coins: number;
  label: string;
};

export const COIN_PACKAGES: CoinPackage[] = [
  { productId: "coins_100", coins: 100, label: "100 coins" },
  { productId: "coins_500", coins: 500, label: "500 coins" },
  { productId: "coins_1200", coins: 1200, label: "1,200 coins" },
];

export type {
  AssetNeed,
  CharacterProxy,
  ScriptScene,
  ShowCharacter,
  ShowEpisode,
  ShowLocation,
  ShowProp,
  ShowScript,
  SpatialCamera,
  SpatialCameraKeyframe,
  SpatialTimeline,
  StageGeometry,
  StageLandmark,
  Vec3,
} from "./script";
