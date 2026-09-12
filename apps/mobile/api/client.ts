import { Platform } from "react-native";
import type { Episode, PlayResponse, Series, Wallet } from "@reelshort/shared";

const DEFAULT_API =
  Platform.OS === "android" ? "http://10.0.2.2:8787" : "http://127.0.0.1:8787";

export const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL ?? DEFAULT_API;

type Options = {
  method?: string;
  body?: unknown;
  deviceId: string;
};

async function request<T>(path: string, options: Options): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: options.method ?? "GET",
    headers: {
      "Content-Type": "application/json",
      "X-Device-Id": options.deviceId,
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const json = (await res.json()) as T & { error?: string };
  if (!res.ok) {
    throw new Error(json.error ?? `Request failed: ${res.status}`);
  }
  return json;
}

export const api = {
  listSeries: (deviceId: string) =>
    request<{ series: Series[] }>("/series", { deviceId }),
  seriesDetail: (deviceId: string, id: string) =>
    request<{ series: Series; episodes: Episode[] }>(`/series/${id}/episodes`, { deviceId }),
  play: (deviceId: string, episodeId: string) =>
    request<PlayResponse>(`/episodes/${episodeId}/play`, { deviceId }),
  unlock: (deviceId: string, episodeId: string) =>
    request<{ ok: boolean; coinBalance?: number }>(`/episodes/${episodeId}/unlock`, {
      deviceId,
      method: "POST",
    }),
  wallet: (deviceId: string) => request<Wallet>("/users/me/wallet", { deviceId }),
  me: (deviceId: string) =>
    request<{ user: { id: string; coinBalance: number }; unlockedEpisodeIds: string[] }>(
      "/users/me",
      { deviceId },
    ),
  verifyPurchase: (deviceId: string, productId: string, purchaseToken: string) =>
    request<{ ok: boolean; coinBalance: number; coinsGranted?: number }>("/purchases/verify", {
      deviceId,
      method: "POST",
      body: { productId, purchaseToken },
    }),
};
