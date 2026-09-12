import * as SecureStore from "expo-secure-store";
import { create } from "zustand";

const DEVICE_KEY = "device-id";

function randomId() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (char) => {
    const r = (Math.random() * 16) | 0;
    const v = char === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

type SessionState = {
  deviceId: string | null;
  hydrated: boolean;
  hydrate: () => Promise<void>;
};

export const useSession = create<SessionState>((set, get) => ({
  deviceId: null,
  hydrated: false,
  hydrate: async () => {
    if (get().hydrated && get().deviceId) return;
    let deviceId = await SecureStore.getItemAsync(DEVICE_KEY);
    if (!deviceId) {
      deviceId = randomId();
      await SecureStore.setItemAsync(DEVICE_KEY, deviceId);
    }
    set({ deviceId, hydrated: true });
  },
}));
