import { create } from "zustand";
import { api } from "../api/client";

type WalletState = {
  coinBalance: number;
  unlockedEpisodeIds: string[];
  refresh: (deviceId: string) => Promise<void>;
  setBalance: (coinBalance: number) => void;
};

export const useWallet = create<WalletState>((set) => ({
  coinBalance: 0,
  unlockedEpisodeIds: [],
  setBalance: (coinBalance) => set({ coinBalance }),
  refresh: async (deviceId: string) => {
    const me = await api.me(deviceId);
    set({ coinBalance: me.user.coinBalance, unlockedEpisodeIds: me.unlockedEpisodeIds });
  },
}));
