import type { Episode } from "@reelshort/shared";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useVideoPlayer, VideoView } from "expo-video";
import { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Dimensions,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  View,
  type ViewToken,
} from "react-native";
import { api } from "../../api/client";
import { useSession } from "../../stores/session";
import { useWallet } from "../../stores/wallet";

const { height } = Dimensions.get("window");

function EpisodePage({
  episode,
  active,
  deviceId,
}: {
  episode: Episode;
  active: boolean;
  deviceId: string;
}) {
  const router = useRouter();
  const refresh = useWallet((state) => state.refresh);
  const coinBalance = useWallet((state) => state.coinBalance);
  const [url, setUrl] = useState<string | null>(null);
  const [locked, setLocked] = useState(episode.locked);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const player = useVideoPlayer(null, (instance) => {
    instance.loop = true;
  });

  useEffect(() => {
    if (!active || locked) {
      player.pause();
      return;
    }
    let cancelled = false;
    api
      .play(deviceId, episode.id)
      .then((data) => {
        if (!cancelled) {
          setUrl(data.url);
          player.replace(data.url);
          player.play();
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Playback failed");
      });
    return () => {
      cancelled = true;
      player.pause();
    };
  }, [active, deviceId, episode.id, locked, player]);

  async function unlock() {
    setBusy(true);
    setError(null);
    try {
      const result = await api.unlock(deviceId, episode.id);
      if (result.coinBalance != null) {
        useWallet.getState().setBalance(result.coinBalance);
      }
      await refresh(deviceId);
      setLocked(false);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unlock failed";
      setError(message);
      if (message.toLowerCase().includes("not enough")) {
        router.push("/store");
      }
    } finally {
      setBusy(false);
    }
  }

  if (locked) {
    return (
      <View style={styles.page}>
        <Text style={styles.payTitle}>{episode.title}</Text>
        <Text style={styles.payBody}>Unlock this episode for {episode.coinCost} coins.</Text>
        <Text style={styles.payBody}>You have {coinBalance} coins.</Text>
        {error ? <Text style={styles.error}>{error}</Text> : null}
        <Pressable style={styles.button} onPress={() => void unlock()} disabled={busy}>
          <Text style={styles.buttonText}>{busy ? "Unlocking…" : "Unlock"}</Text>
        </Pressable>
        <Pressable onPress={() => router.push("/store")}>
          <Text style={styles.link}>Buy coins</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.page}>
      {url ? (
        <VideoView player={player} style={StyleSheet.absoluteFill} contentFit="cover" nativeControls={false} />
      ) : (
        <ActivityIndicator color="#fff" />
      )}
      <View style={styles.overlay}>
        <Text style={styles.epTitle}>{episode.title}</Text>
        {error ? <Text style={styles.error}>{error}</Text> : null}
      </View>
    </View>
  );
}

export default function PlayerScreen() {
  const { seriesId, episodeId } = useLocalSearchParams<{ seriesId: string; episodeId?: string }>();
  const deviceId = useSession((state) => state.deviceId);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [activeId, setActiveId] = useState<string | null>(episodeId ?? null);

  useEffect(() => {
    if (!deviceId || !seriesId) return;
    api.seriesDetail(deviceId, seriesId).then((data) => {
      setEpisodes(data.episodes);
      setActiveId((current) => current ?? data.episodes[0]?.id ?? null);
    });
  }, [deviceId, seriesId]);

  const initialIndex = useMemo(() => {
    const index = episodes.findIndex((item) => item.id === episodeId);
    return index >= 0 ? index : 0;
  }, [episodes, episodeId]);

  if (!deviceId || episodes.length === 0) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color="#fff" />
      </View>
    );
  }

  return (
    <FlatList
      data={episodes}
      keyExtractor={(item) => item.id}
      pagingEnabled
      initialScrollIndex={initialIndex}
      getItemLayout={(_, index) => ({ length: height, offset: height * index, index })}
      showsVerticalScrollIndicator={false}
      onViewableItemsChanged={({ viewableItems }: { viewableItems: ViewToken[] }) => {
        const first = viewableItems[0]?.item as Episode | undefined;
        if (first) setActiveId(first.id);
      }}
      viewabilityConfig={{ itemVisiblePercentThreshold: 80 }}
      renderItem={({ item }) => (
        <EpisodePage episode={item} active={item.id === activeId} deviceId={deviceId} />
      )}
    />
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, backgroundColor: "#000", alignItems: "center", justifyContent: "center" },
  page: {
    height,
    backgroundColor: "#000",
    alignItems: "center",
    justifyContent: "center",
  },
  overlay: { position: "absolute", left: 20, right: 20, bottom: 48 },
  epTitle: { color: "#fff", fontSize: 20, fontWeight: "700" },
  payTitle: { color: "#fff", fontSize: 28, fontWeight: "800", marginBottom: 12 },
  payBody: { color: "#ddd", fontSize: 16, marginBottom: 8 },
  button: { marginTop: 16, backgroundColor: "#fff", paddingHorizontal: 24, paddingVertical: 12, borderRadius: 999 },
  buttonText: { color: "#000", fontWeight: "700" },
  link: { color: "#fff", marginTop: 16, textDecorationLine: "underline" },
  error: { color: "#ffb4b4", marginTop: 8 },
});
