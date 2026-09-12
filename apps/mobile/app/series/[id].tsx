import type { Episode, Series } from "@reelshort/shared";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { api } from "../../api/client";
import { useSession } from "../../stores/session";

export default function SeriesDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const deviceId = useSession((state) => state.deviceId);
  const [series, setSeries] = useState<Series | null>(null);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!deviceId || !id) return;
    api
      .seriesDetail(deviceId, id)
      .then((data) => {
        setSeries(data.series);
        setEpisodes(data.episodes);
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to load series"));
  }, [deviceId, id]);

  if (error) {
    return (
      <View style={styles.center}>
        <Text style={styles.error}>{error}</Text>
      </View>
    );
  }

  if (!series) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color="#fff" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Image source={{ uri: series.coverImageUrl }} style={styles.cover} />
      <Text style={styles.title}>{series.title}</Text>
      <Text style={styles.description}>{series.description}</Text>
      <FlatList
        data={episodes}
        keyExtractor={(item) => item.id}
        contentContainerStyle={{ paddingBottom: 48 }}
        renderItem={({ item }) => (
          <Pressable
            style={styles.row}
            onPress={() =>
              router.push({
                pathname: "/player/[seriesId]",
                params: { seriesId: series.id, episodeId: item.id },
              })
            }
          >
            <View>
              <Text style={styles.epTitle}>
                {item.order}. {item.title}
              </Text>
              <Text style={styles.meta}>
                {item.locked ? `Locked · ${item.coinCost} coins` : item.isFree ? "Free" : "Unlocked"}
              </Text>
            </View>
            <Text style={styles.lock}>{item.locked ? "🔒" : "▶"}</Text>
          </Pressable>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#050505", padding: 16 },
  center: { flex: 1, backgroundColor: "#050505", alignItems: "center", justifyContent: "center" },
  cover: { width: "100%", height: 220, borderRadius: 12, backgroundColor: "#111" },
  title: { color: "#fff", fontSize: 28, fontWeight: "800", marginTop: 16 },
  description: { color: "#bbb", marginTop: 8, marginBottom: 16 },
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 14,
    borderBottomColor: "#222",
    borderBottomWidth: 1,
  },
  epTitle: { color: "#fff", fontSize: 16, fontWeight: "600" },
  meta: { color: "#888", marginTop: 4 },
  lock: { color: "#fff", fontSize: 18 },
  error: { color: "#fff" },
});
