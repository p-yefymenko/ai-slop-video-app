import type { Series } from "@reelshort/shared";
import { useFocusEffect, useRouter } from "expo-router";
import { useCallback, useState } from "react";
import {
  ActivityIndicator,
  Dimensions,
  FlatList,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { api } from "../../api/client";
import { useSession } from "../../stores/session";

const { height } = Dimensions.get("window");

export default function FeedScreen() {
  const router = useRouter();
  const deviceId = useSession((state) => state.deviceId);
  const [series, setSeries] = useState<Series[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    if (!deviceId) return;
    setLoading(true);
    try {
      const data = await api.listSeries(deviceId);
      setSeries(data.series);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load feed");
    } finally {
      setLoading(false);
    }
  }, [deviceId]);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load]),
  );

  if (loading && series.length === 0) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color="#fff" />
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.center}>
        <Text style={styles.error}>{error}</Text>
        <Pressable onPress={() => void load()} style={styles.retry}>
          <Text style={styles.retryText}>Retry</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <FlatList
      data={series}
      keyExtractor={(item) => item.id}
      pagingEnabled
      showsVerticalScrollIndicator={false}
      renderItem={({ item }) => (
        <Pressable style={styles.page} onPress={() => router.push(`/series/${item.id}`)}>
          <Image source={{ uri: item.coverImageUrl }} style={StyleSheet.absoluteFill} />
          <View style={styles.scrim} />
          <View style={styles.meta}>
            <Text style={styles.title}>{item.title}</Text>
            <Text style={styles.description}>{item.description}</Text>
            <Text style={styles.cta}>Watch series</Text>
          </View>
        </Pressable>
      )}
    />
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, backgroundColor: "#050505", alignItems: "center", justifyContent: "center", padding: 24 },
  page: { height, backgroundColor: "#111", justifyContent: "flex-end" },
  scrim: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(0,0,0,0.35)" },
  meta: { padding: 24, paddingBottom: 96 },
  title: { color: "#fff", fontSize: 32, fontWeight: "700" },
  description: { color: "#ddd", marginTop: 8, fontSize: 16 },
  cta: { color: "#fff", marginTop: 16, fontWeight: "600" },
  error: { color: "#fff", textAlign: "center", marginBottom: 16 },
  retry: { backgroundColor: "#fff", paddingHorizontal: 16, paddingVertical: 10, borderRadius: 999 },
  retryText: { color: "#000", fontWeight: "700" },
});
