import { useRouter } from "expo-router";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { useWallet } from "../../stores/wallet";

export default function ProfileScreen() {
  const router = useRouter();
  const coinBalance = useWallet((state) => state.coinBalance);
  const unlockedEpisodeIds = useWallet((state) => state.unlockedEpisodeIds);

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Coins</Text>
      <Text style={styles.balance}>{coinBalance}</Text>
      <Text style={styles.sub}>{unlockedEpisodeIds.length} paid episodes unlocked</Text>
      <Pressable style={styles.button} onPress={() => router.push("/store")}>
        <Text style={styles.buttonText}>Buy coins</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#050505", padding: 24, paddingTop: 72 },
  label: { color: "#888", fontSize: 14, textTransform: "uppercase", letterSpacing: 1 },
  balance: { color: "#fff", fontSize: 56, fontWeight: "800", marginTop: 8 },
  sub: { color: "#aaa", marginTop: 8 },
  button: {
    marginTop: 32,
    backgroundColor: "#fff",
    alignSelf: "flex-start",
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 999,
  },
  buttonText: { color: "#000", fontWeight: "700" },
});
