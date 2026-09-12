import { COIN_PACKAGES } from "@reelshort/shared";
import { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { buyCoinPack, buyDevPack } from "../api/billing";
import { useSession } from "../stores/session";
import { useWallet } from "../stores/wallet";

export default function StoreScreen() {
  const deviceId = useSession((state) => state.deviceId);
  const coinBalance = useWallet((state) => state.coinBalance);
  const setBalance = useWallet((state) => state.setBalance);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  async function buy(productId: string, dev: boolean) {
    if (!deviceId) return;
    setBusy(productId + (dev ? "-dev" : ""));
    setMessage(null);
    try {
      const result = dev ? await buyDevPack(deviceId, productId) : await buyCoinPack(deviceId, productId);
      setBalance(result.coinBalance);
      setMessage(`Balance is now ${result.coinBalance} coins.`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Purchase failed");
    } finally {
      setBusy(null);
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.balance}>You have {coinBalance} coins</Text>
      {COIN_PACKAGES.map((pack) => (
        <View key={pack.productId} style={styles.card}>
          <Text style={styles.label}>{pack.label}</Text>
          <Pressable
            style={styles.button}
            disabled={busy != null}
            onPress={() => void buy(pack.productId, false)}
          >
            <Text style={styles.buttonText}>
              {busy === pack.productId ? "…" : "Buy with Google Play"}
            </Text>
          </Pressable>
          <Pressable onPress={() => void buy(pack.productId, true)} disabled={busy != null}>
            <Text style={styles.dev}>
              {busy === `${pack.productId}-dev` ? "…" : "Dev grant (local only)"}
            </Text>
          </Pressable>
        </View>
      ))}
      {message ? <Text style={styles.message}>{message}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#050505", padding: 20 },
  balance: { color: "#fff", fontSize: 18, marginBottom: 16 },
  card: {
    backgroundColor: "#141414",
    padding: 16,
    borderRadius: 16,
    marginBottom: 12,
  },
  label: { color: "#fff", fontSize: 20, fontWeight: "700", marginBottom: 12 },
  button: { backgroundColor: "#fff", paddingVertical: 10, borderRadius: 999, alignItems: "center" },
  buttonText: { color: "#000", fontWeight: "700" },
  dev: { color: "#aaa", marginTop: 10, textAlign: "center" },
  message: { color: "#ddd", marginTop: 16 },
});
