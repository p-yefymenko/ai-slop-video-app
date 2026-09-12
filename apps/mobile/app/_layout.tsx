import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { useEffect } from "react";
import { ActivityIndicator, View } from "react-native";
import { useSession } from "../stores/session";
import { useWallet } from "../stores/wallet";

export default function RootLayout() {
  const hydrate = useSession((state) => state.hydrate);
  const deviceId = useSession((state) => state.deviceId);
  const hydrated = useSession((state) => state.hydrated);
  const refresh = useWallet((state) => state.refresh);

  useEffect(() => {
    void hydrate();
  }, [hydrate]);

  useEffect(() => {
    if (deviceId) {
      void refresh(deviceId);
    }
  }, [deviceId, refresh]);

  if (!hydrated) {
    return (
      <View style={{ flex: 1, backgroundColor: "#050505", alignItems: "center", justifyContent: "center" }}>
        <ActivityIndicator color="#fff" />
      </View>
    );
  }

  return (
    <>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: "#050505" },
          headerTintColor: "#fff",
          contentStyle: { backgroundColor: "#050505" },
        }}
      >
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="series/[id]" options={{ title: "Series" }} />
        <Stack.Screen name="player/[seriesId]" options={{ headerShown: false, animation: "fade" }} />
        <Stack.Screen name="store" options={{ title: "Coin store", presentation: "modal" }} />
      </Stack>
    </>
  );
}
