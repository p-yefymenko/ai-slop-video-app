import { COIN_PACKAGES } from "@reelshort/shared";
import { Platform } from "react-native";
import { api } from "./client";

type PurchaseResult = { coinBalance: number };

async function loadIap() {
  try {
    return await import("react-native-iap");
  } catch {
    return null;
  }
}

export async function buyCoinPack(deviceId: string, productId: string): Promise<PurchaseResult> {
  if (Platform.OS !== "android") {
    throw new Error("Google Play Billing is Android-only in v1.");
  }

  const iap = await loadIap();
  if (!iap) {
    throw new Error("Play Billing native module is unavailable. Use a development or production build.");
  }

  await iap.initConnection();
  try {
    const products = await iap.fetchProducts({
      skus: COIN_PACKAGES.map((pack) => pack.productId),
      type: "in-app",
    });
    const found = products.some((product) => "id" in product && product.id === productId);
    if (!found) {
      throw new Error("Product is not available in Play Console yet.");
    }

    const purchase = await new Promise<{ purchaseToken?: string }>((resolve, reject) => {
      const success = iap.purchaseUpdatedListener((item) => {
        success.remove();
        failure.remove();
        resolve(item);
      });
      const failure = iap.purchaseErrorListener((error) => {
        success.remove();
        failure.remove();
        reject(error);
      });
      void iap.requestPurchase({
        type: "in-app",
        request: { android: { skus: [productId] } },
      });
    });

    if (!purchase.purchaseToken) {
      throw new Error("Purchase did not return a token.");
    }
    const verified = await api.verifyPurchase(deviceId, productId, purchase.purchaseToken);
    await iap.finishTransaction({ purchase: purchase as never, isConsumable: true });
    return { coinBalance: verified.coinBalance };
  } finally {
    await iap.endConnection();
  }
}

export async function buyDevPack(deviceId: string, productId: string): Promise<PurchaseResult> {
  const token = `dev:${productId}:${Date.now()}`;
  const verified = await api.verifyPurchase(deviceId, productId, token);
  return { coinBalance: verified.coinBalance };
}
