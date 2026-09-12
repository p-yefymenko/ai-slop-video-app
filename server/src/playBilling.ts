import { COIN_PACKAGES } from "@reelshort/shared";
import type { Env } from "./env";

type ServiceAccount = {
  client_email: string;
  private_key: string;
  token_uri?: string;
};

function base64Url(input: string | ArrayBuffer): string {
  const bytes = typeof input === "string" ? new TextEncoder().encode(input) : new Uint8Array(input);
  let binary = "";
  for (const b of bytes) {
    binary += String.fromCharCode(b);
  }
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}

function pemToArrayBuffer(pem: string): ArrayBuffer {
  const cleaned = pem.replace("-----BEGIN PRIVATE KEY-----", "").replace("-----END PRIVATE KEY-----", "").replaceAll("\\n", "").replaceAll("\n", "").replaceAll(" ", "");
  const binary = atob(cleaned);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

async function getAccessToken(sa: ServiceAccount): Promise<string> {
  const now = Math.floor(Date.now() / 1000);
  const header = base64Url(JSON.stringify({ alg: "RS256", typ: "JWT" }));
  const claim = base64Url(
    JSON.stringify({
      iss: sa.client_email,
      scope: "https://www.googleapis.com/auth/androidpublisher",
      aud: sa.token_uri ?? "https://oauth2.googleapis.com/token",
      iat: now,
      exp: now + 3600,
    }),
  );
  const unsigned = `${header}.${claim}`;
  const key = await crypto.subtle.importKey(
    "pkcs8",
    pemToArrayBuffer(sa.private_key),
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, new TextEncoder().encode(unsigned));
  const jwt = `${unsigned}.${base64Url(signature)}`;

  const body = new URLSearchParams({
    grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
    assertion: jwt,
  });
  const res = await fetch(sa.token_uri ?? "https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!res.ok) {
    throw new Error(`Google token exchange failed: ${await res.text()}`);
  }
  const json = (await res.json()) as { access_token: string };
  return json.access_token;
}

export async function verifyPlayPurchase(
  env: Env,
  productId: string,
  purchaseToken: string,
): Promise<{ ok: boolean; coins: number; orderId: string; amountPaid: number | null; reason?: string }> {
  const pack = COIN_PACKAGES.find((item) => item.productId === productId);
  if (!pack) {
    return { ok: false, coins: 0, orderId: "", amountPaid: null, reason: "Unknown product" };
  }

  if (env.ENVIRONMENT !== "production" && purchaseToken.startsWith("dev:")) {
    return {
      ok: true,
      coins: pack.coins,
      orderId: purchaseToken,
      amountPaid: null,
    };
  }

  if (!env.GOOGLE_PLAY_SERVICE_ACCOUNT) {
    return { ok: false, coins: 0, orderId: "", amountPaid: null, reason: "Play billing is not configured" };
  }

  const sa = JSON.parse(env.GOOGLE_PLAY_SERVICE_ACCOUNT) as ServiceAccount;
  const accessToken = await getAccessToken(sa);
  const packageName = env.ANDROID_PACKAGE_NAME;
  const url = `https://androidpublisher.googleapis.com/androidpublisher/v3/applications/${packageName}/purchases/products/${productId}/tokens/${purchaseToken}`;
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${accessToken}` },
  });
  if (!res.ok) {
    return { ok: false, coins: 0, orderId: "", amountPaid: null, reason: await res.text() };
  }
  const payload = (await res.json()) as {
    purchaseState?: number;
    orderId?: string;
    consumptionState?: number;
  };
  if (payload.purchaseState !== 0) {
    return { ok: false, coins: 0, orderId: payload.orderId ?? "", amountPaid: null, reason: "Purchase not completed" };
  }

  await fetch(`${url}:acknowledge`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({}),
  });

  return {
    ok: true,
    coins: pack.coins,
    orderId: payload.orderId ?? purchaseToken,
    amountPaid: null,
  };
}
