import type { Env } from "./env";

const encoder = new TextEncoder();

async function hmacSign(secret: string, payload: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(payload));
  return bufferToBase64Url(signature);
}

async function hmacVerify(secret: string, payload: string, signature: string): Promise<boolean> {
  const expected = await hmacSign(secret, payload);
  return expected === signature;
}

function bufferToBase64Url(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (const b of bytes) {
    binary += String.fromCharCode(b);
  }
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}

export function isRemoteObjectKey(videoUrl: string | null): boolean {
  if (!videoUrl) return false;
  return !videoUrl.startsWith("http://") && !videoUrl.startsWith("https://");
}

export async function createPlaybackUrl(
  env: Env,
  requestUrl: string,
  objectKey: string,
  ttlSeconds = 3600,
): Promise<{ url: string; expiresAt: number }> {
  const secret = env.MEDIA_SIGNING_SECRET ?? env.ADMIN_SECRET ?? "dev-media-signing-secret";
  const expiresAt = Math.floor(Date.now() / 1000) + ttlSeconds;
  const payload = `${objectKey}:${expiresAt}`;
  const token = await hmacSign(secret, payload);
  const origin = new URL(requestUrl).origin;
  const url = `${origin}/media/${encodeURIComponent(objectKey)}?exp=${expiresAt}&token=${token}`;
  return { url, expiresAt: expiresAt * 1000 };
}

export async function verifyPlaybackToken(env: Env, objectKey: string, exp: string | null, token: string | null) {
  if (!exp || !token) return false;
  const expiresAt = Number(exp);
  if (!Number.isFinite(expiresAt) || expiresAt < Math.floor(Date.now() / 1000)) {
    return false;
  }
  const secret = env.MEDIA_SIGNING_SECRET ?? env.ADMIN_SECRET ?? "dev-media-signing-secret";
  return hmacVerify(secret, `${objectKey}:${expiresAt}`, token);
}

export async function putObject(env: Env, key: string, body: ArrayBuffer, contentType: string) {
  await env.VIDEO_BUCKET.put(key, body, {
    httpMetadata: { contentType },
  });
  return key;
}
