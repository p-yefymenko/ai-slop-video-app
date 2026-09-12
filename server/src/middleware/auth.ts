import { eq } from "drizzle-orm";
import type { Context, Next } from "hono";
import { users } from "../db/schema";
import { getDb } from "../db/client";
import type { Env } from "../env";

export async function deviceAuth(c: Context<{ Bindings: Env; Variables: { userId: string } }>, next: Next) {
  const deviceId = c.req.header("X-Device-Id");
  if (!deviceId) {
    return c.json({ error: "X-Device-Id header is required" }, 401);
  }

  const db = getDb(c.env);
  const existing = await db.select().from(users).where(eq(users.deviceId, deviceId)).get();
  if (existing) {
    c.set("userId", existing.id);
    await next();
    return;
  }

  const id = crypto.randomUUID();
  await db.insert(users).values({
    id,
    deviceId,
    coinBalance: 200,
    createdAt: new Date(),
  });
  c.set("userId", id);
  await next();
}

export function requireAdmin(secret: string | undefined, header: string | undefined) {
  if (!secret) {
    return false;
  }
  const token = header?.startsWith("Bearer ") ? header.slice(7) : header;
  return Boolean(token && token === secret);
}
