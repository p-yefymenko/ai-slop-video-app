import { and, eq } from "drizzle-orm";
import { Hono } from "hono";
import { getDb } from "../db/client";
import { episodes } from "../db/schema";
import type { Env } from "../env";
import { requireAdmin } from "../middleware/auth";

export const adminEpisodeRoutes = new Hono<{ Bindings: Env }>();

function isAuthorized(env: Env, header: string | undefined) {
  const secret = env.ADMIN_SECRET ?? (env.ENVIRONMENT === "production" ? undefined : "dev-admin-secret");
  return requireAdmin(secret, header);
}

adminEpisodeRoutes.post("/admin/episodes", async (c) => {
  if (!isAuthorized(c.env, c.req.header("Authorization"))) {
    return c.json({ error: "Unauthorized" }, 401);
  }
  const body = await c.req.json<{
    id?: string;
    seriesId: string;
    order: number;
    title: string;
    videoUrl: string;
    thumbnailUrl?: string;
    coinCost?: number;
    isFree?: boolean;
  }>();
  if (!body.seriesId || !body.title || !body.videoUrl || body.order == null) {
    return c.json({ error: "seriesId, order, title, and videoUrl are required" }, 400);
  }
  const db = getDb(c.env);
  const id = body.id ?? crypto.randomUUID();
  const existing = await db
    .select()
    .from(episodes)
    .where(and(eq(episodes.seriesId, body.seriesId), eq(episodes.order, body.order)))
    .get();
  if (existing) {
    await db
      .update(episodes)
      .set({
        title: body.title,
        videoUrl: body.videoUrl,
        thumbnailUrl: body.thumbnailUrl ?? existing.thumbnailUrl,
        coinCost: body.coinCost ?? existing.coinCost,
        isFree: body.isFree ?? existing.isFree,
      })
      .where(eq(episodes.id, existing.id));
    return c.json({ id: existing.id, updated: true });
  }
  await db.insert(episodes).values({
    id,
    seriesId: body.seriesId,
    order: body.order,
    title: body.title,
    videoUrl: body.videoUrl,
    thumbnailUrl: body.thumbnailUrl ?? null,
    coinCost: body.coinCost ?? 0,
    isFree: body.isFree ?? false,
  });
  return c.json({ id, created: true });
});
