import { and, asc, eq } from "drizzle-orm";
import { Hono } from "hono";
import { getDb } from "../db/client";
import { episodes, series, unlockedEpisodes, users } from "../db/schema";
import type { AppVariables, Env } from "../env";
import { createPlaybackUrl, isRemoteObjectKey } from "../storage";

export const episodeRoutes = new Hono<{ Bindings: Env; Variables: AppVariables }>();

episodeRoutes.get("/series/:id/episodes", async (c) => {
  const seriesId = c.req.param("id");
  const userId = c.get("userId");
  const db = getDb(c.env);

  const found = await db.select().from(series).where(eq(series.id, seriesId)).get();
  if (!found) {
    return c.json({ error: "Series not found" }, 404);
  }

  const episodeRows = await db
    .select()
    .from(episodes)
    .where(eq(episodes.seriesId, seriesId))
    .orderBy(asc(episodes.order))
    .all();

  const unlocked = await db
    .select()
    .from(unlockedEpisodes)
    .where(eq(unlockedEpisodes.userId, userId))
    .all();
  const unlockedIds = new Set(unlocked.map((row) => row.episodeId));

  return c.json({
    series: {
      id: found.id,
      title: found.title,
      description: found.description,
      coverImageUrl: found.coverImageUrl,
      isPublished: found.isPublished,
    },
    episodes: episodeRows.map((row) => ({
      id: row.id,
      seriesId: row.seriesId,
      order: row.order,
      title: row.title,
      videoUrl: row.videoUrl,
      thumbnailUrl: row.thumbnailUrl,
      coinCost: row.coinCost,
      isFree: row.isFree,
      locked: !(row.isFree || unlockedIds.has(row.id)),
    })),
  });
});

episodeRoutes.get("/episodes/:id/play", async (c) => {
  const episodeId = c.req.param("id");
  const userId = c.get("userId");
  const db = getDb(c.env);
  const episode = await db.select().from(episodes).where(eq(episodes.id, episodeId)).get();
  if (!episode || !episode.videoUrl) {
    return c.json({ error: "Episode not found" }, 404);
  }

  if (!episode.isFree) {
    const unlock = await db
      .select()
      .from(unlockedEpisodes)
      .where(and(eq(unlockedEpisodes.userId, userId), eq(unlockedEpisodes.episodeId, episodeId)))
      .get();
    if (!unlock) {
      return c.json({ error: "Episode is locked", coinCost: episode.coinCost }, 402);
    }
  }

  if (!isRemoteObjectKey(episode.videoUrl)) {
    return c.json({ url: episode.videoUrl, expiresAt: Date.now() + 24 * 60 * 60 * 1000 });
  }

  const playback = await createPlaybackUrl(c.env, c.req.url, episode.videoUrl);
  return c.json(playback);
});

episodeRoutes.post("/episodes/:id/unlock", async (c) => {
  const episodeId = c.req.param("id");
  const userId = c.get("userId");
  const db = getDb(c.env);
  const episode = await db.select().from(episodes).where(eq(episodes.id, episodeId)).get();
  if (!episode) {
    return c.json({ error: "Episode not found" }, 404);
  }
  if (episode.isFree) {
    return c.json({ ok: true, alreadyFree: true });
  }

  const existing = await db
    .select()
    .from(unlockedEpisodes)
    .where(and(eq(unlockedEpisodes.userId, userId), eq(unlockedEpisodes.episodeId, episodeId)))
    .get();
  if (existing) {
    return c.json({ ok: true, alreadyUnlocked: true });
  }

  const user = await db.select().from(users).where(eq(users.id, userId)).get();
  if (!user) {
    return c.json({ error: "User not found" }, 404);
  }
  if (user.coinBalance < episode.coinCost) {
    return c.json({ error: "Not enough coins", coinBalance: user.coinBalance, coinCost: episode.coinCost }, 402);
  }

  await db.batch([
    db
      .update(users)
      .set({ coinBalance: user.coinBalance - episode.coinCost })
      .where(eq(users.id, userId)),
    db.insert(unlockedEpisodes).values({
      userId,
      episodeId,
      unlockedAt: new Date(),
    }),
  ]);

  return c.json({
    ok: true,
    coinBalance: user.coinBalance - episode.coinCost,
  });
});
