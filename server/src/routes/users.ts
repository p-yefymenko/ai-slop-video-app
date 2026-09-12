import { eq } from "drizzle-orm";
import { Hono } from "hono";
import { getDb } from "../db/client";
import { unlockedEpisodes, users } from "../db/schema";
import type { AppVariables, Env } from "../env";

export const userRoutes = new Hono<{ Bindings: Env; Variables: AppVariables }>();

userRoutes.get("/users/me", async (c) => {
  const userId = c.get("userId");
  const db = getDb(c.env);
  const user = await db.select().from(users).where(eq(users.id, userId)).get();
  if (!user) {
    return c.json({ error: "User not found" }, 404);
  }
  const unlocked = await db
    .select()
    .from(unlockedEpisodes)
    .where(eq(unlockedEpisodes.userId, userId))
    .all();
  return c.json({
    user: {
      id: user.id,
      googlePlayAccountId: user.googlePlayAccountId,
      deviceId: user.deviceId,
      coinBalance: user.coinBalance,
      createdAt: user.createdAt.toISOString(),
    },
    unlockedEpisodeIds: unlocked.map((row) => row.episodeId),
  });
});
