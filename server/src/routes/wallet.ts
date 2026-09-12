import { eq } from "drizzle-orm";
import { Hono } from "hono";
import { getDb } from "../db/client";
import { users } from "../db/schema";
import type { AppVariables, Env } from "../env";

export const walletRoutes = new Hono<{ Bindings: Env; Variables: AppVariables }>();

walletRoutes.get("/users/me/wallet", async (c) => {
  const userId = c.get("userId");
  const db = getDb(c.env);
  const user = await db.select().from(users).where(eq(users.id, userId)).get();
  if (!user) {
    return c.json({ error: "User not found" }, 404);
  }
  return c.json({ userId: user.id, coinBalance: user.coinBalance });
});
