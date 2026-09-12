import { eq } from "drizzle-orm";
import { Hono } from "hono";
import { getDb } from "../db/client";
import { purchases, users } from "../db/schema";
import type { AppVariables, Env } from "../env";
import { verifyPlayPurchase } from "../playBilling";

export const purchaseRoutes = new Hono<{ Bindings: Env; Variables: AppVariables }>();

purchaseRoutes.post("/purchases/verify", async (c) => {
  const userId = c.get("userId");
  let body: { productId?: string; purchaseToken?: string };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON body" }, 400);
  }
  if (!body.productId || !body.purchaseToken) {
    return c.json({ error: "productId and purchaseToken are required" }, 400);
  }

  const verified = await verifyPlayPurchase(c.env, body.productId, body.purchaseToken);
  const db = getDb(c.env);

  if (!verified.ok) {
    await db.insert(purchases).values({
      id: crypto.randomUUID(),
      userId,
      playOrderId: body.purchaseToken,
      coinsGranted: 0,
      amountPaid: null,
      verifiedAt: new Date(),
      status: "failed",
      productId: body.productId,
    });
    return c.json({ error: verified.reason ?? "Verification failed" }, 400);
  }

  const existing = await db.select().from(purchases).where(eq(purchases.playOrderId, verified.orderId)).get();
  if (existing?.status === "verified") {
    const user = await db.select().from(users).where(eq(users.id, userId)).get();
    return c.json({ ok: true, alreadyProcessed: true, coinBalance: user?.coinBalance ?? 0 });
  }

  const user = await db.select().from(users).where(eq(users.id, userId)).get();
  if (!user) {
    return c.json({ error: "User not found" }, 404);
  }

  const purchaseId = existing?.id ?? crypto.randomUUID();
  const nextBalance = user.coinBalance + verified.coins;

  if (existing) {
    await db.batch([
      db.update(users).set({ coinBalance: nextBalance }).where(eq(users.id, userId)),
      db
        .update(purchases)
        .set({
          coinsGranted: verified.coins,
          verifiedAt: new Date(),
          status: "verified",
          productId: body.productId,
        })
        .where(eq(purchases.id, existing.id)),
    ]);
  } else {
    await db.batch([
      db.update(users).set({ coinBalance: nextBalance }).where(eq(users.id, userId)),
      db.insert(purchases).values({
        id: purchaseId,
        userId,
        playOrderId: verified.orderId,
        coinsGranted: verified.coins,
        amountPaid: verified.amountPaid,
        verifiedAt: new Date(),
        status: "verified",
        productId: body.productId,
      }),
    ]);
  }

  return c.json({ ok: true, coinsGranted: verified.coins, coinBalance: nextBalance });
});
