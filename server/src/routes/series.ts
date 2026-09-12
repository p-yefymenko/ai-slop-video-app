import { eq } from "drizzle-orm";
import { Hono } from "hono";
import { getDb } from "../db/client";
import { series } from "../db/schema";
import type { Env } from "../env";

export const seriesRoutes = new Hono<{ Bindings: Env }>();

seriesRoutes.get("/", async (c) => {
  const db = getDb(c.env);
  const rows = await db.select().from(series).where(eq(series.isPublished, true)).all();
  return c.json({
    series: rows.map((row) => ({
      id: row.id,
      title: row.title,
      description: row.description,
      coverImageUrl: row.coverImageUrl,
      isPublished: row.isPublished,
    })),
  });
});
