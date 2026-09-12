import { eq } from "drizzle-orm";
import { Hono } from "hono";
import { getDb } from "../db/client";
import { series as seriesTable } from "../db/schema";
import type { Env } from "../env";
import { requireAdmin } from "../middleware/auth";
import { putObject } from "../storage";

function isAuthorized(env: Env, header: string | undefined) {
  const secret = env.ADMIN_SECRET ?? (env.ENVIRONMENT === "production" ? undefined : "dev-admin-secret");
  return requireAdmin(secret, header);
}

export const adminRoutes = new Hono<{ Bindings: Env }>();

adminRoutes.use("/*", async (c, next) => {
  if (!isAuthorized(c.env, c.req.header("Authorization"))) {
    return c.json({ error: "Unauthorized" }, 401);
  }
  await next();
});

adminRoutes.post("/series", async (c) => {
  const body = await c.req.json<{
    id?: string;
    title: string;
    description?: string;
    coverImageUrl?: string;
    isPublished?: boolean;
    slug?: string;
  }>();
  if (!body.title) {
    return c.json({ error: "title is required" }, 400);
  }
  const db = getDb(c.env);
  const id = body.id ?? crypto.randomUUID();
  const existing = await db.select().from(seriesTable).where(eq(seriesTable.id, id)).get();
  if (existing) {
    await db
      .update(seriesTable)
      .set({
        title: body.title,
        description: body.description ?? existing.description,
        coverImageUrl: body.coverImageUrl ?? existing.coverImageUrl,
        isPublished: body.isPublished ?? existing.isPublished,
        slug: body.slug ?? existing.slug,
      })
      .where(eq(seriesTable.id, id));
    return c.json({ id, updated: true });
  }
  await db.insert(seriesTable).values({
    id,
    title: body.title,
    description: body.description ?? "",
    coverImageUrl: body.coverImageUrl ?? "",
    isPublished: body.isPublished ?? false,
    slug: body.slug ?? null,
  });
  return c.json({ id, created: true });
});

adminRoutes.post("/upload", async (c) => {
  const form = await c.req.formData();
  const file = form.get("file");
  const key = form.get("key");
  if (!(file instanceof File) || typeof key !== "string") {
    return c.json({ error: "file and key are required" }, 400);
  }
  const buffer = await file.arrayBuffer();
  await putObject(c.env, key, buffer, file.type || "application/octet-stream");
  return c.json({ key });
});
