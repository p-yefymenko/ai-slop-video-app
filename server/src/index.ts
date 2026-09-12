import { Hono } from "hono";
import { cors } from "hono/cors";
import type { AppVariables, Env } from "./env";
import { deviceAuth } from "./middleware/auth";
import { PRIVACY_HTML } from "./privacy";
import { adminRoutes } from "./routes/admin";
import { adminEpisodeRoutes } from "./routes/adminEpisodes";
import { episodeRoutes } from "./routes/episodes";
import { purchaseRoutes } from "./routes/purchases";
import { seriesRoutes } from "./routes/series";
import { userRoutes } from "./routes/users";
import { walletRoutes } from "./routes/wallet";
import { verifyPlaybackToken } from "./storage";

const app = new Hono<{ Bindings: Env; Variables: AppVariables }>();

app.use(
  "*",
  cors({
    origin: "*",
    allowHeaders: ["Content-Type", "Authorization", "X-Device-Id"],
    allowMethods: ["GET", "POST", "PUT", "OPTIONS"],
  }),
);

app.get("/", (c) => c.json({ ok: true, name: "reelshort-api" }));
app.get("/privacy", (c) => c.html(PRIVACY_HTML));

app.route("/admin", adminRoutes);
app.route("/", adminEpisodeRoutes);

app.get("/media/:key", async (c) => {
  const key = decodeURIComponent(c.req.param("key"));
  const ok = await verifyPlaybackToken(c.env, key, c.req.query("exp") ?? null, c.req.query("token") ?? null);
  if (!ok) {
    return c.json({ error: "Invalid or expired media token" }, 403);
  }
  const object = await c.env.VIDEO_BUCKET.get(key);
  if (!object) {
    return c.json({ error: "Not found" }, 404);
  }
  const headers = new Headers();
  headers.set("Content-Type", object.httpMetadata?.contentType ?? "video/mp4");
  headers.set("Cache-Control", "private, max-age=60");
  if (object.httpEtag) {
    headers.set("ETag", object.httpEtag);
  }
  return new Response(object.body, { headers });
});

const authed = new Hono<{ Bindings: Env; Variables: AppVariables }>();
authed.use("*", deviceAuth);
authed.route("/series", seriesRoutes);
authed.route("/", episodeRoutes);
authed.route("/", purchaseRoutes);
authed.route("/", walletRoutes);
authed.route("/", userRoutes);
app.route("/", authed);

app.notFound((c) => c.json({ error: "Not found" }, 404));

export default app;
