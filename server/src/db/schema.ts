import { integer, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";

export const users = sqliteTable("users", {
  id: text("id").primaryKey(),
  googlePlayAccountId: text("google_play_account_id"),
  deviceId: text("device_id"),
  coinBalance: integer("coin_balance").notNull().default(0),
  createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
});

export const series = sqliteTable("series", {
  id: text("id").primaryKey(),
  title: text("title").notNull(),
  description: text("description").notNull().default(""),
  coverImageUrl: text("cover_image_url").notNull().default(""),
  isPublished: integer("is_published", { mode: "boolean" }).notNull().default(false),
  slug: text("slug"),
});

export const episodes = sqliteTable("episodes", {
  id: text("id").primaryKey(),
  seriesId: text("series_id")
    .notNull()
    .references(() => series.id),
  order: integer("order").notNull(),
  title: text("title").notNull(),
  videoUrl: text("video_url"),
  thumbnailUrl: text("thumbnail_url"),
  coinCost: integer("coin_cost").notNull().default(0),
  isFree: integer("is_free", { mode: "boolean" }).notNull().default(false),
});

export const unlockedEpisodes = sqliteTable(
  "unlocked_episodes",
  {
    userId: text("user_id")
      .notNull()
      .references(() => users.id),
    episodeId: text("episode_id")
      .notNull()
      .references(() => episodes.id),
    unlockedAt: integer("unlocked_at", { mode: "timestamp" }).notNull(),
  },
  (table) => ({
    pk: uniqueIndex("unlocked_episodes_pk").on(table.userId, table.episodeId),
  }),
);

export const purchases = sqliteTable(
  "purchases",
  {
    id: text("id").primaryKey(),
    userId: text("user_id")
      .notNull()
      .references(() => users.id),
    playOrderId: text("play_order_id").notNull(),
    coinsGranted: integer("coins_granted").notNull(),
    amountPaid: integer("amount_paid"),
    verifiedAt: integer("verified_at", { mode: "timestamp" }),
    status: text("status", { enum: ["pending", "verified", "failed"] })
      .notNull()
      .default("pending"),
    productId: text("product_id"),
  },
  (table) => ({
    playOrder: uniqueIndex("purchases_play_order_id").on(table.playOrderId),
  }),
);

export const subscriptions = sqliteTable("subscriptions", {
  userId: text("user_id")
    .notNull()
    .references(() => users.id),
  tier: text("tier").notNull(),
  renewsAt: integer("renews_at", { mode: "timestamp" }),
  status: text("status").notNull().default("inactive"),
});
