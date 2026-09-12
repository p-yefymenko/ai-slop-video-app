import { eq } from "drizzle-orm";
import { getPlatformProxy } from "wrangler";
import { getDb } from "./client";
import { episodes, series } from "./schema";

const SAMPLE_VIDEOS = [
  "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
  "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
  "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
];

const SEED = [
  {
    id: "series-midnight-heiress",
    slug: "midnight-heiress",
    title: "Midnight Heiress",
    description: "She came home to claim the empire they stole.",
    coverImageUrl: "https://picsum.photos/seed/midnight-heiress/720/1280",
    episodes: [
      { id: "ep-mh-1", order: 1, title: "The Return", isFree: true, coinCost: 0 },
      { id: "ep-mh-2", order: 2, title: "The Contract", isFree: true, coinCost: 0 },
      { id: "ep-mh-3", order: 3, title: "The Price", isFree: false, coinCost: 50 },
    ],
  },
  {
    id: "series-neon-vow",
    slug: "neon-vow",
    title: "Neon Vow",
    description: "A fake engagement in a city that never sleeps.",
    coverImageUrl: "https://picsum.photos/seed/neon-vow/720/1280",
    episodes: [
      { id: "ep-nv-1", order: 1, title: "Rain Check", isFree: true, coinCost: 0 },
      { id: "ep-nv-2", order: 2, title: "Witness", isFree: true, coinCost: 0 },
      { id: "ep-nv-3", order: 3, title: "The Ring", isFree: false, coinCost: 50 },
    ],
  },
  {
    id: "series-last-train",
    slug: "last-train",
    title: "Last Train Home",
    description: "One missed stop, one second chance.",
    coverImageUrl: "https://picsum.photos/seed/last-train/720/1280",
    episodes: [
      { id: "ep-lt-1", order: 1, title: "Platform 9", isFree: true, coinCost: 0 },
      { id: "ep-lt-2", order: 2, title: "Signal Lost", isFree: true, coinCost: 0 },
      { id: "ep-lt-3", order: 3, title: "End of the Line", isFree: false, coinCost: 80 },
    ],
  },
];

async function seed() {
  const proxy = await getPlatformProxy<{ DB: D1Database }>();
  const db = getDb(proxy.env as never);

  for (const item of SEED) {
    const existing = await db.select().from(series).where(eq(series.id, item.id)).get();
    if (existing) {
      await db
        .update(series)
        .set({
          title: item.title,
          description: item.description,
          coverImageUrl: item.coverImageUrl,
          isPublished: true,
          slug: item.slug,
        })
        .where(eq(series.id, item.id));
    } else {
      await db.insert(series).values({
        id: item.id,
        title: item.title,
        description: item.description,
        coverImageUrl: item.coverImageUrl,
        isPublished: true,
        slug: item.slug,
      });
    }

    for (const episode of item.episodes) {
      const videoUrl = SAMPLE_VIDEOS[(episode.order - 1) % SAMPLE_VIDEOS.length];
      const thumbnailUrl = `https://picsum.photos/seed/${episode.id}/720/1280`;
      const found = await db.select().from(episodes).where(eq(episodes.id, episode.id)).get();
      if (found) {
        await db
          .update(episodes)
          .set({
            seriesId: item.id,
            order: episode.order,
            title: episode.title,
            videoUrl,
            thumbnailUrl,
            coinCost: episode.coinCost,
            isFree: episode.isFree,
          })
          .where(eq(episodes.id, episode.id));
      } else {
        await db.insert(episodes).values({
          id: episode.id,
          seriesId: item.id,
          order: episode.order,
          title: episode.title,
          videoUrl,
          thumbnailUrl,
          coinCost: episode.coinCost,
          isFree: episode.isFree,
        });
      }
    }
  }

  await proxy.dispose();
  console.log("Seeded local D1 with dummy series and episodes.");
}

seed().catch((error) => {
  console.error(error);
  process.exit(1);
});
