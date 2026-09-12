export interface Env {
  DB: D1Database;
  VIDEO_BUCKET: R2Bucket;
  ENVIRONMENT: string;
  ANDROID_PACKAGE_NAME: string;
  ADMIN_SECRET?: string;
  MEDIA_SIGNING_SECRET?: string;
  GOOGLE_PLAY_SERVICE_ACCOUNT?: string;
}

export type AppVariables = {
  userId: string;
};
