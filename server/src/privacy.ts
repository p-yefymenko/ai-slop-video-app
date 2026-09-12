export const PRIVACY_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Privacy Policy — ReelShort Clone</title>
    <style>
      body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #111; }
      h1 { font-size: 1.75rem; }
    </style>
  </head>
  <body>
    <h1>Privacy Policy</h1>
    <p>Last updated: 12 September 2026</p>
    <p>
      This app (“ReelShort Clone”) is a short-drama video player. We collect only what is needed to run
      the service: a device identifier, coin wallet balance, episode unlock records, and Google Play
      purchase tokens used to verify coin packs you buy.
    </p>
    <h2>Data we store</h2>
    <ul>
      <li>Device ID (generated on first launch) to associate your wallet and unlocks</li>
      <li>Google Play purchase tokens and order IDs for coin pack verification</li>
      <li>Episode unlock history and coin balance</li>
    </ul>
    <h2>How we use it</h2>
    <p>
      Data is used to serve videos you are allowed to watch, credit coins after a verified Play purchase,
      and prevent replay of the same purchase token. We do not sell personal data. Video files are stored
      on Cloudflare R2; account records are stored in Cloudflare D1.
    </p>
    <h2>Retention and deletion</h2>
    <p>
      You can request deletion of your device record and associated wallet/unlock/purchase rows by
      contacting the developer listed on the Google Play store listing. We will delete that data unless
      we must keep a purchase record to satisfy Google Play’s refund and fraud rules.
    </p>
    <h2>Contact</h2>
    <p>Use the email address published on the app’s Google Play listing.</p>
  </body>
</html>
`;
