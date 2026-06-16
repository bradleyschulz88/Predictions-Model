# Phase 1: Supabase + Vercel cloud sync

This guide sets up **accounts** and **cross-device bet sync** for the predictions dashboard.

Stack:
- **Auth + DB:** Supabase
- **API:** Vercel serverless (`/api/*`)
- **Frontend:** existing `dashboard/` app
- **Payments:** Stripe webhook stub (Phase 3)

---

## 1. Create a Supabase project

1. Go to [supabase.com](https://supabase.com) and create a project.
2. Open **SQL Editor** and run the migration:

   `supabase/migrations/001_phase1.sql`

3. Open **Authentication → Providers** and enable **Email** (password sign-in).
4. Optional: disable “Confirm email” for faster testing (re-enable for production).

Copy from **Project Settings → API**:
- Project URL → `SUPABASE_URL`
- `anon` public key → `SUPABASE_ANON_KEY`
- `service_role` key → `SUPABASE_SERVICE_ROLE_KEY` (server only — never put in frontend)

---

## 2. Deploy API + frontend on Vercel

1. Install the [Vercel CLI](https://vercel.com/docs/cli) or connect the GitHub repo in the Vercel dashboard.
2. Import this repository as a new Vercel project.
3. Set **Root Directory** to the repo root (where `vercel.json` lives).
4. Add environment variables (Production + Preview):

| Variable | Value |
|----------|--------|
| `SUPABASE_URL` | `https://xxxx.supabase.co` |
| `SUPABASE_ANON_KEY` | anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | service role key |
| `NEXT_PUBLIC_SUPABASE_URL` | same as `SUPABASE_URL` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | same as `SUPABASE_ANON_KEY` |
| `ALLOWED_ORIGINS` | `https://your-app.vercel.app,https://bradleyschulz88.github.io` |

5. Deploy. Vercel runs `npm run build`, which writes `dashboard/config.js` from env vars.

Your live app URL will be something like `https://predictions-dashboard.vercel.app`.

---

## 3. Configure Supabase auth redirect URLs

In Supabase **Authentication → URL configuration**, add:

- Site URL: `https://your-app.vercel.app`
- Redirect URLs: `https://your-app.vercel.app/**`

---

## 4. Test sign-in and bet sync

1. Open your Vercel URL.
2. Click **Sign in to sync bets** in the top bar.
3. Create an account.
4. Add a bet on **My Bets**.
5. Open the same URL on another device/browser, sign in with the same email — bets should appear.

Local-only mode still works on GitHub Pages when `config.js` has empty Supabase keys.

---

## API routes

| Route | Method | Description |
|-------|--------|-------------|
| `/api/me` | GET | Profile + subscription summary |
| `/api/sync` | GET | Full sync bundle (bets + settings) |
| `/api/bets` | GET, PUT | List or replace all bets |
| `/api/settings` | GET, PUT | Bankroll + odds format |
| `/api/stripe/webhook` | POST | Stripe stub (Phase 3) |

All routes require `Authorization: Bearer <supabase_access_token>` except the Stripe webhook.

---

## Local development

```bash
cp .env.example .env
# Fill in Supabase keys in .env

npm install
npm run build          # writes dashboard/config.js
npx vercel dev         # serves dashboard + /api locally
```

---

## GitHub Pages vs Vercel

| Host | Predictions | Cloud bet sync |
|------|-------------|----------------|
| GitHub Pages | Yes (public JSON) | No (unless you point `config.js` at Vercel API + Supabase) |
| Vercel | Yes (same static JSON copied in build) | Yes |

You can keep GitHub Pages for predictions and set `apiBase` in Vercel-generated `config.js` to your Vercel URL so GH Pages frontend talks to Vercel API. Add the GitHub Pages origin to `ALLOWED_ORIGINS`.

---

## Phase 3 preview (Stripe)

`subscriptions` table and `/api/stripe/webhook` stub are ready. Next steps:

1. Create a Stripe product + price.
2. Add `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_ID`.
3. Implement Checkout + webhook to set `subscriptions.status = 'active'`.
4. Gate `/api/games` (Phase 2) behind active subscription.

---

## Security notes

- Never commit `.env` or the Supabase **service role** key.
- Row Level Security is enabled on all user tables.
- API routes verify the user JWT before using the service role to read/write that user's data.
