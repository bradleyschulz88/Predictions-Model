/**
 * Stripe webhook stub — wired in Phase 3.
 * Configure STRIPE_WEBHOOK_SECRET and subscription update logic later.
 */

import { applyCors, sendJson } from "../_lib/http.js";

export default async function handler(req, res) {
  applyCors(req, res);
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }

  if (req.method !== "POST") {
    sendJson(res, 405, { error: "Method not allowed." });
    return;
  }

  if (!process.env.STRIPE_WEBHOOK_SECRET) {
    sendJson(res, 501, { error: "Stripe webhook is not configured yet." });
    return;
  }

  sendJson(res, 501, {
    error: "Stripe webhook handler will be enabled in Phase 3.",
  });
}
