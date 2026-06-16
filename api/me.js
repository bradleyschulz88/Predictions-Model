import { applyCors, readJsonBody, requireUser, sendJson } from "../_lib/http.js";
import { fetchUserBundle } from "../_lib/users.js";

export default async function handler(req, res) {
  applyCors(req, res);
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }

  if (req.method !== "GET") {
    sendJson(res, 405, { error: "Method not allowed." });
    return;
  }

  try {
    const user = await requireUser(req, res);
    if (!user) return;

    const bundle = await fetchUserBundle(user.id);
    sendJson(res, 200, {
      user: {
        id: user.id,
        email: user.email || bundle.profile?.email || null,
        createdAt: bundle.profile?.created_at || user.created_at,
      },
      settings: {
        startingBankroll: Number(bundle.settings.starting_bankroll) || 0,
        oddsFormat: bundle.settings.odds_format || "decimal",
        updatedAt: bundle.settings.updated_at,
      },
      subscription: {
        status: bundle.subscription.status || "inactive",
        stripeCustomerId: bundle.subscription.stripe_customer_id || null,
        stripeSubscriptionId: bundle.subscription.stripe_subscription_id || null,
        currentPeriodEnd: bundle.subscription.current_period_end || null,
      },
      betCount: bundle.bets.length,
    });
  } catch (error) {
    sendJson(res, 500, { error: error.message || "Failed to load profile." });
  }
}
