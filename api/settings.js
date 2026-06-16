import { applyCors, readJsonBody, requireUser, sendJson } from "../_lib/http.js";
import { fetchUserBundle, upsertUserSettings } from "../_lib/users.js";

export default async function handler(req, res) {
  applyCors(req, res);
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }

  try {
    const user = await requireUser(req, res);
    if (!user) return;

    if (req.method === "GET") {
      const bundle = await fetchUserBundle(user.id);
      sendJson(res, 200, {
        startingBankroll: Number(bundle.settings.starting_bankroll) || 0,
        oddsFormat: bundle.settings.odds_format || "decimal",
        updatedAt: bundle.settings.updated_at,
      });
      return;
    }

    if (req.method === "PUT") {
      const body = await readJsonBody(req);
      const settings = await upsertUserSettings(user.id, body);
      sendJson(res, 200, {
        startingBankroll: Number(settings.starting_bankroll) || 0,
        oddsFormat: settings.odds_format || "decimal",
        updatedAt: settings.updated_at,
      });
      return;
    }

    sendJson(res, 405, { error: "Method not allowed." });
  } catch (error) {
    sendJson(res, 500, { error: error.message || "Settings sync failed." });
  }
}
