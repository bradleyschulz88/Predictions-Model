import { applyCors, readJsonBody, requireUser, sendJson } from "../_lib/http.js";
import { fetchUserBundle, replaceUserBets } from "../_lib/users.js";

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
      sendJson(res, 200, { bets: bundle.bets });
      return;
    }

    if (req.method === "PUT") {
      const body = await readJsonBody(req);
      const bets = await replaceUserBets(user.id, body.bets || []);
      sendJson(res, 200, { bets, count: bets.length });
      return;
    }

    sendJson(res, 405, { error: "Method not allowed." });
  } catch (error) {
    sendJson(res, 500, { error: error.message || "Bet sync failed." });
  }
}
