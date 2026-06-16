import { adminClient } from "./http.js";

export async function fetchUserBundle(userId) {
  const supabase = adminClient();

  const [settingsResult, betsResult, subscriptionResult, profileResult] = await Promise.all([
    supabase.from("user_settings").select("starting_bankroll, odds_format, updated_at").eq("user_id", userId).maybeSingle(),
    supabase.from("bets").select("id, payload, updated_at").eq("user_id", userId).order("updated_at", { ascending: false }),
    supabase.from("subscriptions").select("status, stripe_customer_id, stripe_subscription_id, current_period_end, updated_at").eq("user_id", userId).maybeSingle(),
    supabase.from("profiles").select("email, created_at").eq("id", userId).maybeSingle(),
  ]);

  if (settingsResult.error) throw settingsResult.error;
  if (betsResult.error) throw betsResult.error;
  if (subscriptionResult.error) throw subscriptionResult.error;
  if (profileResult.error) throw profileResult.error;

  return {
    profile: profileResult.data,
    settings: settingsResult.data || { starting_bankroll: 0, odds_format: "decimal" },
    subscription: subscriptionResult.data || { status: "inactive" },
    bets: (betsResult.data || []).map((row) => ({
      ...row.payload,
      id: row.id,
      cloudUpdatedAt: row.updated_at,
    })),
  };
}

export async function replaceUserBets(userId, bets) {
  const supabase = adminClient();
  const normalized = Array.isArray(bets) ? bets : [];

  const { error: clearError } = await supabase.from("bets").delete().eq("user_id", userId);
  if (clearError) throw clearError;

  if (!normalized.length) {
    return [];
  }

  const rows = normalized.map((bet) => ({
    id: String(bet.id),
    user_id: userId,
    payload: bet,
    updated_at: bet.cloudUpdatedAt || bet.updatedAt || new Date().toISOString(),
  }));

  const { error } = await supabase.from("bets").insert(rows);
  if (error) throw error;
  return normalized;
}

export async function upsertUserSettings(userId, settings) {
  const supabase = adminClient();
  const row = {
    user_id: userId,
    starting_bankroll: Math.max(0, Number(settings?.starting_bankroll ?? settings?.startingBankroll ?? 0)),
    odds_format: settings?.odds_format === "american" || settings?.oddsFormat === "american" ? "american" : "decimal",
    updated_at: new Date().toISOString(),
  };

  const { data, error } = await supabase.from("user_settings").upsert(row, { onConflict: "user_id" }).select().maybeSingle();
  if (error) throw error;
  return data;
}
