/**
 * Supabase auth + Vercel API sync for My Bets.
 */
(function initPredictionsCloud(global) {
  const state = {
    enabled: false,
    supabase: null,
    session: null,
    user: null,
    syncTimer: null,
    pendingSync: false,
    listeners: new Set(),
  };

  function config() {
    return global.APP_CONFIG || {};
  }

  function isEnabled() {
    const cfg = config();
    return Boolean(cfg.supabaseUrl && cfg.supabaseAnonKey);
  }

  function apiBase() {
    const cfg = config();
    return cfg.apiBase || "";
  }

  function notify() {
    for (const listener of state.listeners) {
      try {
        listener(state);
      } catch {
        /* ignore */
      }
    }
  }

  async function apiFetch(path, options = {}) {
    const token = state.session?.access_token;
    if (!token) {
      throw new Error("Not signed in.");
    }

    const headers = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(options.headers || {}),
    };

    const response = await fetch(`${apiBase()}${path}`, {
      ...options,
      headers,
      body: options.body != null ? JSON.stringify(options.body) : undefined,
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json") ? await response.json() : null;

    if (!response.ok) {
      throw new Error(payload?.error || `Request failed (${response.status})`);
    }

    return payload;
  }

  function mergeBets(localBets, remoteBets) {
    const merged = new Map();

    for (const bet of remoteBets || []) {
      if (!bet?.id) continue;
      merged.set(bet.id, { ...bet, cloudUpdatedAt: bet.cloudUpdatedAt || bet.updatedAt || null });
    }

    for (const bet of localBets || []) {
      if (!bet?.id) continue;
      const existing = merged.get(bet.id);
      const localUpdated = Date.parse(bet.updatedAt || bet.createdAt || 0) || 0;
      const remoteUpdated = Date.parse(existing?.cloudUpdatedAt || existing?.updatedAt || existing?.createdAt || 0) || 0;
      if (!existing || localUpdated >= remoteUpdated) {
        merged.set(bet.id, { ...bet, updatedAt: bet.updatedAt || new Date().toISOString() });
      }
    }

    return [...merged.values()];
  }

  const PredictionsCloud = {
    get enabled() {
      return state.enabled;
    },

    get session() {
      return state.session;
    },

    get user() {
      return state.user;
    },

    onChange(listener) {
      state.listeners.add(listener);
      return () => state.listeners.delete(listener);
    },

    async init() {
      state.enabled = isEnabled();
      if (!state.enabled || !global.supabase?.createClient) {
        return false;
      }

      const cfg = config();
      state.supabase = global.supabase.createClient(cfg.supabaseUrl, cfg.supabaseAnonKey, {
        auth: {
          persistSession: true,
          autoRefreshToken: true,
          detectSessionInUrl: true,
        },
      });

      const { data } = await state.supabase.auth.getSession();
      state.session = data.session;
      state.user = data.session?.user || null;
      notify();

      state.supabase.auth.onAuthStateChange((_event, session) => {
        state.session = session;
        state.user = session?.user || null;
        notify();
      });

      return Boolean(state.session);
    },

    async signUp(email, password) {
      if (!state.supabase) throw new Error("Cloud sync is not configured.");
      const { data, error } = await state.supabase.auth.signUp({ email, password });
      if (error) throw error;
      state.session = data.session;
      state.user = data.user;
      notify();
      return data;
    },

    async signIn(email, password) {
      if (!state.supabase) throw new Error("Cloud sync is not configured.");
      const { data, error } = await state.supabase.auth.signInWithPassword({ email, password });
      if (error) throw error;
      state.session = data.session;
      state.user = data.user;
      notify();
      return data;
    },

    async signOut() {
      if (!state.supabase) return;
      await state.supabase.auth.signOut();
      state.session = null;
      state.user = null;
      notify();
    },

    async pullSync() {
      if (!state.session) return null;
      return apiFetch("/api/sync");
    },

    async pushBets(bets) {
      if (!state.session) return null;
      const stamped = (bets || []).map((bet) => ({
        ...bet,
        updatedAt: new Date().toISOString(),
      }));
      return apiFetch("/api/bets", { method: "PUT", body: { bets: stamped } });
    },

    async pushSettings(settings) {
      if (!state.session) return null;
      return apiFetch("/api/settings", {
        method: "PUT",
        body: {
          startingBankroll: settings.startingBankroll,
          oddsFormat: settings.oddsFormat,
        },
      });
    },

    queueSync(fn) {
      if (!state.session) return;
      state.pendingSync = true;
      if (state.syncTimer) clearTimeout(state.syncTimer);
      state.syncTimer = setTimeout(async () => {
        state.syncTimer = null;
        if (!state.pendingSync) return;
        state.pendingSync = false;
        try {
          await fn();
        } catch (error) {
          console.warn("Cloud sync failed:", error);
        }
      }, 600);
    },
  };

  PredictionsCloud.mergeBets = mergeBets;
  global.PredictionsCloud = PredictionsCloud;
})(window);
