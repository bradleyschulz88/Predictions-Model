const sportSelect = document.getElementById("sport-select");
const dateDisplayBtn = document.getElementById("date-display");
const datePickerInput = document.getElementById("date-picker-input");
const datePrevBtn = document.getElementById("date-prev");
const dateNextBtn = document.getElementById("date-next");
const dateQuickEl = document.getElementById("date-quick");
const dateHintEl = document.getElementById("date-hint");
const confidenceFilter = document.getElementById("confidence-filter");
const confidenceFilterMobile = document.getElementById("confidence-filter-mobile");
const oddsFormatSelect = document.getElementById("odds-format");
const teamSearch = document.getElementById("team-search");
const teamSearchMobile = document.getElementById("team-search-mobile");
const refreshBtn = document.getElementById("refresh-btn");
const autoRefresh = document.getElementById("auto-refresh");
const autoRefreshMobile = document.getElementById("auto-refresh-mobile");
const gamesEl = document.getElementById("games");
const bannerEl = document.getElementById("banner");
const bannerSummaryEl = document.getElementById("banner-summary");
const bannerDetailsEl = document.getElementById("banner-details");
const bannerDetailTextEl = document.getElementById("banner-detail-text");
const filterPanelDesktopEl = document.getElementById("filter-panel-desktop");
const filterOpenBtn = document.getElementById("filter-open-btn");
const filterCloseBtn = document.getElementById("filter-close-btn");
const filterSheetEl = document.getElementById("filter-sheet");
const filterSheetBackdropEl = document.getElementById("filter-sheet-backdrop");
const filterApplyBtn = document.getElementById("filter-apply-btn");
const topPicksEl = document.getElementById("top-picks");
const dashboardTitle = document.getElementById("dashboard-title");
const statsEl = document.getElementById("stats");

const statGames = document.getElementById("stat-games");
const statPicks = document.getElementById("stat-picks");
const statStrong = document.getElementById("stat-strong");
const statLean = document.getElementById("stat-lean");
const statLive = document.getElementById("stat-live");
const statTopPick = document.getElementById("stat-top-pick");
const statLeague = document.getElementById("stat-league");
const statDate = document.getElementById("stat-date");
const statUpdated = document.getElementById("stat-updated");
const statFreshness = document.getElementById("stat-freshness");
const freshnessNote = document.getElementById("freshness-note");
const liveScoresToggle = document.getElementById("live-scores");
const liveScoresToggleMobile = document.getElementById("live-scores-mobile");
const viewTabsEl = document.getElementById("view-tabs");
const predictionsViewEl = document.getElementById("predictions-view");
const accuracyViewEl = document.getElementById("accuracy-view");
const myBetsViewEl = document.getElementById("my-bets-view");
const modelTrackerViewEl = document.getElementById("model-tracker-view");
const dateFieldEl = document.getElementById("date-field");
const modelDayResultEl = document.getElementById("model-day-result");

const COVERAGE_LABELS = {
  lineup: "Lineup",
  injuries: "Injuries",
  espnPredictor: "ESPN",
  advancedStats: "Adv",
  restData: "Rest",
  scheduleFlags: "Sched",
  mlbPitching: "Pitch",
  leagueMetrics: "League",
  impliedOdds: "Odds",
};

const STALE_LIVE_MINUTES = 20;
const STALE_SCORELESS_MINUTES = 10;

function parseEspnNoteText(notes) {
  if (!Array.isArray(notes)) return "";
  return notes
    .map((note) => [note?.headline, note?.text, note?.description].filter(Boolean).join(" "))
    .join(" ");
}

function isWashedOutInProgress({ name, state, attendance, totalRuns, startDate }) {
  if (name !== "STATUS_IN_PROGRESS" || state !== "in") return false;
  if (attendance != null && attendance !== 0) return false;
  if (totalRuns != null && totalRuns !== 0) return false;
  const started = startDate ? new Date(startDate) : null;
  if (!started || Number.isNaN(started.getTime())) return false;
  const elapsedMinutes = (Date.now() - started.getTime()) / 60000;
  return elapsedMinutes >= STALE_SCORELESS_MINUTES;
}

function normalizeEspnStatus(status, { startDate = null, attendance = null, notes = null, homeScore = null, awayScore = null } = {}) {
  const name = status?.name || "";
  const state = status?.state || "";
  const completed = Boolean(status?.completed);
  const description = status?.description || status?.shortDetail || "Scheduled";
  const detail = status?.detail || status?.shortDetail || description;
  const blob = [name, description, detail, parseEspnNoteText(notes)].join(" ").toLowerCase();

  const isPostponed = name === "STATUS_POSTPONED" || blob.includes("postpon");
  const isCanceled = name === "STATUS_CANCELED" || name === "STATUS_CANCELLED" || blob.includes("canceled") || blob.includes("cancelled");
  const isSuspended = name === "STATUS_SUSPENDED" || blob.includes("suspend");
  let isDelayed = name === "STATUS_DELAYED" || blob.includes("delay");
  let isVoided = isPostponed || isCanceled;

  const isFinal = completed || name === "STATUS_FINAL";
  const isScheduled = name === "STATUS_SCHEDULED" || state === "pre";

  const home = homeScore != null ? Number(homeScore) : null;
  const away = awayScore != null ? Number(awayScore) : null;
  const totalRuns = home != null && away != null && !Number.isNaN(home) && !Number.isNaN(away) ? home + away : null;
  const started = startDate ? new Date(startDate) : null;
  const elapsedMinutes = started && !Number.isNaN(started.getTime()) ? (Date.now() - started.getTime()) / 60000 : null;

  let isLive = name === "STATUS_IN_PROGRESS" && state === "in" && !isVoided && !isSuspended;
  let isWashedOut = false;

  if (isLive) {
    if (isWashedOutInProgress({ name, state, attendance, totalRuns, startDate })) {
      isLive = false;
      isWashedOut = true;
      isDelayed = true;
      isVoided = true;
    } else if (
      attendance === 0 &&
      elapsedMinutes != null &&
      elapsedMinutes >= STALE_LIVE_MINUTES &&
      totalRuns != null &&
      totalRuns <= 1
    ) {
      isLive = false;
      isWashedOut = true;
      isDelayed = true;
      isVoided = true;
    } else if (attendance === 0 && elapsedMinutes != null && elapsedMinutes >= STALE_LIVE_MINUTES * 2 && totalRuns != null && totalRuns > 1) {
      isLive = false;
      isDelayed = true;
    }
  }

  if (!isLive && name === "STATUS_IN_PROGRESS" && state === "in") {
    isWashedOut =
      isWashedOut ||
      isWashedOutInProgress({ name, state, attendance, totalRuns, startDate }) ||
      (attendance === 0 &&
        elapsedMinutes != null &&
        elapsedMinutes >= STALE_LIVE_MINUTES &&
        totalRuns != null &&
        totalRuns <= 1);
    if (isWashedOut) {
      isDelayed = true;
      isVoided = true;
    }
  }

  let gameStatusText = description;
  if (isPostponed) gameStatusText = "Postponed";
  else if (isCanceled) gameStatusText = "Canceled";
  else if (isSuspended) gameStatusText = "Suspended";
  else if (isWashedOut) gameStatusText = "Washed out";
  else if (isDelayed && !isLive) gameStatusText = "Delayed";

  return {
    statusType: name,
    gameStatusText,
    gameStatusDetail: detail,
    isLive,
    isFinal,
    isScheduled,
    isPostponed,
    isCanceled,
    isSuspended,
    isDelayed,
    isVoided: isVoided || isWashedOut,
    isWashedOut,
    attendance,
  };
}

function refreshGameStatusFlags(game) {
  if (!game) return game;
  const statusName =
    game.statusType ||
    (game.isPostponed ? "STATUS_POSTPONED" : game.isFinal ? "STATUS_FINAL" : game.isLive ? "STATUS_IN_PROGRESS" : "STATUS_SCHEDULED");
  const statusState =
    statusName === "STATUS_IN_PROGRESS"
      ? "in"
      : statusName === "STATUS_SCHEDULED"
        ? "pre"
        : "post";
  const inferredAttendance =
    game.attendance != null
      ? game.attendance
      : statusName === "STATUS_IN_PROGRESS"
        ? (() => {
            const runs = Number(game.homeScore ?? 0) + Number(game.awayScore ?? 0);
            const started = game.startDate ? new Date(game.startDate) : null;
            const elapsed =
              started && !Number.isNaN(started.getTime()) ? (Date.now() - started.getTime()) / 60000 : null;
            if (runs === 0) return 0;
            if (runs <= 1 && elapsed != null && elapsed >= STALE_LIVE_MINUTES) return 0;
            return null;
          })()
        : null;

  const flags = normalizeEspnStatus(
    {
      name: statusName,
      state: statusState,
      completed: game.isFinal && statusName === "STATUS_FINAL",
      description: game.gameStatusText,
      detail: game.gameStatusDetail || game.gameStatusText,
    },
    {
      startDate: game.startDate,
      attendance: inferredAttendance,
      homeScore: game.homeScore,
      awayScore: game.awayScore,
    }
  );
  return { ...game, ...flags };
}

function isGameVoided(game) {
  return Boolean(game?.isVoided || game?.isWashedOut || game?.isPostponed || game?.isCanceled);
}

function isUnplayableGame(game) {
  return Boolean(isGameVoided(game) || game?.isSuspended);
}

function gameStatusLabel(game) {
  if (game?.isLive) return "LIVE";
  if (game?.isWashedOut) return "Washed out";
  if (game?.isFinal) return "Final";
  if (game?.isPostponed) return "Postponed";
  if (game?.isCanceled) return "Canceled";
  if (game?.isSuspended) return "Suspended";
  if (game?.isDelayed) return "Delayed";
  return game?.gameStatusText || "Scheduled";
}

const ESPN_PATHS = {
  mlb: "baseball/mlb",
  nfl: "football/nfl",
  nba: "basketball/nba",
  wnba: "basketball/wnba",
  worldcup: "soccer/fifa.world",
  epl: "soccer/eng.1",
  afl: "australian-football/afl",
};

const US_SCHEDULE_SPORTS = new Set(["mlb", "nfl", "nba", "wnba"]);

const GITHUB_PAGES_REPO = "bradleyschulz88/Predictions-Model";
const UPDATE_WORKFLOW_URL = `https://github.com/${GITHUB_PAGES_REPO}/actions/workflows/pages.yml`;
const IS_STATIC_HOST =
  window.location.hostname.endsWith("github.io") ||
  new URLSearchParams(window.location.search).has("static");

const LEAGUE_TIMEZONES = {
  mlb: "America/New_York",
  nfl: "America/New_York",
  nba: "America/New_York",
  wnba: "America/New_York",
  worldcup: "America/New_York",
  epl: "Europe/London",
  afl: "Australia/Melbourne",
};

const SPORT_LABELS = {
  mlb: "MLB Baseball",
  nfl: "NFL Football",
  nba: "NBA Basketball",
  wnba: "WNBA Basketball",
  worldcup: "FIFA World Cup",
  epl: "Premier League",
  afl: "AFL",
};

let refreshTimer = null;
let liveScoresTimer = null;
let loadingDashboard = false;
let lastPayload = null;
let accuracyData = null;
let calibrationData = null;
let manifestData = null;
let overviewData = null;
let lastLiveScoreAt = null;
let activeScheduleDate = null;
let datePickerGameCount = null;

const TIMEZONE_LABELS = {
  "America/New_York": "Eastern Time",
  "Europe/London": "UK time",
  "Australia/Melbourne": "Melbourne time",
};

let activeView = "predictions";
let betFormDraft = {
  betDate: "",
  game: "",
  stake: "",
  odds: "",
  eventId: null,
  league: null,
  scheduleDate: null,
};

const MY_BETS_KEY = "predictions-dashboard-my-bets";
const BANKROLL_KEY = "predictions-dashboard-bankroll";
const MODEL_TRACKER_KEY = "predictions-dashboard-model-tracker";
const MODEL_BANKROLL_KEY = "predictions-dashboard-model-bankroll";
const MODEL_PICKS_KEY = "predictions-dashboard-model-picks";
const ODDS_FORMAT_KEY = "predictions-dashboard-odds-format";
let importPreviewRows = null;
let modelTrackerFocusId = null;
const storageWarningsShown = new Set();
let sheetPasteDraft = "";
let importFileName = "";
let importDetectedBankroll = null;
let bannerTimer = null;

function createBetId() {
  return `bet-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function storageErrorMessage(error) {
  const name = error?.name || "";
  if (name === "QuotaExceededError") {
    return "Storage is full. Free browser space or remove old site data, then try again.";
  }
  return "Could not save to this device. Check that private browsing is off and storage is allowed.";
}

function loadNormalizedList(key, normalizer) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return { ok: true, items: [] };
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return { ok: false, items: [], corrupt: true };
    const items = [];
    for (const entry of parsed) {
      try {
        const normalized = normalizer(entry);
        if (normalized) items.push(normalized);
      } catch {
        /* skip bad row, keep the rest */
      }
    }
    return { ok: true, items };
  } catch {
    return { ok: false, items: [], corrupt: true };
  }
}

function writeNormalizedList(key, items, normalizer) {
  try {
    localStorage.setItem(key, JSON.stringify(items.map(normalizer)));
    return true;
  } catch (error) {
    showBanner(storageErrorMessage(error), { type: "error" });
    return false;
  }
}

function betsChangedAfterSettle(before, after) {
  if (before.length !== after.length) return true;
  for (let index = 0; index < before.length; index += 1) {
    const left = before[index];
    const right = after[index];
    if (left?.id !== right?.id) return true;
    if (left?.status !== right?.status) return true;
    if (left?.profit !== right?.profit) return true;
  }
  return false;
}

function getOddsFormat() {
  return localStorage.getItem(ODDS_FORMAT_KEY) === "american" ? "american" : "decimal";
}

function setOddsFormat(format) {
  localStorage.setItem(ODDS_FORMAT_KEY, format === "american" ? "american" : "decimal");
}

function syncOddsFormatSelect() {
  if (!oddsFormatSelect) return;
  oddsFormatSelect.value = getOddsFormat();
}

function cleanNumericCell(value) {
  let text = String(value ?? "").trim();
  if (!text || text.startsWith("=")) return "";
  text = text.replace(/^["']+|["']+$/g, "");
  text = text.replace(/[\uFF04$£€¥₹\s\u00A0]/g, "");
  if (/^\d+,\d{1,3}$/.test(text)) {
    text = text.replace(",", ".");
  } else {
    text = text.replace(/,/g, "");
  }
  return text;
}

function parseDecimalOddsInput(value) {
  const text = cleanNumericCell(value);
  if (!text) return null;
  const num = Number(text);
  if (!Number.isFinite(num) || num < 1.01) return null;
  return Math.round(num * 1000) / 1000;
}

function parseStakeInput(value) {
  const text = cleanNumericCell(value);
  if (!text) return null;
  const num = Number(text);
  if (!Number.isFinite(num) || num <= 0) return null;
  return Math.round(num * 100) / 100;
}

function formatMoney(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const amount = Number(value);
  const prefix = amount >= 0 ? "+$" : "-$";
  return `${prefix}${Math.abs(amount).toFixed(2)}`;
}

function formatPlainMoney(value) {
  const amount = Number(value);
  if (!Number.isFinite(amount)) return "";
  return amount.toFixed(2);
}

function formatBetDateShort(iso) {
  if (!iso) return "—";
  const isoDate = String(iso).trim().match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (isoDate) return `${isoDate[3]}/${isoDate[2]}/${isoDate[1]}`;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  const day = String(date.getUTCDate()).padStart(2, "0");
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const year = date.getUTCFullYear();
  return `${day}/${month}/${year}`;
}

function todayBetDateInput() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
}

function betDateInputFromIso(iso) {
  if (!iso) return todayBetDateInput();
  const isoDate = String(iso).trim().match(/^(\d{4}-\d{2}-\d{2})/);
  if (isoDate) return isoDate[1];
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return todayBetDateInput();
  return `${parsed.getUTCFullYear()}-${String(parsed.getUTCMonth() + 1).padStart(2, "0")}-${String(parsed.getUTCDate()).padStart(2, "0")}`;
}

function parseBetDateInput(value) {
  const text = String(value ?? "").trim();
  if (!text) return new Date().toISOString();
  const isoMatch = text.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (isoMatch) {
    return new Date(`${isoMatch[1]}-${isoMatch[2]}-${isoMatch[3]}T12:00:00`).toISOString();
  }
  return parseSheetDate(text);
}

function calcBetProfitDecimal(decimalOdds, stake, won) {
  const amount = Number(stake);
  if (!won) return -amount;
  const odds = Number(decimalOdds);
  return Math.round(amount * (odds - 1) * 100) / 100;
}

function formatDecimalOdds(odds) {
  const value = Number(odds);
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(2);
}

function parseAmericanOddsInput(value) {
  const text = cleanNumericCell(value).replace(/^\+/, "");
  if (!text) return null;
  const num = Number(text);
  if (!Number.isFinite(num) || num === 0 || Math.abs(num) < 100) return null;
  return Math.round(num);
}

function americanToDecimal(american) {
  const value = Number(american);
  if (!Number.isFinite(value) || value === 0) return null;
  if (value > 0) return Math.round((value / 100 + 1) * 1000) / 1000;
  return Math.round((100 / Math.abs(value) + 1) * 1000) / 1000;
}

function resolveStoredOdds(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return null;
  if (num >= 1.01 && num < 100) return Math.round(num * 1000) / 1000;
  return americanToDecimal(num);
}

function resolveOddsToDecimal(value) {
  if (value == null || value === "") return null;
  const decimal = parseDecimalOddsInput(value);
  if (decimal != null) return decimal;
  const raw = String(value).trim();
  if (/^[+-]/.test(raw.replace(/[\s$£€¥₹]/g, ""))) {
    const american = parseAmericanOddsInput(value);
    if (american != null) return americanToDecimal(american);
  }
  return resolveStoredOdds(Number(cleanNumericCell(value)));
}

function decimalToAmerican(decimal) {
  const value = Number(decimal);
  if (!Number.isFinite(value) || value <= 1) return null;
  if (value >= 2) return Math.round((value - 1) * 100);
  return Math.round(-100 / (value - 1));
}

function parseOddsInput(value, format = getOddsFormat()) {
  if (format === "american") {
    const american = parseAmericanOddsInput(value);
    return american != null ? americanToDecimal(american) : null;
  }
  return parseDecimalOddsInput(value);
}

function formatOddsDisplay(decimalOdds, format = getOddsFormat()) {
  const value = Number(decimalOdds);
  if (!Number.isFinite(value)) return "—";
  if (format === "american") {
    const american = decimalToAmerican(value);
    if (american == null) return "—";
    return american > 0 ? `+${american}` : String(american);
  }
  return formatDecimalOdds(value);
}

function formatOddsInputPlaceholder(format = getOddsFormat()) {
  return format === "american" ? "-150 or +130" : "1.91";
}

function oddsFieldLabel({ combined = false, leg = false, format = getOddsFormat() } = {}) {
  const formatLabel = format === "american" ? "American" : "Decimal";
  if (leg) return `Leg odds (${formatLabel})`;
  if (combined) return `Combined ${formatLabel.toLowerCase()} odds`;
  return `${formatLabel} odds`;
}

function convertOddsInputBetweenFormats(value, fromFormat, toFormat) {
  if (!value || fromFormat === toFormat) return value;
  const decimal = parseOddsInput(value, fromFormat);
  if (decimal == null) return value;
  return toFormat === "american" ? formatOddsDisplay(decimal, "american") : formatDecimalOdds(decimal);
}

function convertBetFormDraftOddsFormat(newFormat, oldFormat = getOddsFormat()) {
  if (!betFormDraft || oldFormat === newFormat) return;
  const currentOdds = betFormDraft.odds ?? betFormDraft.decimalOdds ?? "";
  betFormDraft = {
    ...betFormDraft,
    odds: convertOddsInputBetweenFormats(currentOdds, oldFormat, newFormat),
  };
}

function stripBettingLinesFromPayload(payload) {
  if (!payload) return payload;
  return {
    ...payload,
    sportsbooks: undefined,
    sportsbookCount: undefined,
    games: (payload.games || []).map((game) => {
      const { lines, oddsSource, ...rest } = game;
      return rest;
    }),
  };
}

function productDecimalOdds(legs) {
  const values = (legs || [])
    .map((leg) => {
      if (typeof leg.legDecimalOdds === "number") return leg.legDecimalOdds;
      return resolveOddsToDecimal(leg.legDecimalOdds ?? leg.legOdds);
    })
    .filter((value) => value != null);
  if (!values.length) return null;
  const product = values.reduce((total, value) => total * value, 1);
  return Math.round(product * 1000) / 1000;
}

function escapeAttr(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;");
}

function escapeHtml(value) {
  return escapeAttr(value);
}

function defaultBetFormDraft(overrides = {}) {
  return {
    betDate: todayBetDateInput(),
    game: "",
    stake: "",
    odds: "",
    eventId: null,
    league: null,
    scheduleDate: null,
    linkedGame: "",
    ...overrides,
  };
}

function collectBetFormDraftFromForm(form, draft = {}) {
  return {
    ...draft,
    betDate: form?.querySelector("#bet-date")?.value ?? draft.betDate ?? todayBetDateInput(),
    game: form?.querySelector("#bet-game")?.value?.trim() ?? draft.game ?? "",
    stake: form?.querySelector("#bet-stake")?.value ?? draft.stake ?? "",
    odds: form?.querySelector("#bet-odds")?.value ?? draft.odds ?? "",
  };
}

function syncBetFormDraftFromDom() {
  const form = myBetsViewEl?.querySelector("#my-bet-form");
  if (!form) return;
  betFormDraft = collectBetFormDraftFromForm(form, betFormDraft);
}

function resolveBetFormGameLink(draft, gameText) {
  const game = String(gameText ?? "").trim();
  const linked = String(draft?.linkedGame ?? "").trim();
  if (draft?.eventId && linked && game === linked) {
    return {
      eventId: draft.eventId,
      league: draft.league || null,
      scheduleDate: draft.scheduleDate || null,
    };
  }
  return { eventId: null, league: null, scheduleDate: null };
}

function normalizeBet(bet) {
  if (!bet) return bet;
  if (Array.isArray(bet.legs) && bet.legs.length) {
    const legs = bet.legs.map((leg) => ({
      id: leg.id || createBetId(),
      matchup: leg.matchup || "Unknown matchup",
      pick: leg.pick || "Pick",
      legDecimalOdds: leg.legDecimalOdds ?? resolveOddsToDecimal(leg.legOdds) ?? null,
      eventId: leg.eventId || null,
      league: leg.league || null,
      scheduleDate: leg.scheduleDate || null,
      status: leg.status || bet.status || "pending",
      resultWinner: leg.resultWinner || null,
    }));
    const legStatuses = legs.map((leg) => leg.status);
    const derivedStatus =
      bet.status && bet.status !== "pending"
        ? bet.status
        : legStatuses.some((status) => status === "lost")
          ? "lost"
          : legStatuses.length && legStatuses.every((status) => status === "won")
            ? "won"
            : bet.status || "pending";
    const betType = bet.type || (legs.length > 1 ? "parlay" : "single");
    return ensureBetProfit({
      ...bet,
      id: bet.id || createBetId(),
      createdAt: bet.createdAt || new Date().toISOString(),
      type: betType,
      status: derivedStatus,
      decimalOdds: bet.decimalOdds ?? resolveOddsToDecimal(bet.odds) ?? productDecimalOdds(legs),
      legs,
    });
  }

  const decimalOdds = bet.decimalOdds ?? resolveOddsToDecimal(bet.odds);
  const legStatus =
    bet.status === "won" ? "won" : bet.status === "lost" ? "lost" : "pending";

  return ensureBetProfit({
    ...bet,
    id: bet.id || createBetId(),
    createdAt: bet.createdAt || new Date().toISOString(),
    type: "single",
    status: legStatus,
    decimalOdds,
    legs: [
      {
        id: createBetId(),
        matchup: bet.matchup || "Unknown matchup",
        pick: bet.pick || "Pick",
        legDecimalOdds: decimalOdds,
        eventId: bet.eventId || null,
        league: bet.league || null,
        scheduleDate: bet.scheduleDate || null,
        status: legStatus,
        resultWinner: bet.resultWinner || null,
      },
    ],
  });
}

function ensureBetProfit(bet) {
  if (!bet) return bet;
  if (bet.status === "pending") return { ...bet, profit: null };
  const decimalOdds = Number(bet.decimalOdds);
  const stake = Number(bet.stake);
  if (!Number.isFinite(decimalOdds) || !Number.isFinite(stake)) return bet;
  return {
    ...bet,
    profit: calcBetProfitDecimal(decimalOdds, stake, bet.status === "won"),
  };
}

function warnStorageOnce(key, message) {
  if (storageWarningsShown.has(key)) return;
  storageWarningsShown.add(key);
  showBanner(message, { type: "error", autoHideMs: 6000 });
}

function loadMyBets() {
  const result = loadNormalizedList(MY_BETS_KEY, normalizeBet);
  if (!result.ok && result.corrupt) {
    warnStorageOnce(
      MY_BETS_KEY,
      "Saved bets could not be read. Your data was not erased — try refreshing."
    );
  }
  return result.items;
}

function saveMyBets(bets) {
  return writeNormalizedList(MY_BETS_KEY, bets, normalizeBet);
}

function loadBankrollSettings() {
  try {
    const raw = localStorage.getItem(BANKROLL_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return { startingBankroll: Math.max(0, Number(parsed.startingBankroll) || 0) };
    }
  } catch {
    /* ignore */
  }
  return { startingBankroll: 0 };
}

function saveBankrollSettings(settings) {
  try {
    localStorage.setItem(
      BANKROLL_KEY,
      JSON.stringify({ startingBankroll: Math.max(0, Number(settings.startingBankroll) || 0) })
    );
    return true;
  } catch (error) {
    showBanner(storageErrorMessage(error), { type: "error" });
    return false;
  }
}

function normalizeModelTrackerEntry(entry) {
  if (!entry) return entry;
  const stakeRaw = entry.stake;
  const stake =
    stakeRaw == null || stakeRaw === ""
      ? null
      : Number.isFinite(Number(stakeRaw))
        ? Math.round(Number(stakeRaw) * 100) / 100
        : null;
  return ensureModelTrackerProfit({
    ...entry,
    id: entry.id || createBetId(),
    eventId: entry.eventId || null,
    league: entry.league || null,
    scheduleDate: entry.scheduleDate || null,
    matchup: entry.matchup || "Unknown matchup",
    pick: entry.pick || entry.outcomeLabel || "Pick",
    outcomeLabel: entry.outcomeLabel || entry.pick || "Pick",
    confidence: entry.confidence ?? null,
    modelDecimalOdds:
      entry.modelDecimalOdds ?? resolveOddsToDecimal(entry.modelOdds) ?? null,
    userDecimalOdds:
      entry.userDecimalOdds ?? resolveOddsToDecimal(entry.userOdds) ?? null,
    stake,
    status: entry.status || "pending",
    profit: entry.profit ?? null,
    createdAt: entry.createdAt || new Date().toISOString(),
  });
}

function ensureModelTrackerProfit(entry) {
  if (!entry) return entry;
  if (entry.status === "pending") return { ...entry, profit: null };
  const userOdds = Number(entry.userDecimalOdds);
  const stake = Number(entry.stake);
  if (!Number.isFinite(userOdds) || !Number.isFinite(stake)) return { ...entry, profit: null };
  return {
    ...entry,
    profit: calcBetProfitDecimal(userOdds, stake, entry.status === "won"),
  };
}

function loadModelTracker() {
  const result = loadNormalizedList(MODEL_TRACKER_KEY, normalizeModelTrackerEntry);
  if (!result.ok && result.corrupt) {
    warnStorageOnce(
      MODEL_TRACKER_KEY,
      "Model tracker data could not be read. Your data was not erased — try refreshing."
    );
  }
  return result.items;
}

function saveModelTracker(entries) {
  return writeNormalizedList(MODEL_TRACKER_KEY, entries, normalizeModelTrackerEntry);
}

function loadModelBankrollSettings() {
  try {
    const raw = localStorage.getItem(MODEL_BANKROLL_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return { startingBankroll: Math.max(0, Number(parsed.startingBankroll) || 0) };
    }
  } catch {
    /* ignore */
  }
  return { startingBankroll: 0 };
}

function saveModelBankrollSettings(settings) {
  try {
    localStorage.setItem(
      MODEL_BANKROLL_KEY,
      JSON.stringify({ startingBankroll: Math.max(0, Number(settings.startingBankroll) || 0) })
    );
    return true;
  } catch (error) {
    showBanner(storageErrorMessage(error), { type: "error" });
    return false;
  }
}

function flushModelTrackerEditsFromDom() {
  if (!modelTrackerViewEl || modelTrackerViewEl.classList.contains("hidden")) return;

  modelTrackerViewEl.querySelectorAll(".model-tracker-stake-input").forEach((input) => {
    updateModelTrackerStake(input.dataset.entryId, input.value, { silent: true });
  });
  modelTrackerViewEl.querySelectorAll(".model-tracker-odds-input").forEach((input) => {
    updateModelTrackerUserOdds(input.dataset.entryId, input.value, { silent: true });
  });
  modelTrackerViewEl.querySelectorAll(".model-tracker-date-input").forEach((input) => {
    updateModelTrackerDate(input.dataset.entryId, input.value, { silent: true });
  });
}

function findGameForModelEntry(entry) {
  if (!entry?.eventId) return null;
  if (lastPayload?.games?.length) {
    return lastPayload.games.find((game) => String(game.eventId) === String(entry.eventId)) || null;
  }
  return null;
}

function autoSettleModelBet(entry) {
  const normalized = normalizeModelTrackerEntry(entry);
  if (normalized.status !== "pending") return normalized;
  const game = findGameForModelEntry(normalized);
  if (!game) return normalized;
  const winner = winnerFromGame(game);
  if (!winner) return normalized;
  const pickLabel = normalized.outcomeLabel || normalized.pick;
  const won =
    winner === "Draw" ? namesMatch(pickLabel, "Draw") : pickMatchesWinner(pickLabel, winner);
  return ensureModelTrackerProfit({
    ...normalized,
    status: won ? "won" : "lost",
    settledAt: new Date().toISOString(),
  });
}

function autoSettleModelBets(entries) {
  return entries.map(autoSettleModelBet);
}

function summarizeModelTracker(entries, bankrollSettings = loadModelBankrollSettings()) {
  const settled = entries.filter((entry) => entry.status === "won" || entry.status === "lost");
  const pending = entries.filter((entry) => entry.status === "pending");
  const wins = settled.filter((entry) => entry.status === "won").length;
  const losses = settled.filter((entry) => entry.status === "lost").length;
  const profit = settled.reduce((sum, entry) => sum + Number(entry.profit || 0), 0);
  const staked = settled.reduce((sum, entry) => sum + Number(entry.stake || 0), 0);
  const pendingStake = pending.reduce((sum, entry) => sum + Number(entry.stake || 0), 0);
  const roiPct = staked > 0 ? Math.round((profit / staked) * 1000) / 10 : null;
  const startingBankroll = Math.round((bankrollSettings.startingBankroll || 0) * 100) / 100;
  const currentBankroll = Math.round((startingBankroll + profit) * 100) / 100;
  const available = Math.round((currentBankroll - pendingStake) * 100) / 100;
  return {
    wins,
    losses,
    pending: pending.length,
    profit: Math.round(profit * 100) / 100,
    staked: Math.round(staked * 100) / 100,
    pendingStake: Math.round(pendingStake * 100) / 100,
    roiPct,
    startingBankroll,
    currentBankroll,
    available,
  };
}

function isModelPickTracked(eventId) {
  const id = String(eventId || "");
  return loadModelTracker().some(
    (entry) => entry.status === "pending" && String(entry.eventId) === id
  );
}

function addModelTrackerFromGame(game) {
  if (!game?.prediction) return;
  const eventId = game.eventId;
  if (isModelPickTracked(eventId)) {
    switchView("model-tracker");
    showBanner("This pick is already tracked.", { autoHideMs: 3000, type: "success" });
    return;
  }

  const entry = normalizeModelTrackerEntry({
    id: createBetId(),
    eventId,
    league: game.league || sportSelect.value,
    scheduleDate: lastPayload?.scheduleDate || getSelectedDate(),
    matchup: game.matchup || "",
    pick: game.prediction.outcomeLabel || game.prediction.predictedWinner || "",
    outcomeLabel: game.prediction.outcomeLabel || "",
    confidence: game.prediction.confidence ?? null,
    userDecimalOdds: null,
    stake: null,
    status: "pending",
    profit: null,
    createdAt: new Date().toISOString(),
  });

  const entries = loadModelTracker();
  entries.unshift(entry);
  saveModelTracker(autoSettleModelBets(entries));
  modelTrackerFocusId = entry.id;
  switchView("model-tracker");
  showBanner("Pick added — enter your stake and odds.", { autoHideMs: 3500, type: "success" });
}

function deleteModelTrackerEntry(entryId) {
  saveModelTracker(loadModelTracker().filter((entry) => entry.id !== entryId));
  renderModelTrackerView();
}

function updateModelTrackerDate(entryId, dateValue, { silent = false } = {}) {
  const entries = loadModelTracker().map((entry) => {
    if (entry.id !== entryId) return entry;
    return normalizeModelTrackerEntry({ ...entry, createdAt: parseBetDateInput(dateValue) });
  });
  saveModelTracker(entries);
  if (!silent) renderModelTrackerView();
}

function updateModelTrackerStake(entryId, value, { silent = false } = {}) {
  const stake = parseStakeInput(value);
  const entries = loadModelTracker().map((entry) => {
    if (entry.id !== entryId) return entry;
    const nextStake = stake ?? entry.stake;
    const next = normalizeModelTrackerEntry({ ...entry, stake: nextStake });
    return ensureModelTrackerProfit(next);
  });
  saveModelTracker(entries);
  if (!silent) renderModelTrackerView();
}

function updateModelTrackerUserOdds(entryId, value, { silent = false } = {}) {
  const userDecimalOdds = parseOddsInput(value);
  const entries = loadModelTracker().map((entry) => {
    if (entry.id !== entryId) return entry;
    const nextOdds = userDecimalOdds ?? entry.userDecimalOdds;
    const next = normalizeModelTrackerEntry({ ...entry, userDecimalOdds: nextOdds });
    return ensureModelTrackerProfit(next);
  });
  saveModelTracker(entries);
  if (!silent) renderModelTrackerView();
}

function settleModelTrackerManual(entryId, won) {
  const entries = loadModelTracker().map((entry) => {
    if (entry.id !== entryId || entry.status !== "pending") return entry;
    return ensureModelTrackerProfit({
      ...normalizeModelTrackerEntry(entry),
      status: won ? "won" : "lost",
      settledAt: new Date().toISOString(),
      manual: true,
    });
  });
  saveModelTracker(entries);
  renderModelTrackerView();
}

function exportModelTrackerToClipboard(entries) {
  const header = ["Date", "Game", "Pick", "Stake", "Odds", "W/L", "P/L"].join("\t");
  const lines = entries.map((entry) => {
    const normalized = normalizeModelTrackerEntry(entry);
    const wl = normalized.status === "won" ? "W" : normalized.status === "lost" ? "L" : "";
    const user = normalized.userDecimalOdds != null ? formatDecimalOdds(normalized.userDecimalOdds) : "";
    const profit =
      normalized.profit != null
        ? formatPlainMoney(normalized.profit)
        : normalized.status !== "pending" &&
            normalized.userDecimalOdds != null &&
            normalized.stake != null
          ? formatPlainMoney(
              calcBetProfitDecimal(
                normalized.userDecimalOdds,
                normalized.stake,
                normalized.status === "won"
              )
            )
          : "";
    return [
      formatBetDateShort(normalized.scheduleDate || normalized.createdAt),
      normalized.matchup,
      normalized.outcomeLabel || normalized.pick,
      normalized.stake != null ? normalized.stake.toFixed(2) : "",
      user,
      wl,
      profit,
    ].join("\t");
  });
  return [header, ...lines].join("\n");
}

function loadModelPickCache() {
  try {
    const raw = localStorage.getItem(MODEL_PICKS_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function cacheModelPick(eventId, pick) {
  if (!eventId || !pick || pick.status !== "graded") return;
  const cache = loadModelPickCache();
  cache[eventId] = {
    predicted: pick.predicted,
    outcomeLabel: pick.outcomeLabel,
    actual: pick.actual,
    correct: pick.correct,
    homeScore: pick.homeScore,
    awayScore: pick.awayScore,
    confidence: pick.confidence,
    status: "graded",
  };
  localStorage.setItem(MODEL_PICKS_KEY, JSON.stringify(cache));
}

function findGameForLeg(leg) {
  if (!leg?.eventId) return null;
  if (lastPayload?.games?.length) {
    return lastPayload.games.find((game) => String(game.eventId) === String(leg.eventId)) || null;
  }
  return null;
}

function gradeLeg(leg, resultWinner) {
  if (!resultWinner || leg.status !== "pending") return leg;
  const won =
    resultWinner === "Draw" ? namesMatch(leg.pick, "Draw") : pickMatchesWinner(leg.pick, resultWinner);
  return {
    ...leg,
    status: won ? "won" : "lost",
    resultWinner: resultWinner,
  };
}

function autoSettleLeg(leg) {
  if (leg.status !== "pending") return leg;
  const game = findGameForLeg(leg);
  if (!game) return leg;
  const winner = winnerFromGame(game);
  if (!winner) return leg;
  return gradeLeg(leg, winner);
}

function settleBetFromLegs(bet, legs) {
  if (bet.status !== "pending") return { ...bet, legs };

  if (legs.some((leg) => leg.status === "lost")) {
    return {
      ...bet,
      legs,
      status: "lost",
      profit: calcBetProfitDecimal(bet.decimalOdds, bet.stake, false),
      settledAt: new Date().toISOString(),
    };
  }

  if (bet.type === "parlay") {
    if (legs.length > 0 && legs.every((leg) => leg.status === "won")) {
      return {
        ...bet,
        legs,
        status: "won",
        profit: calcBetProfitDecimal(bet.decimalOdds, bet.stake, true),
        settledAt: new Date().toISOString(),
      };
    }
    return { ...bet, legs };
  }

  const leg = legs[0];
  if (leg?.status === "won") {
    return {
      ...bet,
      legs,
      status: "won",
      resultWinner: leg.resultWinner,
      profit: calcBetProfitDecimal(bet.decimalOdds, bet.stake, true),
      settledAt: new Date().toISOString(),
    };
  }
  if (leg?.status === "lost") {
    return {
      ...bet,
      legs,
      status: "lost",
      resultWinner: leg.resultWinner,
      profit: calcBetProfitDecimal(bet.decimalOdds, bet.stake, false),
      settledAt: new Date().toISOString(),
    };
  }
  return { ...bet, legs };
}

function autoSettleBet(bet) {
  const normalized = normalizeBet(bet);
  if (normalized.status !== "pending") return normalized;
  const legs = normalized.legs.map(autoSettleLeg);
  return settleBetFromLegs(normalized, legs);
}

function autoSettleMyBets(bets) {
  return bets.map(autoSettleBet);
}

function betLabel(bet) {
  const normalized = normalizeBet(bet);
  if (normalized.type === "parlay") {
    return `Parlay (${normalized.legs.length} legs)`;
  }
  return normalized.legs[0]?.matchup || "Single bet";
}

function renderBetLegsDetail(bet) {
  const normalized = normalizeBet(bet);
  return normalized.legs
    .map((leg) => {
      const legClass =
        leg.status === "won" ? "leg-won" : leg.status === "lost" ? "leg-lost" : "leg-pending";
      const legOdds = leg.legDecimalOdds != null ? ` @ ${formatOddsDisplay(leg.legDecimalOdds)}` : "";
      const legResult =
        leg.resultWinner && leg.status !== "pending" ? ` · ${leg.status === "won" ? "✓" : "✗"} ${leg.resultWinner}` : "";
      return `<li class="bet-leg-line ${legClass}">${escapeHtml(leg.matchup)} — ${escapeHtml(leg.pick)}${legOdds}${legResult}</li>`;
    })
    .join("");
}

function betGameLabel(bet) {
  const normalized = normalizeBet(bet);
  if (normalized.type === "parlay") {
    return normalized.legs.map((leg) => leg.matchup || leg.pick).join(" + ");
  }
  return normalized.legs[0]?.matchup || normalized.legs[0]?.pick || "Bet";
}

function summarizeMyBets(bets, bankrollSettings = loadBankrollSettings()) {
  const settled = bets.filter((bet) => bet.status === "won" || bet.status === "lost");
  const pending = bets.filter((bet) => bet.status === "pending");
  const wins = settled.filter((bet) => bet.status === "won").length;
  const losses = settled.filter((bet) => bet.status === "lost").length;
  const profit = settled.reduce((sum, bet) => sum + Number(bet.profit || 0), 0);
  const staked = settled.reduce((sum, bet) => sum + Number(bet.stake || 0), 0);
  const pendingStake = pending.reduce((sum, bet) => sum + Number(bet.stake || 0), 0);
  const roiPct = staked > 0 ? Math.round((profit / staked) * 1000) / 10 : null;
  const startingBankroll = Math.round((bankrollSettings.startingBankroll || 0) * 100) / 100;
  const currentBankroll = Math.round((startingBankroll + profit) * 100) / 100;
  const available = Math.round((currentBankroll - pendingStake) * 100) / 100;
  return {
    wins,
    losses,
    pending: pending.length,
    profit: Math.round(profit * 100) / 100,
    staked: Math.round(staked * 100) / 100,
    pendingStake: Math.round(pendingStake * 100) / 100,
    roiPct,
    total: bets.length,
    startingBankroll,
    currentBankroll,
    available,
  };
}

function splitSheetRow(line, delimiter) {
  if (delimiter === "\t") return line.split("\t").map((cell) => cell.trim());
  const cells = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (char === delimiter && !inQuotes) {
      cells.push(current.trim());
      current = "";
      continue;
    }
    current += char;
  }
  cells.push(current.trim());
  return cells;
}

function detectSheetDelimiter(lines) {
  const sample = lines.find((line) => line.trim()) || "";
  const tabCount = (sample.match(/\t/g) || []).length;
  const commaCount = (sample.match(/,/g) || []).length;
  return tabCount >= commaCount ? "\t" : ",";
}

function normalizeSheetHeader(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

function parseSheetDate(value) {
  const text = String(value ?? "").trim();
  if (!text) return new Date().toISOString();
  const slash = text.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (slash) {
    const first = Number(slash[1]);
    const second = Number(slash[2]);
    const year = slash[3];
    let day = first;
    let month = second;
    if (first > 12 && second <= 12) {
      day = first;
      month = second;
    } else if (second > 12 && first <= 12) {
      day = second;
      month = first;
    }
    return new Date(`${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}T12:00:00`).toISOString();
  }
  const parsed = new Date(text);
  return Number.isNaN(parsed.getTime()) ? new Date().toISOString() : parsed.toISOString();
}

function parseSheetStake(value) {
  return parseStakeInput(value);
}

function parseSheetOdds(value) {
  const raw = sanitizeSheetCell(value);
  const text = cleanNumericCell(raw);
  if (!text) return null;

  const hasAmericanSign = /^[+-]/.test(String(value ?? "").trim().replace(/[\s$£€¥₹]/g, ""));

  if (!hasAmericanSign) {
    const decimal = parseDecimalOddsInput(text);
    if (decimal != null) return decimal;
  }

  const american = parseAmericanOddsInput(text);
  if (american != null) return americanToDecimal(american);

  return parseDecimalOddsInput(text);
}

function parseSheetWl(value) {
  const token = String(value ?? "").trim().toUpperCase();
  if (token === "W" || token === "WIN" || token === "WON") return "won";
  if (token === "L" || token === "LOSS" || token === "LOST") return "lost";
  return "pending";
}

function sanitizeSheetCell(value) {
  const text = String(value ?? "").trim();
  if (!text) return "";
  if (text.startsWith("=")) return "";
  return text;
}

function readSheetCell(cells, index) {
  if (index < 0 || index >= cells.length) return "";
  return sanitizeSheetCell(cells[index]);
}

function mapSheetColumns(headers) {
  const normalized = headers.map(normalizeSheetHeader);
  const find = (...names) => normalized.findIndex((header) => names.includes(header));
  return {
    date: find("dateplaced", "date", "placed"),
    game: find("game", "matchup", "event", "description"),
    stake: find("bet", "stake", "amount", "wager"),
    odds: find("odd", "odds", "price", "decimal"),
    wl: find("wl", "winlose", "result", "outcome"),
    index: find("d", "id", "no", "num"),
  };
}

function detectSheetBankroll(parsedLines, headerIndex) {
  for (let i = 0; i < Math.min(headerIndex, 4); i += 1) {
    const amounts = (parsedLines[i] || [])
      .map((cell) => parseSheetStake(sanitizeSheetCell(cell)))
      .filter((value) => value != null && value > 0);
    if (amounts.length === 1) return amounts[0];
  }
  return null;
}

function parseSheetPaste(text) {
  const cleaned = String(text || "")
    .replace(/^\uFEFF/, "")
    .trim();
  const lines = cleaned
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) return { rows: [], errors: ["Paste or file is empty."], bankroll: null };

  const delimiter = detectSheetDelimiter(lines);
  const parsedLines = lines.map((line) => splitSheetRow(line, delimiter));
  let headerIndex = parsedLines.findIndex((cells) => {
    const normalized = cells.map(normalizeSheetHeader);
    const matches = ["game", "bet", "odd", "wl"].filter((key) => normalized.includes(key)).length;
    return matches >= 2;
  });
  if (headerIndex < 0) {
    headerIndex = parsedLines.findIndex((cells) => {
      const normalized = cells.map(normalizeSheetHeader);
      return normalized.includes("game") || normalized.includes("bet") || normalized.includes("odd");
    });
  }
  if (headerIndex < 0) headerIndex = 0;

  const detectedBankroll = detectSheetBankroll(parsedLines, headerIndex);

  const headers = parsedLines[headerIndex];
  const columns = mapSheetColumns(headers);
  const dataLines = parsedLines.slice(headerIndex + 1);
  const rows = [];
  const errors = [];

  dataLines.forEach((cells, index) => {
    const rowNumber = headerIndex + index + 2;
    const game =
      columns.game >= 0
        ? readSheetCell(cells, columns.game)
        : columns.index >= 0
          ? readSheetCell(cells, columns.index + 1)
          : readSheetCell(cells, 1);
    const stake = columns.stake >= 0 ? readSheetCell(cells, columns.stake) : readSheetCell(cells, 3);
    const odds = columns.odds >= 0 ? readSheetCell(cells, columns.odds) : readSheetCell(cells, 4);
    const wl = columns.wl >= 0 ? readSheetCell(cells, columns.wl) : readSheetCell(cells, 5);
    const dateValue = columns.date >= 0 ? readSheetCell(cells, columns.date) : "";

    if (!game) return;

    const stakeAmount = parseSheetStake(stake);
    const decimalOdds = parseSheetOdds(odds);
    const status = parseSheetWl(wl);

    if (stakeAmount == null) {
      errors.push(`Row ${rowNumber}: invalid stake "${stake || "(empty)"}".`);
      return;
    }
    if (decimalOdds == null) {
      const hint = String(cells[columns.odds >= 0 ? columns.odds : 4] ?? "").trim().startsWith("=")
        ? " (looks like a formula — export CSV from Google Sheets instead)"
        : "";
      errors.push(`Row ${rowNumber}: invalid odds "${odds || "(empty)"}"${hint}.`);
      return;
    }

    const profit =
      status === "pending"
        ? null
        : calcBetProfitDecimal(decimalOdds, stakeAmount, status === "won");

    rows.push({
      createdAt: dateValue ? parseSheetDate(dateValue) : new Date().toISOString(),
      matchup: String(game).trim(),
      pick: String(game).trim(),
      stake: stakeAmount,
      decimalOdds,
      status,
      profit,
      type: "single",
      legs: [
        {
          id: createBetId(),
          matchup: String(game).trim(),
          pick: String(game).trim(),
          legDecimalOdds: decimalOdds,
          status,
          resultWinner: null,
        },
      ],
    });
  });

  if (!rows.length && !errors.length) {
    errors.push(
      "No bet rows found. Export from Google Sheets as CSV (File → Download → CSV), or paste values only — formula cells are ignored."
    );
  }
  return { rows, errors, bankroll: detectedBankroll };
}

function parseSheetFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(parseSheetPaste(String(reader.result || "")));
    reader.onerror = () => reject(new Error("Could not read that file."));
    reader.readAsText(file);
  });
}

function importPreviewMarkup(rows, errors, bankroll) {
  const errorBlock = errors.length
    ? `<div class="import-errors">${errors.map((error) => `<p>${escapeHtml(error)}</p>`).join("")}</div>`
    : "";
  if (!rows.length) return errorBlock || `<p class="field-hint">No rows to import yet.</p>`;

  const previewRows = rows
    .slice(0, 12)
    .map(
      (row) => `
      <tr>
        <td>${formatBetDateShort(row.createdAt)}</td>
        <td>${escapeHtml(row.matchup)}</td>
        <td>$${row.stake.toFixed(2)}</td>
        <td>${formatDecimalOdds(row.decimalOdds)}</td>
        <td>${row.status === "won" ? "W" : row.status === "lost" ? "L" : "—"}</td>
        <td>${row.profit != null ? formatPlainMoney(row.profit) : "—"}</td>
      </tr>
    `
    )
    .join("");
  const more = rows.length > 12 ? `<p class="field-hint">Showing first 12 of ${rows.length} rows.</p>` : "";
  const warn = errors.length ? `<p class="field-hint">${errors.length} row(s) skipped — valid rows shown below.</p>` : "";
  const bankrollNote =
    bankroll != null
      ? `<p class="field-hint">Detected starting bankroll <strong>$${bankroll.toFixed(2)}</strong> — will apply on import if yours is currently $0.</p>`
      : "";
  const totalPl = rows.reduce((sum, row) => sum + Number(row.profit || 0), 0);
  return `
    <div class="import-confirm-bar">
      <div>
        <strong>${rows.length} bet${rows.length === 1 ? "" : "s"} ready</strong>
        <span class="import-confirm-meta">Settled P/L from file: ${formatMoney(totalPl)}</span>
      </div>
      <div class="import-confirm-actions">
        <button type="button" class="primary-btn" id="confirm-import-btn">Add to bet history</button>
        <button type="button" class="bet-action-btn" id="cancel-import-btn">Cancel</button>
      </div>
    </div>
    ${errorBlock}
    ${warn}
    ${bankrollNote}
    ${more}
    <div class="import-preview-wrap">
      <table class="import-preview-table">
        <thead><tr><th>Date</th><th>Game</th><th>Stake</th><th>Odds</th><th>W/L</th><th>P/L</th></tr></thead>
        <tbody>${previewRows}</tbody>
      </table>
    </div>
  `;
}

async function handleSheetFileUpload(file) {
  if (!file) return;
  try {
    importPreviewRows = await parseSheetFile(file);
    importDetectedBankroll = importPreviewRows.bankroll ?? null;
    sheetPasteDraft = "";
    importFileName = file.name;
    renderMyBetsView();
    if (importPreviewRows.rows.length) {
      showBanner(`Loaded ${importPreviewRows.rows.length} bet(s). Click "Add to bet history" to save them.`);
      requestAnimationFrame(() => {
        myBetsViewEl.querySelector(".import-collapse")?.setAttribute("open", "");
        myBetsViewEl.querySelector("#import-preview")?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    } else if (importPreviewRows.errors.length) {
      showBanner(importPreviewRows.errors[0]);
    }
  } catch (error) {
    showBanner(error.message || "Could not read that file.");
  }
}

function finalizeImportedBet(row) {
  const status = row.status || "pending";
  const profit =
    status === "pending"
      ? null
      : row.profit ?? calcBetProfitDecimal(row.decimalOdds, row.stake, status === "won");
  const legs = (row.legs || []).map((leg) => ({
    ...leg,
    status: leg.status || status,
  }));
  return normalizeBet({
    id: createBetId(),
    createdAt: row.createdAt || new Date().toISOString(),
    type: row.type || "single",
    stake: row.stake,
    decimalOdds: row.decimalOdds,
    status,
    profit,
    legs,
    imported: true,
  });
}

function importBetsFromRows(rows) {
  const existing = loadMyBets();
  const imported = rows.map(finalizeImportedBet);
  saveMyBets([...imported, ...existing]);

  if (importDetectedBankroll != null && loadBankrollSettings().startingBankroll <= 0) {
    saveBankrollSettings({ startingBankroll: importDetectedBankroll });
  }

  importPreviewRows = null;
  importDetectedBankroll = null;
  sheetPasteDraft = "";
  importFileName = "";
  hideBanner();
  renderMyBetsView();
  const totalPl = imported.reduce((sum, bet) => sum + Number(bet.profit || 0), 0);
  showBanner(
    `Added ${imported.length} bet${imported.length === 1 ? "" : "s"} to history. Settled P/L: ${formatMoney(totalPl)}.`,
    { autoHideMs: 5000, type: "success" }
  );
  requestAnimationFrame(() => {
    myBetsViewEl.querySelector(".my-bets-table-card")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function cancelImportPreview() {
  importPreviewRows = null;
  importDetectedBankroll = null;
  importFileName = "";
  renderMyBetsView();
}

function exportBetsToClipboard(bets) {
  const header = ["Date Placed", "Game", "Bet", "Odd", "W/L", "P/L"].join("\t");
  const lines = bets.map((bet) => {
    const normalized = normalizeBet(bet);
    const wl = normalized.status === "won" ? "W" : normalized.status === "lost" ? "L" : "";
    const profit =
      normalized.profit != null
        ? formatPlainMoney(normalized.profit)
        : normalized.status === "pending"
          ? ""
          : formatPlainMoney(calcBetProfitDecimal(normalized.decimalOdds, normalized.stake, normalized.status === "won"));
    return [
      formatBetDateShort(normalized.createdAt),
      betGameLabel(normalized),
      normalized.stake.toFixed(2),
      formatDecimalOdds(normalized.decimalOdds),
      wl,
      profit,
    ].join("\t");
  });
  return [header, ...lines].join("\n");
}

function addMyBet(entry, format = getOddsFormat()) {
  const game = String(entry.game ?? "").trim();
  const stake = parseStakeInput(entry.stake);
  const decimalOdds = parseOddsInput(entry.odds ?? entry.decimalOdds, format);

  if (!game) {
    showBanner("Enter a game.", { type: "error" });
    return false;
  }
  if (stake == null) {
    showBanner("Enter bet amount.", { type: "error" });
    return false;
  }
  if (decimalOdds == null) {
    showBanner(`Enter odds (e.g. ${formatOddsInputPlaceholder(format)}).`, { type: "error" });
    return false;
  }

  const legs = [
    {
      id: createBetId(),
      matchup: game,
      pick: game,
      legDecimalOdds: decimalOdds,
      eventId: entry.eventId || null,
      league: entry.league || null,
      scheduleDate: entry.scheduleDate || null,
      status: "pending",
      resultWinner: null,
    },
  ];

  const bets = loadMyBets();
  bets.unshift(
    normalizeBet({
      id: createBetId(),
      createdAt: parseBetDateInput(entry.betDate),
      type: "single",
      stake,
      decimalOdds,
      legs,
      status: "pending",
      profit: null,
      resultWinner: null,
    })
  );
  if (!saveMyBets(autoSettleMyBets(bets))) {
    return false;
  }
  betFormDraft = defaultBetFormDraft();
  renderMyBetsView();
  return true;
}

function deleteMyBet(betId) {
  saveMyBets(loadMyBets().filter((bet) => bet.id !== betId));
  renderMyBetsView();
}

function updateMyBetDate(betId, dateValue) {
  const bets = loadMyBets().map((bet) => {
    if (bet.id !== betId) return bet;
    return normalizeBet({ ...bet, createdAt: parseBetDateInput(dateValue) });
  });
  saveMyBets(bets);
  renderMyBetsView();
}

function settleMyBetManual(betId, won) {
  const bets = loadMyBets().map((bet) => {
    if (bet.id !== betId || bet.status !== "pending") return bet;
    const normalized = normalizeBet(bet);
    const legs = normalized.legs.map((leg) =>
      leg.status === "pending"
        ? {
            ...leg,
            status: won ? "won" : "lost",
            resultWinner: won ? leg.pick : "Manual loss",
          }
        : leg
    );
    return {
      ...normalized,
      legs,
      status: won ? "won" : "lost",
      resultWinner: won ? betLabel(normalized) : "Manual loss",
      profit: calcBetProfitDecimal(normalized.decimalOdds, normalized.stake, won),
      settledAt: new Date().toISOString(),
      manual: true,
    };
  });
  saveMyBets(bets);
  renderMyBetsView();
}

function openBetFormFromGame(game) {
  if (!game?.prediction) return;
  const pick = game.prediction.outcomeLabel || game.prediction.predictedWinner || "";
  const gameLabel = pick ? `${game.matchup || ""} — ${pick}`.replace(/^ — /, "") : game.matchup || "";
  betFormDraft = defaultBetFormDraft({
    game: gameLabel,
    linkedGame: gameLabel,
    betDate: betDateInputFromIso(lastPayload?.scheduleDate || getSelectedDate()),
    eventId: game.eventId || null,
    league: game.league || sportSelect.value,
    scheduleDate: lastPayload?.scheduleDate || getSelectedDate(),
  });
  switchView("my-bets");
}

function isBetLoggedForGame(eventId) {
  const id = String(eventId || "");
  return loadMyBets().some(
    (bet) =>
      bet.status === "pending" &&
      normalizeBet(bet).legs.some((leg) => String(leg.eventId) === id)
  );
}

function viewFromHash() {
  const hash = window.location.hash;
  if (hash === "#my-bets") return "my-bets";
  if (hash === "#model-tracker") return "model-tracker";
  if (hash === "#accuracy") return "accuracy";
  if (hash === "#predictions" || hash.startsWith("#game-")) return "predictions";
  return null;
}

function updateViewHash(view) {
  if (window.location.hash.startsWith("#game-")) return;
  const hashMap = {
    predictions: "#predictions",
    accuracy: "#accuracy",
    "my-bets": "#my-bets",
    "model-tracker": "#model-tracker",
  };
  const next = hashMap[view] || "#predictions";
  const nextUrl = `${window.location.pathname}${window.location.search}${next}`;
  const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (currentUrl !== nextUrl) {
    history.replaceState(null, "", nextUrl);
  }
}

function updateRefreshButtonLabel() {
  if (!refreshBtn) return;
  if (activeView === "my-bets" || activeView === "model-tracker") {
    refreshBtn.textContent = loadingDashboard ? "Loading…" : "Refresh scores";
    refreshBtn.title = "Reload live scores to auto-settle bets";
  } else if (activeView === "accuracy") {
    refreshBtn.textContent = loadingDashboard ? "Loading…" : "Refresh";
    refreshBtn.title = "Reload accuracy and calibration data";
  } else {
    refreshBtn.textContent = loadingDashboard ? "Loading…" : "Refresh";
    refreshBtn.title = "Reload schedule and predictions";
  }
}

function formatAccuracyRecord(summary) {
  if (!summary) return "—";
  const wins = summary.correct || 0;
  const pending = summary.pending || 0;
  const losses = Math.max(0, (summary.total || 0) - wins - pending);
  return `${wins}-${losses}`;
}

function renderCalibrationChart(buckets) {
  if (!buckets?.length) {
    return `<p class="lineup-note">No calibration buckets yet — need more graded picks.</p>`;
  }

  const width = 640;
  const height = 220;
  const pad = { top: 16, right: 16, bottom: 36, left: 40 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const barGap = 8;
  const barW = Math.max(12, (chartW - barGap * (buckets.length - 1)) / buckets.length);

  const bars = buckets
    .map((row, index) => {
      const x = pad.left + index * (barW + barGap);
      const predH = (row.avgPredictedPct / 100) * chartH;
      const actH = (row.actualWinPct / 100) * chartH;
      const predY = pad.top + chartH - predH;
      const actY = pad.top + chartH - actH;
      const label = row.confidenceRange;
      return `
        <g>
          <rect x="${x}" y="${predY}" width="${barW / 2 - 1}" height="${predH}" fill="var(--info)" opacity="0.85" rx="2" />
          <rect x="${x + barW / 2 + 1}" y="${actY}" width="${barW / 2 - 1}" height="${actH}" fill="var(--accent)" opacity="0.9" rx="2" />
          <text x="${x + barW / 2}" y="${height - 10}" text-anchor="middle" fill="var(--muted)" font-size="10">${label}</text>
        </g>
      `;
    })
    .join("");

  return `
    <svg class="calibration-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Calibration chart comparing predicted and actual win rates by confidence bucket">
      <line x1="${pad.left}" y1="${pad.top + chartH}" x2="${width - pad.right}" y2="${pad.top + chartH}" stroke="var(--panel-border)" />
      <text x="${pad.left - 8}" y="${pad.top + 8}" text-anchor="end" fill="var(--muted)" font-size="10">100%</text>
      <text x="${pad.left - 8}" y="${pad.top + chartH}" text-anchor="end" fill="var(--muted)" font-size="10">0%</text>
      ${bars}
    </svg>
    <div class="calibration-legend">
      <span class="legend-predicted">Predicted</span>
      <span class="legend-actual">Actual</span>
    </div>
  `;
}

function renderAccuracyView() {
  if (!accuracyViewEl) return;

  const sport = sportSelect.value;
  if (sport === "overview") {
    accuracyViewEl.innerHTML = `<div class="accuracy-empty"><strong>Pick a league</strong>Select a sport to view model accuracy and calibration.</div>`;
    return;
  }

  const summary = accuracyData?.summary;
  const leagueStats = summary?.byLeague?.[sport];
  const last7 = summary?.last7Days || {};
  const allTime = summary?.allTime || {};
  const recent = (accuracyData?.recentResults || []).filter((item) => item.league === sport).slice(0, 10);
  const calSummary = calibrationData?.summary;
  const calBuckets = (calibrationData?.calibration || []).filter((row) => row.picks >= 3);

  if (!summary && !calSummary) {
    accuracyViewEl.innerHTML = `
      <div class="accuracy-empty">
        <strong>Accuracy data not loaded</strong>
        Run <code>python scripts/build_pages_data.py</code> or wait for the next GitHub Actions build to generate accuracy.json and calibration.json.
      </div>
    `;
    return;
  }

  const recentHtml = recent.length
    ? `<ul class="accuracy-list">${recent
        .map((item) => {
          const resultClass = item.correct ? "acc-correct" : "acc-wrong";
          const resultLabel = item.correct ? "W" : "L";
          return `<li><span class="${resultClass}">${resultLabel}</span> ${item.matchup || item.outcomeLabel || "Pick"} · ${item.confidence ?? "—"}% · ${item.scheduleDate || item.date || ""}</li>`;
        })
        .join("")}</ul>`
    : `<p class="lineup-note">No recent graded picks for ${SPORT_LABELS[sport] || sport}.</p>`;

  const leagueBlock = leagueStats
    ? `<div class="accuracy-metric"><span class="accuracy-metric-label">${SPORT_LABELS[sport]?.split(" ")[0] || sport}</span><span class="accuracy-metric-value">${formatAccuracyRecord(leagueStats)}</span><span class="lineup-note">${leagueStats.pct ?? "—"}% hit rate</span></div>`
    : "";

  accuracyViewEl.innerHTML = `
    <section class="accuracy-hero">
      <h2 class="section-title">Model accuracy · ${escapeHtml(SPORT_LABELS[sport] || sport)}</h2>
      <div class="accuracy-summary-grid">
        <div class="accuracy-metric">
          <span class="accuracy-metric-label">Hit rate (all time)</span>
          <span class="accuracy-metric-value">${allTime.pct ?? calSummary?.winPct ?? "—"}%</span>
          <span class="lineup-note">${formatAccuracyRecord(allTime)} graded</span>
        </div>
        <div class="accuracy-metric">
          <span class="accuracy-metric-label">Last 7 days</span>
          <span class="accuracy-metric-value">${last7.pct ?? "—"}%</span>
          <span class="lineup-note">${formatAccuracyRecord(last7)} · ${last7.pending || 0} pending</span>
        </div>
        <div class="accuracy-metric">
          <span class="accuracy-metric-label">Brier / overconf.</span>
          <span class="accuracy-metric-value">${calSummary?.avgOverconfidencePct != null ? `${calSummary.avgOverconfidencePct}%` : "—"}</span>
          <span class="lineup-note">${calSummary?.graded ?? 0} picks in calibration</span>
        </div>
        ${leagueBlock}
      </div>
    </section>
    <section class="calibration-chart">
      <h3>Calibration (predicted vs actual)</h3>
      ${calBuckets.length ? renderCalibrationChart(calBuckets) : `<div class="accuracy-empty"><strong>Calibration pending</strong>Need more graded picks. Re-run the build to refresh calibration.json.</div>`}
    </section>
    <section class="accuracy-hero">
      <h3 class="section-title">Recent results</h3>
      ${recentHtml}
    </section>
  `;
}

function renderAccuracyPanel() {
  if (activeView === "accuracy") renderAccuracyView();
}

function switchView(view, { skipHashUpdate = false } = {}) {
  if (activeView === "model-tracker" && view !== "model-tracker") {
    flushModelTrackerEditsFromDom();
  }

  activeView = view;
  const isPredictions = view === "predictions";
  const isAccuracy = view === "accuracy";
  const isMyBets = view === "my-bets";
  const isModelTracker = view === "model-tracker";

  viewTabsEl?.querySelectorAll(".view-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });

  predictionsViewEl?.classList.toggle("hidden", !isPredictions);
  accuracyViewEl?.classList.toggle("hidden", !isAccuracy);
  myBetsViewEl?.classList.toggle("hidden", !isMyBets);
  modelTrackerViewEl?.classList.toggle("hidden", !isModelTracker);

  filterPanelDesktopEl?.classList.toggle("hidden", !isPredictions);
  filterOpenBtn?.classList.toggle("hidden", !isPredictions);
  document.querySelector(".page-bar-odds")?.classList.toggle("hidden", isPredictions || isAccuracy);

  if (isPredictions) {
    dashboardTitle.textContent =
      sportSelect.value === "overview"
        ? "All Sports"
        : `${SPORT_LABELS[sportSelect.value] || "Sports"}`;
    if (lastPayload) renderGames(lastPayload.games || []);
    else loadDashboard(true);
  } else if (isAccuracy) {
    dashboardTitle.textContent = "Model Accuracy";
    hideBanner();
    modelDayResultEl?.classList.add("hidden");
    renderAccuracyView();
  } else if (isMyBets) {
    dashboardTitle.textContent = "My Bet Tracker";
    hideBanner();
    modelDayResultEl?.classList.add("hidden");
    renderMyBetsView();
  } else if (isModelTracker) {
    dashboardTitle.textContent = "Model Tracker";
    hideBanner();
    modelDayResultEl?.classList.add("hidden");
    renderModelTrackerView();
  }

  updateRefreshButtonLabel();
  if (!skipHashUpdate) updateViewHash(view);
}

function renderMyBetsView() {
  if (!myBetsViewEl) return;

  syncBetFormDraftFromDom();

  const loaded = loadMyBets();
  const bets = autoSettleMyBets(loaded);
  if (betsChangedAfterSettle(loaded, bets)) {
    saveMyBets(bets);
  }
  const bankroll = loadBankrollSettings();
  const summary = summarizeMyBets(bets, bankroll);
  const draft = defaultBetFormDraft(betFormDraft);
  const oddsFormat = getOddsFormat();
  const oddsLabel = oddsFormat === "american" ? "Odds (American)" : "Odds (Decimal)";
  const oddsPlaceholder = formatOddsInputPlaceholder(oddsFormat);

  const sortedBets = [...bets].sort(
    (left, right) => new Date(right.createdAt || 0).getTime() - new Date(left.createdAt || 0).getTime()
  );

  const rows = sortedBets.length
    ? sortedBets
        .map((bet) => {
          const normalized = normalizeBet(bet);
          const statusClass =
            normalized.status === "won" ? "bet-won" : normalized.status === "lost" ? "bet-lost" : "bet-pending";
          const wlLabel = normalized.status === "won" ? "W" : normalized.status === "lost" ? "L" : "—";
          const profitCell =
            normalized.status === "pending"
              ? `<span class="bet-pending-label">—</span>`
              : `<strong class="${normalized.profit >= 0 ? "acc-correct" : "acc-wrong"}">${formatPlainMoney(normalized.profit)}</strong>`;
          const actions =
            normalized.status === "pending"
              ? `<div class="bet-row-actions">
                  <button type="button" class="bet-action-btn" data-bet-action="win" data-bet-id="${normalized.id}">W</button>
                  <button type="button" class="bet-action-btn" data-bet-action="loss" data-bet-id="${normalized.id}">L</button>
                  <button type="button" class="bet-action-btn danger" data-bet-action="delete" data-bet-id="${normalized.id}">Delete</button>
                </div>`
              : `<div class="bet-row-actions">
                  <button type="button" class="bet-action-btn danger" data-bet-action="delete" data-bet-id="${normalized.id}">Delete</button>
                </div>`;

          return `
            <tr class="${statusClass}">
              <td data-label="Date">
                <input type="date" class="bet-date-input" data-bet-id="${normalized.id}" value="${betDateInputFromIso(normalized.createdAt)}" title="Date placed">
              </td>
              <td data-label="Game">
                <strong>${escapeHtml(betGameLabel(normalized))}</strong>
                ${normalized.type === "parlay" ? `<span class="bet-pick-line">${normalized.legs.length} legs</span>` : ""}
              </td>
              <td data-label="Bet">$${Number.isFinite(Number(normalized.stake)) ? Number(normalized.stake).toFixed(2) : "—"}</td>
              <td data-label="Odd">${formatOddsDisplay(normalized.decimalOdds, oddsFormat)}</td>
              <td data-label="W/L"><span class="bet-wl-pill ${statusClass}">${wlLabel}</span></td>
              <td data-label="P/L">${profitCell}</td>
              <td data-label="Actions">${actions}</td>
            </tr>
          `;
        })
        .join("")
    : `<tr><td colspan="7" class="empty-bets-cell">No bets yet. Log one above or import from a spreadsheet below.</td></tr>`;

  const importPreview = importPreviewRows
    ? importPreviewMarkup(
        importPreviewRows.rows,
        importPreviewRows.errors,
        importPreviewRows.bankroll ?? importDetectedBankroll
      )
    : "";

  myBetsViewEl.innerHTML = `
    <section class="my-bets-hero">
      <div class="my-bets-hero-head">
        <div>
          <h2 class="section-title">Bankroll</h2>
          <p class="my-bets-note">Track your bets here. Model picks stay on the Predictions tab.</p>
        </div>
        <label class="field bankroll-field">
          <span>Starting bankroll</span>
          <div class="bankroll-input-wrap">
            <span class="input-prefix">$</span>
            <input id="starting-bankroll" type="number" min="0" step="0.01" value="${summary.startingBankroll.toFixed(2)}">
          </div>
        </label>
      </div>
      <div class="my-bets-stats">
        <article class="tracker-stat tracker-stat-highlight ${summary.profit >= 0 ? "stat-positive" : "stat-negative"}">
          <span class="tracker-stat-label">Total P/L</span>
          <strong>${formatMoney(summary.profit)}</strong>
        </article>
        <article class="tracker-stat">
          <span class="tracker-stat-label">W/L</span>
          <strong>${summary.wins}-${summary.losses}</strong>
        </article>
        <article class="tracker-stat">
          <span class="tracker-stat-label">Bankroll</span>
          <strong>$${summary.currentBankroll.toFixed(2)}</strong>
        </article>
        <article class="tracker-stat">
          <span class="tracker-stat-label">Available</span>
          <strong>$${summary.available.toFixed(2)}</strong>
        </article>
        <article class="tracker-stat">
          <span class="tracker-stat-label">At risk</span>
          <strong>$${summary.pendingStake.toFixed(2)}</strong>
        </article>
        <article class="tracker-stat">
          <span class="tracker-stat-label">Pending</span>
          <strong>${summary.pending}</strong>
        </article>
        <article class="tracker-stat">
          <span class="tracker-stat-label">ROI</span>
          <strong>${summary.roiPct != null ? `${summary.roiPct}%` : "—"}</strong>
        </article>
      </div>
    </section>

    <div class="my-bets-grid">
      <section class="my-bets-form-card panel-card">
        <h2 class="section-title">Log a bet</h2>
        <form id="my-bet-form" class="my-bet-form bet-log-form">
          <label class="field">
            <span>Date</span>
            <input id="bet-date" type="date" required value="${escapeAttr(draft.betDate ?? todayBetDateInput())}">
          </label>
          <label class="field field-game">
            <span>Game</span>
            <input id="bet-game" type="text" required placeholder="Yankees @ Red Sox — Yankees ML" value="${escapeAttr(draft.game ?? "")}">
          </label>
          <label class="field">
            <span>Stake</span>
            <div class="bankroll-input-wrap">
              <span class="input-prefix">$</span>
              <input id="bet-stake" type="text" inputmode="decimal" required placeholder="100" value="${escapeAttr(draft.stake ?? "")}">
            </div>
          </label>
          <label class="field">
            <span>${oddsLabel}</span>
            <input id="bet-odds" type="text" inputmode="decimal" required placeholder="${escapeAttr(oddsPlaceholder)}" value="${escapeAttr(draft.odds ?? "")}">
          </label>
          <button type="submit" class="btn-primary bet-form-submit">Add bet</button>
        </form>
      </section>

      <section class="my-bets-table-card panel-card">
        <div class="panel-card-head">
          <h2 class="section-title">Bet history</h2>
          <button type="button" class="btn-ghost" id="export-bets-btn">Export CSV</button>
        </div>
        <div class="my-bets-table-wrap tracker-table-wrap">
          <table class="my-bets-table sheet-style tracker-sheet-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Game</th>
                <th>Bet</th>
                <th>Odd</th>
                <th>W/L</th>
                <th>P/L</th>
                <th></th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </section>
    </div>

    <details class="my-bets-import-card panel-card import-collapse"${importPreviewRows ? " open" : ""}>
      <summary class="import-collapse-summary">
        <span class="section-title">Import from spreadsheet</span>
        <span class="import-collapse-hint">CSV upload or paste</span>
      </summary>
      <div class="import-collapse-body">
        <p class="my-bets-note">Download from Google Sheets as CSV. Columns: <strong>Date, Game, Bet, Odd, W/L</strong>. P/L is recalculated from your stake and odds.</p>

        <div class="csv-dropzone" id="csv-dropzone">
          <input id="sheet-file" type="file" accept=".csv,.tsv,.txt,text/csv,text/tab-separated-values" hidden>
          <p class="csv-dropzone-title">Drop your CSV here</p>
          <p class="csv-dropzone-sub">or</p>
          <button type="button" class="btn-secondary" id="choose-csv-btn">Choose file</button>
          ${importFileName ? `<p class="csv-dropzone-file">Selected: <strong>${escapeHtml(importFileName)}</strong></p>` : ""}
        </div>

        <details class="paste-fallback">
          <summary>Paste rows instead</summary>
          <textarea id="sheet-paste" class="sheet-paste" rows="4" placeholder="Paste spreadsheet rows here…">${escapeHtml(sheetPasteDraft)}</textarea>
          <button type="button" class="btn-ghost" id="preview-import-btn">Preview pasted rows</button>
        </details>

        <div id="import-preview" class="import-preview">${importPreview}</div>
      </div>
    </details>
  `;

  myBetsViewEl.querySelector("#my-bet-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const draftValues = collectBetFormDraftFromForm(form, betFormDraft);
    const link = resolveBetFormGameLink(betFormDraft, draftValues.game);
    try {
      const success = addMyBet({
        game: draftValues.game,
        stake: draftValues.stake,
        odds: draftValues.odds,
        betDate: draftValues.betDate,
        eventId: link.eventId,
        league: link.league,
        scheduleDate: link.scheduleDate,
      });
      if (success) {
        showBanner("Bet added to history.", { autoHideMs: 3200, type: "success" });
      }
    } catch (error) {
      showBanner(storageErrorMessage(error), { type: "error" });
    }
  });

  myBetsViewEl.querySelectorAll(".bet-date-input").forEach((input) => {
    input.addEventListener("change", () => {
      updateMyBetDate(input.dataset.betId, input.value);
    });
  });

  myBetsViewEl.querySelectorAll("[data-bet-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const betId = button.dataset.betId;
      const action = button.dataset.betAction;
      if (action === "delete") deleteMyBet(betId);
      if (action === "win") settleMyBetManual(betId, true);
      if (action === "loss") settleMyBetManual(betId, false);
    });
  });

  myBetsViewEl.querySelector("#starting-bankroll")?.addEventListener("change", (event) => {
    saveBankrollSettings({ startingBankroll: Number(event.currentTarget.value) || 0 });
    renderMyBetsView();
  });

  myBetsViewEl.querySelector("#preview-import-btn")?.addEventListener("click", () => {
    sheetPasteDraft = myBetsViewEl.querySelector("#sheet-paste")?.value || "";
    importFileName = "";
    importPreviewRows = parseSheetPaste(sheetPasteDraft);
    importDetectedBankroll = importPreviewRows.bankroll ?? null;
    renderMyBetsView();
    if (importPreviewRows.rows.length) {
      requestAnimationFrame(() => {
        myBetsViewEl.querySelector(".import-collapse")?.setAttribute("open", "");
        myBetsViewEl.querySelector("#import-preview")?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
  });

  myBetsViewEl.querySelector("#choose-csv-btn")?.addEventListener("click", () => {
    myBetsViewEl.querySelector("#sheet-file")?.click();
  });

  myBetsViewEl.querySelector("#sheet-file")?.addEventListener("change", async (event) => {
    const file = event.currentTarget.files?.[0];
    await handleSheetFileUpload(file);
    event.currentTarget.value = "";
  });

  const dropzone = myBetsViewEl.querySelector("#csv-dropzone");
  dropzone?.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone?.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });
  dropzone?.addEventListener("drop", async (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    const file = event.dataTransfer?.files?.[0];
    if (file) await handleSheetFileUpload(file);
  });

  myBetsViewEl.querySelector("#export-bets-btn")?.addEventListener("click", async () => {
    const text = exportBetsToClipboard(loadMyBets());
    try {
      await navigator.clipboard.writeText(text);
      showBanner("Bet history copied — paste into Excel or Google Sheets.", { autoHideMs: 4000, type: "success" });
    } catch {
      showBanner("Could not copy automatically. Select the export text manually.");
    }
  });
}

function renderModelTrackerView() {
  if (!modelTrackerViewEl) return;

  const loaded = loadModelTracker();
  const entries = autoSettleModelBets(loaded);
  if (betsChangedAfterSettle(loaded, entries)) {
    saveModelTracker(entries);
  }
  const bankroll = loadModelBankrollSettings();
  const summary = summarizeModelTracker(entries, bankroll);
  const oddsFormat = getOddsFormat();
  const oddsPlaceholder = formatOddsInputPlaceholder(oddsFormat);

  const sortedEntries = [...entries].sort(
    (left, right) =>
      new Date(right.createdAt || 0).getTime() - new Date(left.createdAt || 0).getTime()
  );

  const rows = sortedEntries.length
    ? sortedEntries
        .map((entry) => {
          const normalized = normalizeModelTrackerEntry(entry);
          const statusClass =
            normalized.status === "won"
              ? "bet-won"
              : normalized.status === "lost"
                ? "bet-lost"
                : "bet-pending";
          const wlLabel = normalized.status === "won" ? "W" : normalized.status === "lost" ? "L" : "—";
          const profitCell =
            normalized.status === "pending"
              ? `<span class="bet-pending-label">—</span>`
              : normalized.profit != null
                ? `<strong class="${normalized.profit >= 0 ? "acc-correct" : "acc-wrong"}">${formatPlainMoney(normalized.profit)}</strong>`
                : `<span class="bet-pending-label">—</span>`;
          const stakeValue =
            normalized.stake != null && Number.isFinite(Number(normalized.stake))
              ? Number(normalized.stake).toFixed(2)
              : "";
          const userOddsValue =
            normalized.userDecimalOdds != null
              ? formatOddsDisplay(normalized.userDecimalOdds, oddsFormat)
              : "";
          const actions =
            normalized.status === "pending"
              ? `<div class="bet-row-actions">
                  <button type="button" class="bet-action-btn" data-model-action="win" data-entry-id="${normalized.id}">W</button>
                  <button type="button" class="bet-action-btn" data-model-action="loss" data-entry-id="${normalized.id}">L</button>
                  <button type="button" class="bet-action-btn danger" data-model-action="delete" data-entry-id="${normalized.id}">Delete</button>
                </div>`
              : `<div class="bet-row-actions">
                  <button type="button" class="bet-action-btn danger" data-model-action="delete" data-entry-id="${normalized.id}">Delete</button>
                </div>`;
          const focusClass = modelTrackerFocusId === normalized.id ? " model-tracker-row-focus" : "";

          return `
            <tr class="${statusClass}${focusClass}" data-entry-id="${normalized.id}">
              <td data-label="Date">
                <input type="date" class="bet-date-input model-tracker-date-input" data-entry-id="${normalized.id}" value="${betDateInputFromIso(normalized.scheduleDate || normalized.createdAt)}" title="Schedule date">
              </td>
              <td data-label="Game / Pick">
                <div class="tracker-card-head">
                  <div class="tracker-card-title">
                    <strong>${escapeHtml(normalized.matchup)}</strong>
                    <span class="bet-pick-line">${escapeHtml(normalized.outcomeLabel || normalized.pick)}${normalized.confidence != null ? ` · ${normalized.confidence}%` : ""}</span>
                  </div>
                  <div class="tracker-card-meta">
                    <span class="bet-wl-pill ${statusClass}">${wlLabel}</span>
                    <span class="tracker-card-pl">${profitCell}</span>
                  </div>
                </div>
              </td>
              <td data-label="Stake">
                <div class="bankroll-input-wrap model-tracker-input-wrap">
                  <span class="input-prefix">$</span>
                  <input type="text" inputmode="decimal" class="model-tracker-stake-input" data-entry-id="${normalized.id}" placeholder="100" value="${escapeAttr(stakeValue)}" aria-label="Stake">
                </div>
              </td>
              <td data-label="Odds">
                <input type="text" inputmode="decimal" class="model-tracker-odds-input" data-entry-id="${normalized.id}" placeholder="${escapeAttr(oddsPlaceholder)}" value="${escapeAttr(userOddsValue)}" aria-label="Your odds">
              </td>
              <td data-label="W/L"><span class="bet-wl-pill ${statusClass}">${wlLabel}</span></td>
              <td data-label="P/L">${profitCell}</td>
              <td data-label="Actions">${actions}</td>
            </tr>
          `;
        })
        .join("")
    : `<tr><td colspan="7" class="empty-bets-cell">No tracked picks yet. Add picks from the Predictions tab using <strong>Track model pick</strong>.</td></tr>`;

  const mobileCards = sortedEntries.length
    ? sortedEntries
        .map((entry) => {
          const normalized = normalizeModelTrackerEntry(entry);
          const chipClass =
            normalized.status === "won" ? "won" : normalized.status === "lost" ? "lost" : "pending";
          const wlLabel = normalized.status === "won" ? "W" : normalized.status === "lost" ? "L" : "Pending";
          const profitLabel =
            normalized.status === "pending" || normalized.profit == null
              ? "—"
              : formatPlainMoney(normalized.profit);
          return `
            <article class="tracker-mobile-card" data-entry-id="${normalized.id}">
              <div class="tracker-mobile-card-head">
                <div>
                  <div class="tracker-mobile-card-title">${escapeHtml(normalized.matchup)}</div>
                  <div class="tracker-mobile-card-meta">${escapeHtml(normalized.outcomeLabel || normalized.pick)}${normalized.confidence != null ? ` · ${normalized.confidence}%` : ""}</div>
                </div>
                <span class="tracker-status-chip ${chipClass}">${wlLabel}</span>
              </div>
              <div class="tracker-mobile-card-meta">${betDateInputFromIso(normalized.scheduleDate || normalized.createdAt)} · P/L ${profitLabel}</div>
            </article>
          `;
        })
        .join("")
    : `<p class="empty-state">No tracked picks yet.</p>`;

  modelTrackerViewEl.innerHTML = `
    <section class="my-bets-hero model-tracker-hero">
      <div class="my-bets-hero-head">
        <div>
          <h2 class="section-title">Model picks</h2>
          <p class="my-bets-note">Track model picks you take — enter your stake and the odds you got.</p>
        </div>
        <label class="field bankroll-field">
          <span>Starting bankroll</span>
          <div class="bankroll-input-wrap">
            <span class="input-prefix">$</span>
            <input id="model-starting-bankroll" type="number" min="0" step="0.01" value="${summary.startingBankroll.toFixed(2)}">
          </div>
        </label>
      </div>
      <div class="my-bets-stats">
        <article class="tracker-stat tracker-stat-highlight ${summary.profit >= 0 ? "stat-positive" : "stat-negative"}">
          <span class="tracker-stat-label">Total P/L</span>
          <strong>${formatMoney(summary.profit)}</strong>
        </article>
        <article class="tracker-stat">
          <span class="tracker-stat-label">W/L</span>
          <strong>${summary.wins}-${summary.losses}</strong>
        </article>
        <article class="tracker-stat">
          <span class="tracker-stat-label">Bankroll</span>
          <strong>$${summary.currentBankroll.toFixed(2)}</strong>
        </article>
        <article class="tracker-stat">
          <span class="tracker-stat-label">Available</span>
          <strong>$${summary.available.toFixed(2)}</strong>
        </article>
        <article class="tracker-stat">
          <span class="tracker-stat-label">At risk</span>
          <strong>$${summary.pendingStake.toFixed(2)}</strong>
        </article>
        <article class="tracker-stat">
          <span class="tracker-stat-label">Pending</span>
          <strong>${summary.pending}</strong>
        </article>
        <article class="tracker-stat">
          <span class="tracker-stat-label">ROI</span>
          <strong>${summary.roiPct != null ? `${summary.roiPct}%` : "—"}</strong>
        </article>
      </div>
    </section>

    <section class="my-bets-table-card panel-card model-tracker-table-card">
      <div class="panel-card-head">
        <h2 class="section-title">Tracked picks</h2>
        <button type="button" class="btn-ghost" id="export-model-tracker-btn">Copy for spreadsheet</button>
      </div>
      <div class="my-bets-table-wrap tracker-table-wrap">
        <div class="tracker-mobile-cards">${mobileCards}</div>
        <table class="my-bets-table sheet-style model-tracker-table tracker-sheet-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Game / Pick</th>
              <th>Stake</th>
              <th>Odds</th>
              <th>W/L</th>
              <th>P/L</th>
              <th></th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>
  `;

  modelTrackerViewEl.querySelectorAll(".model-tracker-date-input").forEach((input) => {
    const persist = () => updateModelTrackerDate(input.dataset.entryId, input.value);
    input.addEventListener("change", persist);
    input.addEventListener("blur", persist);
  });

  modelTrackerViewEl.querySelectorAll(".model-tracker-stake-input").forEach((input) => {
    const persist = () => updateModelTrackerStake(input.dataset.entryId, input.value);
    input.addEventListener("change", persist);
    input.addEventListener("blur", persist);
  });

  modelTrackerViewEl.querySelectorAll(".model-tracker-odds-input").forEach((input) => {
    const persist = () => updateModelTrackerUserOdds(input.dataset.entryId, input.value);
    input.addEventListener("change", persist);
    input.addEventListener("blur", persist);
  });

  modelTrackerViewEl.querySelectorAll("[data-model-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const entryId = button.dataset.entryId;
      const action = button.dataset.modelAction;
      if (action === "delete") deleteModelTrackerEntry(entryId);
      if (action === "win") settleModelTrackerManual(entryId, true);
      if (action === "loss") settleModelTrackerManual(entryId, false);
    });
  });

  modelTrackerViewEl.querySelector("#model-starting-bankroll")?.addEventListener("change", (event) => {
    saveModelBankrollSettings({ startingBankroll: Number(event.currentTarget.value) || 0 });
    renderModelTrackerView();
  });

  modelTrackerViewEl.querySelector("#export-model-tracker-btn")?.addEventListener("click", async () => {
    const text = exportModelTrackerToClipboard(loadModelTracker());
    try {
      await navigator.clipboard.writeText(text);
      showBanner("Model tracker copied — paste into Excel or Google Sheets.", {
        autoHideMs: 4000,
        type: "success",
      });
    } catch {
      showBanner("Could not copy automatically. Select the export text manually.");
    }
  });

  if (modelTrackerFocusId) {
    const focusId = modelTrackerFocusId;
    modelTrackerFocusId = null;
    requestAnimationFrame(() => {
      const row = modelTrackerViewEl.querySelector(`tr[data-entry-id="${focusId}"]`);
      const input =
        row?.querySelector(".model-tracker-stake-input") || row?.querySelector(".model-tracker-odds-input");
      input?.focus();
      row?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }
}

function bindMyBetsImportActions() {
  if (!myBetsViewEl || myBetsViewEl.dataset.importActionsBound) return;
  myBetsViewEl.dataset.importActionsBound = "1";
  myBetsViewEl.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.id === "confirm-import-btn") {
      event.preventDefault();
      if (importPreviewRows?.rows?.length) importBetsFromRows(importPreviewRows.rows);
      return;
    }
    if (target.id === "cancel-import-btn") {
      event.preventDefault();
      cancelImportPreview();
    }
  });
}

function winnerFromGame(game) {
  if (isGameVoided(game)) return null;
  if (!game?.isFinal || game.homeScore == null || game.awayScore == null) return null;
  const home = Number(game.homeScore);
  const away = Number(game.awayScore);
  if (Number.isNaN(home) || Number.isNaN(away)) return null;
  if (home === away) return "Draw";
  return home > away ? game.homeTeam : game.awayTeam;
}

function namesMatch(left, right) {
  if (!left || !right) return false;
  const a = String(left).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const b = String(right).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  if (!a || !b) return false;
  if (a === b || a.includes(b) || b.includes(a)) return true;
  const aTokens = new Set(a.split(" "));
  const bTokens = new Set(b.split(" "));
  const overlap = [...aTokens].filter((token) => bTokens.has(token)).length;
  return overlap / Math.max(aTokens.size, bTokens.size) >= 0.55;
}

function pickMatchesWinner(predicted, actual) {
  return namesMatch(predicted, actual);
}

function formatScheduleDayLabel(iso) {
  if (!iso) return "this day";
  const match = String(iso).match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!match) return iso;
  const date = new Date(`${match[1]}-${match[2]}-${match[3]}T12:00:00`);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
}

function summarizeModelDayResults(games, scheduleDate, league = sportSelect.value) {
  const picks = new Map();

  const addPick = (eventId, pick) => {
    if (!pick) return;
    picks.set(String(eventId), pick);
  };

  for (const game of games || []) {
    if (!game.prediction?.predictedWinner && !game.prediction?.outcomeLabel) continue;
    const pick = resolvePickStatus(game);
    if (pick) addPick(game.eventId, pick);
  }

  if (accuracyData?.picksByEventId && scheduleDate) {
    for (const [eventId, pick] of Object.entries(accuracyData.picksByEventId)) {
      if (pick.scheduleDate !== scheduleDate) continue;
      if (league !== "overview" && pick.league !== league) continue;
      if (!pick.predicted && !pick.outcomeLabel) continue;
      if (!picks.has(String(eventId))) addPick(eventId, pick);
    }
  }

  let correct = 0;
  let wrong = 0;
  let pending = 0;

  for (const pick of picks.values()) {
    if (pick.status !== "graded") {
      pending += 1;
    } else if (pick.correct) {
      correct += 1;
    } else {
      wrong += 1;
    }
  }

  const graded = correct + wrong;
  return {
    correct,
    wrong,
    pending,
    total: graded + pending,
    graded,
    pct: graded > 0 ? Math.round((correct / graded) * 1000) / 10 : null,
  };
}

function renderModelDayResult(games) {
  if (!modelDayResultEl || sportSelect.value === "overview" || activeView !== "predictions") {
    modelDayResultEl?.classList.add("hidden");
    return;
  }

  const scheduleDate = getSelectedDate();
  const summary = summarizeModelDayResults(games, scheduleDate, sportSelect.value);

  if (!summary.total) {
    modelDayResultEl.classList.add("hidden");
    modelDayResultEl.innerHTML = "";
    return;
  }

  modelDayResultEl.classList.remove("hidden");
  const dayLabel = formatScheduleDayLabel(scheduleDate);
  const pctLabel = summary.graded > 0 ? `${summary.pct}% hit rate` : "Awaiting first result";
  const recordLabel = summary.graded > 0 ? `${summary.correct}–${summary.wrong}` : "—";

  modelDayResultEl.innerHTML = `
    <div class="model-day-result-head">
      <div>
        <h2 class="section-title">Model record · ${escapeHtml(dayLabel)}</h2>
        <p class="model-day-sub">${summary.graded} graded · ${summary.pending} pending · ${summary.total} picks total</p>
      </div>
      <div class="model-day-summary-pill">
        <span class="model-day-record">${recordLabel}</span>
        <span class="model-day-pct">${pctLabel}</span>
      </div>
    </div>
    <div class="model-day-scores">
      <article class="model-day-stat model-day-correct">
        <span class="model-day-num">${summary.correct}</span>
        <span class="model-day-label">Correct</span>
      </article>
      <article class="model-day-stat model-day-wrong">
        <span class="model-day-num">${summary.wrong}</span>
        <span class="model-day-label">Wrong</span>
      </article>
      <article class="model-day-stat model-day-pending">
        <span class="model-day-num">${summary.pending}</span>
        <span class="model-day-label">Pending</span>
      </article>
    </div>
  `;
}

function resolvePickStatus(game) {
  const eventId = String(game.eventId || "");
  const serverPick = accuracyData?.picksByEventId?.[eventId];
  const liveScores =
    game.homeScore != null && game.awayScore != null
      ? { homeScore: game.homeScore, awayScore: game.awayScore }
      : null;

  const withLiveScores = (pick) => (pick && liveScores ? { ...pick, ...liveScores } : pick);

  if (serverPick?.status === "graded") {
    const merged = withLiveScores(serverPick);
    cacheModelPick(eventId, merged);
    return merged;
  }

  const cached = loadModelPickCache()[eventId];
  if (cached?.status === "graded") return withLiveScores(cached);

  const actual = winnerFromGame(game);
  if (actual && game.prediction?.predictedWinner) {
    const correct = pickMatchesWinner(game.prediction.predictedWinner, actual);
    const pick = {
      status: "graded",
      predicted: game.prediction.predictedWinner,
      actual,
      correct,
      homeScore: game.homeScore,
      awayScore: game.awayScore,
      confidence: game.prediction.confidence,
      outcomeLabel: game.prediction.outcomeLabel,
      live: true,
    };
    cacheModelPick(eventId, pick);
    return pick;
  }

  if (serverPick) return withLiveScores(serverPick);
  if (game.prediction?.predictedWinner) {
    return {
      status: "pending",
      predicted: game.prediction.predictedWinner,
      outcomeLabel: game.prediction.outcomeLabel,
      confidence: game.prediction.confidence,
      ...(liveScores || {}),
    };
  }
  return null;
}

function pickTeamAbbrev(game, prediction) {
  if (!prediction) return "—";
  if (prediction.predictedSide === "home") return teamAbbrev(game.homeTeam);
  if (prediction.predictedSide === "away") return teamAbbrev(game.awayTeam);
  if (prediction.predictedSide === "draw") return "DRAW";
  const label = prediction.outcomeLabel || "";
  const match = label.match(/^(.+?)\s+to\s+win/i);
  if (match) return teamAbbrev(match[1]);
  return teamAbbrev(label) || label.slice(0, 10);
}

function renderGameScoreLine(game) {
  if (isGameVoided(game)) {
    return `<p class="game-score-line voided">${escapeHtml(gameStatusLabel(game))}</p>`;
  }
  if (game.homeScore == null || game.awayScore == null) return "";
  const away = teamAbbrev(game.awayTeam);
  const home = teamAbbrev(game.homeTeam);
  const prefix = game.isFinal ? "Final" : game.isLive ? "Live" : "Score";
  const cls = game.isLive ? "game-score-line live" : game.isFinal ? "game-score-line final" : "game-score-line";
  return `<p class="${cls}" data-live-score="${game.eventId}">${prefix}: ${away} <span class="tabular-nums">${game.awayScore}</span> – ${home} <span class="tabular-nums">${game.homeScore}</span></p>`;
}

function renderPickStatusBadge(game) {
  const pick = resolvePickStatus(game);
  if (!pick) return "";
  if (isGameVoided(game)) {
    return `<span class="pick-status pick-voided">${escapeHtml(gameStatusLabel(game))}</span>`;
  }
  if (pick.status === "pending") {
    return `<span class="pick-status pick-pending">${game.isLive ? "In progress" : game.isDelayed ? "Delayed" : "Awaiting result"}</span>`;
  }
  if (pick.correct) {
    return `<span class="pick-status pick-won">Correct</span>`;
  }
  return `<span class="pick-status pick-lost">Wrong</span>`;
}

function renderModelPickBlock(game) {
  const pick = resolvePickStatus(game);
  if (!pick || pick.status !== "graded") return "";
  const actual = pick.actual || "Winner";
  return pick.correct
    ? `<p class="model-pick-result pick-won">${escapeHtml(actual)} won — model was correct</p>`
    : `<p class="model-pick-result pick-lost">${escapeHtml(actual)} won — model was wrong</p>`;
}

function leagueTimezone(sport) {
  return LEAGUE_TIMEZONES[sport] || Intl.DateTimeFormat().resolvedOptions().timeZone;
}

function gameScheduleDate(startDate, league) {
  if (!startDate) return null;
  const tz = leagueTimezone(league);
  try {
    const date = new Date(startDate);
    if (Number.isNaN(date.getTime())) return null;
    const parts = new Intl.DateTimeFormat("en-CA", {
      timeZone: tz,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).formatToParts(date);
    const pick = (type) => parts.find((part) => part.type === type)?.value;
    const year = pick("year");
    const month = pick("month");
    const day = pick("day");
    if (!year || !month || !day) return null;
    return `${year}-${month}-${day}`;
  } catch {
    return null;
  }
}

function filterGamesForScheduleDate(games, scheduleDate, league) {
  if (!scheduleDate || !Array.isArray(games)) return games || [];
  return games.filter((game) => {
    const gameDate = gameScheduleDate(game.startDate, league);
    if (!gameDate) return true;
    return gameDate === scheduleDate;
  });
}

function applyScheduleDateToPayload(payload, scheduleDate, league) {
  if (!payload || !scheduleDate || !league) return payload;
  const snapshotMatchesDate = payload.scheduleDate === scheduleDate;
  const shouldFilterByStartDate =
    payload._liveFallback ||
    payload.liveScheduleOnly ||
    payload.source === "espn-live" ||
    (payload.scheduleDate && !snapshotMatchesDate);
  if (!shouldFilterByStartDate) {
    const games = payload.games || [];
    return { ...payload, games, gameCount: games.length };
  }
  const games = filterGamesForScheduleDate(payload.games || [], scheduleDate, league);
  return {
    ...payload,
    games,
    gameCount: games.length,
  };
}

function describeScheduleSource(payload) {
  if (!payload) return "—";
  if (payload._liveFallback || payload.source === "espn-live" || payload.liveScheduleOnly) {
    if (lastLiveScoreAt && liveScoresToggle?.checked) return "Live ESPN + scores";
    return "Live ESPN";
  }
  if (payload._dateFallback) return "No snapshot";
  if (lastLiveScoreAt && liveScoresToggle?.checked) return "Snapshot + scores";
  return IS_STATIC_HOST ? "Snapshot" : "Live server";
}

function leagueDateParts(timeZone, baseDate = new Date(), daysAhead = 0) {
  const shifted = new Date(baseDate);
  shifted.setDate(shifted.getDate() + daysAhead);
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "numeric",
    hour12: false,
  }).formatToParts(shifted);
  const pick = (type) => parts.find((part) => part.type === type)?.value;
  return {
    iso: `${pick("year")}-${pick("month")}-${pick("day")}`,
    hour: Number(pick("hour") ?? 0),
  };
}

function leagueDateIso(sport, daysAhead = 0) {
  const todayIso = leagueDateParts(leagueTimezone(sport), new Date(), 0).iso;
  if (daysAhead === 0) return todayIso;
  return shiftIsoDate(todayIso, daysAhead, sport);
}

function defaultDateForSport(sport) {
  const tz = leagueTimezone(sport);
  const meta = leagueMeta(sport);
  if (meta?.defaultDate) {
    return meta.defaultDate;
  }
  const { hour } = leagueDateParts(tz);
  if (US_SCHEDULE_SPORTS.has(sport) && hour < 10) {
    return leagueDateIso(sport, -1);
  }
  return leagueDateIso(sport, 0);
}

function formatTimezoneLabel(timeZone) {
  return TIMEZONE_LABELS[timeZone] || timeZone.split("/").pop()?.replace(/_/g, " ") || timeZone;
}

function formatDateBarLabel(iso, sport = sportSelect.value, gameCount = null) {
  const today = leagueDateIso(sport, 0);
  const tomorrow = leagueDateIso(sport, 1);
  const yesterday = leagueDateIso(sport, -1);
  const date = new Date(`${iso}T12:00:00`);
  const formatted = date.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });

  let label;
  if (iso === today) label = `Today · ${formatted}`;
  else if (iso === tomorrow) label = `Tomorrow · ${formatted}`;
  else if (iso === yesterday) label = `Yesterday · ${formatted}`;
  else label = formatted;

  if (gameCount != null && gameCount > 0) {
    label += ` · ${gameCount} game${gameCount === 1 ? "" : "s"}`;
  }
  return label;
}

function formatDateChipLabel(iso, sport = sportSelect.value) {
  const today = leagueDateIso(sport, 0);
  const tomorrow = leagueDateIso(sport, 1);
  const yesterday = leagueDateIso(sport, -1);
  if (iso === today) return "Today";
  if (iso === tomorrow) return "Tomorrow";
  if (iso === yesterday) return "Yesterday";
  return new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

function formatDateHint(sport) {
  const label = SPORT_LABELS[sport] || sport;
  const prefix = label.split(" ")[0] || label;
  return `${prefix} schedule · ${formatTimezoneLabel(leagueTimezone(sport))}`;
}

function getSelectedDate() {
  return activeScheduleDate || datePickerInput?.value || defaultDateForSport(sportSelect.value);
}

function updateDateDisplayCount(count) {
  datePickerGameCount = count > 0 ? count : null;
  if (!dateDisplayBtn || !activeScheduleDate || sportSelect.value === "overview") return;
  dateDisplayBtn.textContent = formatDateBarLabel(activeScheduleDate, sportSelect.value, datePickerGameCount);
}

function syncDatePickerUI() {
  const league = sportSelect.value;
  if (league === "overview") {
    if (dateDisplayBtn) {
      dateDisplayBtn.disabled = true;
      dateDisplayBtn.textContent = "All sports view";
    }
    if (datePickerInput) datePickerInput.disabled = true;
    if (dateQuickEl) dateQuickEl.innerHTML = "";
    if (dateHintEl) dateHintEl.textContent = "";
    updateDateNavButtons();
    return;
  }

  const iso = activeScheduleDate || getSelectedDate() || defaultDateForSport(league);
  const dates = availableDatesForLeague(league);

  if (datePickerInput) {
    datePickerInput.disabled = false;
    if (dates.length) {
      datePickerInput.min = dates[0];
      datePickerInput.max = dates[dates.length - 1];
    }
    datePickerInput.value = iso;
  }
  if (dateDisplayBtn) {
    dateDisplayBtn.disabled = false;
    dateDisplayBtn.textContent = formatDateBarLabel(iso, league, datePickerGameCount);
  }
  if (dateHintEl) dateHintEl.textContent = formatDateHint(league);

  renderDateQuickPicks();
  updateDateNavButtons();
}

function setActiveScheduleDate(iso, { syncPicker = true } = {}) {
  if (!iso) return;
  activeScheduleDate = iso;
  if (!syncPicker) return;
  if (datePickerInput && sportSelect.value !== "overview") {
    datePickerInput.value = iso;
  }
  if (dateDisplayBtn && sportSelect.value !== "overview") {
    dateDisplayBtn.textContent = formatDateBarLabel(iso, sportSelect.value, datePickerGameCount);
  }
  renderDateQuickPicks();
  updateDateNavButtons();
}

function shiftIsoDate(iso, days) {
  const match = String(iso || "").match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) return iso;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const shifted = new Date(Date.UTC(year, month - 1, day + days, 12, 0, 0));
  return shifted.toISOString().slice(0, 10);
}

function renderDateQuickPicks() {
  if (!dateQuickEl || sportSelect.value === "overview") {
    if (dateQuickEl) dateQuickEl.innerHTML = "";
    return;
  }

  const sport = sportSelect.value;
  const current = getSelectedDate();
  const quickDates = [leagueDateIso(sport, -1), leagueDateIso(sport, 0), leagueDateIso(sport, 1)];

  dateQuickEl.innerHTML = quickDates
    .map(
      (iso) =>
        `<button type="button" class="date-chip${iso === current ? " active" : ""}" data-date="${iso}">${formatDateChipLabel(iso, sport)}</button>`
    )
    .join("");

  dateQuickEl.querySelectorAll(".date-chip").forEach((button) => {
    button.addEventListener("click", () => onDateSelected(button.dataset.date));
  });
}
function availableDatesForLeague(league) {
  const meta = leagueMeta(league);
  const dates = new Set(meta?.availableDates || []);

  dates.add(defaultDateForSport(league));
  for (let offset = -3; offset <= 7; offset += 1) {
    dates.add(leagueDateIso(league, offset));
  }

  if (!IS_STATIC_HOST) {
    for (let offset = -14; offset <= 14; offset += 1) {
      dates.add(leagueDateIso(league, offset));
    }
  }

  return [...dates].filter(Boolean).sort();
}

function updateDateNavButtons() {
  if (!datePrevBtn || !dateNextBtn) return;
  const disabled = sportSelect.value === "overview";
  datePrevBtn.disabled = disabled;
  dateNextBtn.disabled = disabled;
}

function syncDatePicker(league, preferredDate = null) {
  if (sportSelect.value === "overview" || league === "overview") {
    datePickerGameCount = null;
    syncDatePickerUI();
    return;
  }

  const dates = availableDatesForLeague(league);
  const preferred = preferredDate || activeScheduleDate || getSelectedDate() || defaultDateForSport(league);
  const options = [...new Set([...dates, preferred])].sort();
  const currentValue = options.includes(preferred)
    ? preferred
    : options[options.length - 1] || defaultDateForSport(league);

  activeScheduleDate = currentValue;
  datePickerGameCount = null;
  syncDatePickerUI();
}

function openDatePicker() {
  if (!datePickerInput || datePickerInput.disabled) return;
  if (typeof datePickerInput.showPicker === "function") {
    try {
      datePickerInput.showPicker();
      return;
    } catch {
      // fall through to click()
    }
  }
  datePickerInput.click();
}

function onDateSelected(iso) {
  if (!iso || sportSelect.value === "overview") return;
  if (iso === activeScheduleDate) return;
  datePickerGameCount = null;
  setActiveScheduleDate(iso, { syncPicker: true });
  loadDashboard(true);
}

function onDateNav(delta) {
  const current = getSelectedDate();
  if (!current) return;
  onDateSelected(shiftIsoDate(current, delta));
}

function formatDateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function showBanner(message, { autoHideMs = 0, type = "info", details = "" } = {}) {
  if (!bannerEl) return;
  if (bannerSummaryEl) {
    bannerSummaryEl.textContent = message;
  } else {
    bannerEl.textContent = message;
  }
  if (bannerDetailsEl && bannerDetailTextEl) {
    if (details) {
      bannerDetailTextEl.textContent = details;
      bannerDetailsEl.classList.remove("hidden");
    } else {
      bannerDetailTextEl.textContent = "";
      bannerDetailsEl.classList.add("hidden");
    }
  }
  bannerEl.classList.remove("hidden", "banner-error", "banner-success");
  if (type === "error") bannerEl.classList.add("banner-error");
  if (type === "success") bannerEl.classList.add("banner-success");
  if (bannerTimer) {
    clearTimeout(bannerTimer);
    bannerTimer = null;
  }
  if (autoHideMs > 0) {
    bannerTimer = setTimeout(() => hideBanner(), autoHideMs);
  }
}

function hideBanner() {
  if (bannerTimer) {
    clearTimeout(bannerTimer);
    bannerTimer = null;
  }
  bannerEl?.classList.add("hidden");
}

function lineupLabelForSport(sport) {
  return sport === "mlb" ? "Batting lineup" : "Key players";
}

function getFilterValues() {
  return {
    minConfidence: Number(confidenceFilter?.value || confidenceFilterMobile?.value || 0),
    query: (teamSearch?.value || teamSearchMobile?.value || "").trim().toLowerCase(),
  };
}

function syncFilterControlsFromDesktop() {
  if (confidenceFilterMobile && confidenceFilter) {
    confidenceFilterMobile.value = confidenceFilter.value;
  }
  if (teamSearchMobile && teamSearch) {
    teamSearchMobile.value = teamSearch.value;
  }
  if (liveScoresToggleMobile && liveScoresToggle) {
    liveScoresToggleMobile.checked = liveScoresToggle.checked;
  }
  if (autoRefreshMobile && autoRefresh) {
    autoRefreshMobile.checked = autoRefresh.checked;
  }
}

function applyFilterControlsFromMobile() {
  if (confidenceFilter && confidenceFilterMobile) {
    confidenceFilter.value = confidenceFilterMobile.value;
  }
  if (teamSearch && teamSearchMobile) {
    teamSearch.value = teamSearchMobile.value;
  }
  if (liveScoresToggle && liveScoresToggleMobile) {
    liveScoresToggle.checked = liveScoresToggleMobile.checked;
    resetLiveScorePolling();
  }
  if (autoRefresh && autoRefreshMobile) {
    autoRefresh.checked = autoRefreshMobile.checked;
    resetAutoRefresh();
  }
}

function openFilterSheet() {
  syncFilterControlsFromDesktop();
  filterSheetEl?.classList.remove("hidden");
  filterSheetBackdropEl?.classList.remove("hidden");
  filterCloseBtn?.focus();
}

function closeFilterSheet() {
  filterSheetEl?.classList.add("hidden");
  filterSheetBackdropEl?.classList.add("hidden");
}

function showLoadingSkeletons() {
  statsEl?.setAttribute("aria-busy", "true");
  gamesEl?.setAttribute("aria-busy", "true");
  statsEl?.classList.add("skeleton-strip");
  statsEl?.querySelectorAll(".stat-value").forEach((el) => {
    el.dataset.prevText = el.textContent;
    el.textContent = "—";
    el.classList.add("skeleton-item");
  });
  if (gamesEl) {
    gamesEl.innerHTML = Array.from({ length: 5 })
      .map(() => `<div class="skeleton-item"></div>`)
      .join("");
    gamesEl.classList.add("skeleton-list");
  }
  topPicksEl?.classList.add("hidden");
}

function summarizePickTiers(games) {
  let strong = 0;
  let lean = 0;
  let live = 0;
  for (const game of games || []) {
    const confidence = game.prediction?.confidence;
    if (confidence == null) continue;
    if (confidence >= 68) strong += 1;
    else if (confidence >= 57) lean += 1;
    if (game.isLive) live += 1;
  }
  return { strong, lean, live, picks: (games || []).filter((g) => g.prediction?.outcomeLabel).length };
}

function teamAbbrev(name) {
  if (!name) return "—";
  const words = String(name).trim().split(/\s+/);
  if (words.length === 1) return words[0].slice(0, 3).toUpperCase();
  return words.map((w) => w[0]).join("").slice(0, 4).toUpperCase();
}

function formatGameTimeShort(startDate) {
  if (!startDate) return "TBD";
  const date = new Date(startDate);
  if (Number.isNaN(date.getTime())) return "TBD";
  return date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

function renderScoreboardTeams(game) {
  const away = teamAbbrev(game.awayTeam);
  const home = teamAbbrev(game.homeTeam);
  const hasScore = game.homeScore != null && game.awayScore != null;
  if (hasScore && (game.isLive || game.isFinal)) {
    const scoreClass = game.isLive ? "score-live" : "";
    return `<p class="scoreboard-teams ${scoreClass}" data-live-score="${game.eventId}">${away} <span class="tabular-nums">${game.awayScore}</span> @ ${home} <span class="tabular-nums">${game.homeScore}</span></p>`;
  }
  return `<p class="scoreboard-teams">${away} @ ${home}</p>`;
}

function renderScoreboardProbBar(prediction) {
  const homePct = Number(prediction?.homeWinPct ?? 50);
  const awayPct = Number(prediction?.awayWinPct ?? 50);
  const total = homePct + awayPct || 100;
  const homeWidth = Math.round((homePct / total) * 100);
  const awayWidth = 100 - homeWidth;
  return `<div class="scoreboard-prob-bar" aria-hidden="true"><span class="scoreboard-prob-away" style="width:${awayWidth}%"></span><span class="scoreboard-prob-home" style="width:${homeWidth}%"></span></div>`;
}

function renderCoverageChips(game) {
  const coverage = game.prediction?.features?.dataCoverage;
  if (!coverage) return "";
  const chips = Object.entries(COVERAGE_LABELS)
    .map(([key, label]) => {
      const on = Boolean(coverage[key]);
      return `<span class="coverage-chip${on ? " on" : ""}">${label}</span>`;
    })
    .join("");
  return `<div class="coverage-chips">${chips}</div>`;
}

function patchLiveScoreDom(games) {
  if (!gamesEl || activeView !== "predictions") return false;
  let patched = false;
  for (const game of games || []) {
    if (game.homeScore == null) continue;
    const away = teamAbbrev(game.awayTeam);
    const home = teamAbbrev(game.homeTeam);
    const nodes = gamesEl.querySelectorAll(`[data-live-score="${game.eventId}"]`);
    if (!nodes.length) continue;

    nodes.forEach((node) => {
      if (node.classList.contains("scoreboard-teams")) {
        const next = `${away} ${game.awayScore} @ ${home} ${game.homeScore}`;
        const current = node.textContent?.replace(/\s+/g, " ").trim();
        if (current !== next) {
          node.innerHTML = `${away} <span class="tabular-nums">${game.awayScore}</span> @ ${home} <span class="tabular-nums">${game.homeScore}</span>`;
          node.classList.add("score-live", "score-updated");
          patched = true;
        }
      } else if (node.classList.contains("game-score-line")) {
        const prefix = game.isFinal ? "Final" : game.isLive ? "Live" : "Score";
        const next = `${prefix}: ${away} ${game.awayScore} – ${home} ${game.homeScore}`;
        const current = node.textContent?.replace(/\s+/g, " ").trim();
        if (current !== next) {
          node.innerHTML = `${prefix}: ${away} <span class="tabular-nums">${game.awayScore}</span> – ${home} <span class="tabular-nums">${game.homeScore}</span>`;
          node.classList.add("score-updated");
          patched = true;
        }
      }
    });

    const timeEl = gamesEl.querySelector(`[data-live-time="${game.eventId}"]`);
    if (timeEl) {
      const nextTime = game.isLive ? "LIVE" : gameStatusLabel(game);
      if (timeEl.textContent !== nextTime) {
        timeEl.textContent = nextTime;
        timeEl.classList.toggle("scoreboard-live-badge", game.isLive);
        timeEl.classList.toggle("scoreboard-voided-badge", isGameVoided(game));
        timeEl.classList.toggle("scoreboard-delayed-badge", Boolean(game.isDelayed && !game.isLive));
        patched = true;
      }
    }
    const featuredTime = document.querySelector(`[data-featured-time="${game.eventId}"]`);
    if (featuredTime && game.isLive) {
      featuredTime.textContent = "LIVE";
      featuredTime.classList.add("featured-pick-live");
    }
  }
  if (patched && statLive) {
    const tiers = summarizePickTiers(games);
    statLive.textContent = String(tiers.live);
  }
  return patched;
}

function filterGames(games) {
  const { minConfidence, query } = getFilterValues();
  return (games || []).filter((game) => {
    const confidence = game.prediction?.confidence;
    if (confidence != null && confidence < minConfidence) return false;
    if (!query) return true;
    const haystack = `${game.homeTeam || ""} ${game.awayTeam || ""} ${game.matchup || ""}`.toLowerCase();
    return haystack.includes(query);
  });
}

function gameDisplayRank(game) {
  return game?.displayRank ?? game?.predictionRank ?? "?";
}

function prepareGamesForDisplay(games) {
  const refreshed = (games || []).map(refreshGameStatusFlags);
  const filtered = filterGames(refreshed);
  const playable = filtered.filter((game) => !isUnplayableGame(game));
  return [...playable]
    .sort((left, right) => (right.prediction?.confidence ?? 0) - (left.prediction?.confidence ?? 0))
    .map((game, index) => ({ ...game, displayRank: index + 1 }));
}

function countRemovedGames(games) {
  const refreshed = (games || []).map(refreshGameStatusFlags);
  return filterGames(refreshed).filter((game) => isUnplayableGame(game)).length;
}

function renderStats(payload, visibleGames, { topPick, gameCount } = {}) {
  const games = visibleGames || [];
  const tiers = summarizePickTiers(games);
  const displayCount = gameCount ?? games.length ?? payload.gameCount ?? 0;
  statGames.textContent = displayCount;
  if (statPicks) statPicks.textContent = tiers.picks;
  if (statStrong) statStrong.textContent = tiers.strong;
  if (statLean) statLean.textContent = tiers.lean;
  if (statLive) statLive.textContent = tiers.live;

  const sport = sportSelect.value;
  const displayDate = getSelectedDate() || payload.scheduleDate || "—";
  const tz = payload.scheduleTimezone || leagueMeta(sport)?.scheduleTimezone;
  if (statDate) statDate.textContent = displayDate !== "—" && tz ? `${displayDate} (${tz})` : displayDate;
  if (statTopPick) statTopPick.textContent = topPick || payload.topPick || games[0]?.prediction?.outcomeLabel || "—";
  if (statLeague) {
    statLeague.textContent =
      sport === "overview" ? "All sports" : payload.leagueLabel || SPORT_LABELS[sport] || "—";
  }
  if (statUpdated) statUpdated.textContent = formatDateTime(payload.fetchedAt);
  dashboardTitle.textContent =
    sport === "overview" ? "All Sports" : `${payload.leagueLabel || SPORT_LABELS[sport] || "Sports"}`;

  if (statFreshness) {
    if (lastLiveScoreAt) {
      const seconds = Math.round((Date.now() - lastLiveScoreAt) / 1000);
      const scoreNote = liveScoresToggle?.checked ? ` · scores ${seconds}s ago` : "";
      statFreshness.textContent = `${describeScheduleSource(payload)}${scoreNote}`;
    } else {
      statFreshness.textContent = describeScheduleSource(payload);
    }
  }

  statsEl?.classList.remove("skeleton-strip");
  statsEl?.setAttribute("aria-busy", "false");
  statsEl?.querySelectorAll(".stat-value.skeleton-item").forEach((el) => {
    el.classList.remove("skeleton-item");
    if (el.dataset.prevText) delete el.dataset.prevText;
  });
}

function renderOverview() {
  if (!overviewData) {
    gamesEl.innerHTML = `<div class="empty-state">Overview not loaded yet.</div>`;
    return;
  }

  const totalGames = (overviewData.leagues || []).reduce((sum, league) => sum + (league.gameCount || 0), 0);
  renderStats(
    {
      gameCount: totalGames,
      topPick: overviewData.topPicksOverall?.[0]?.pick,
      fetchedAt: overviewData.builtAt,
      leagueLabel: "All sports",
    },
    [],
    { gameCount: totalGames }
  );

  const leagueCards = (overviewData.leagues || [])
    .map(
      (league) => `
      <article class="overview-card">
        <h3>${league.label}</h3>
        <p>${league.gameCount} games · ${league.scheduleDate || "—"}</p>
        <p class="overview-pick">${league.topPick || "No pick yet"}${league.topConfidence ? ` · ${league.topConfidence}%` : ""}</p>
        <button type="button" class="share-btn overview-jump" data-league="${league.id}">View league</button>
      </article>
    `
    )
    .join("");

  const topOverall = (overviewData.topPicksOverall || [])
    .map(
      (pick) => `
      <article class="top-pick-card">
        <span class="rank-badge">${pick.leagueLabel}</span>
        <strong>${pick.pick || pick.matchup}</strong>
        <span class="top-pick-meta">${pick.confidence}% · ${pick.confidenceLabel || ""}</span>
      </article>
    `
    )
    .join("");

  topPicksEl.classList.remove("hidden");
  topPicksEl.innerHTML = `<h2 class="top-picks-title">Best picks across all sports</h2><div class="top-picks-grid">${topOverall || "<p>No picks yet.</p>"}</div>`;

  gamesEl.innerHTML = `
    <section class="overview-section">
      <h2>Leagues at a glance</h2>
      <div class="overview-grid">${leagueCards}</div>
    </section>
  `;

  gamesEl.querySelectorAll(".overview-jump").forEach((button) => {
    button.addEventListener("click", () => {
      sportSelect.value = button.dataset.league;
      onSportChange();
    });
  });
}

function rankTierClass(rank) {
  const value = Number(rank);
  if (value === 1) return "rank-tier-gold";
  if (value === 2) return "rank-tier-silver";
  if (value === 3) return "rank-tier-bronze";
  return "";
}

function formatConfidencePct(value) {
  const pct = Math.max(0, Math.min(100, Number(value) || 0));
  return Math.round(pct * 10) / 10;
}

function formatConfidenceDisplay(value) {
  const pct = formatConfidencePct(value);
  return Number.isInteger(pct) ? String(pct) : pct.toFixed(1);
}

function renderConfidenceRing(confidence) {
  const pct = formatConfidencePct(confidence);
  const display = formatConfidenceDisplay(confidence);
  return `<div class="confidence-ring" style="--pct: ${pct}" title="${display}% confidence" aria-label="${display} percent confidence"><span class="confidence-ring-inner"><span class="confidence-ring-value">${display}</span><span class="confidence-ring-unit">%</span></span></div>`;
}

function renderTopPicks(games) {
  const top = (games || [])[0];
  if (!top?.prediction?.outcomeLabel) {
    topPicksEl.classList.add("hidden");
    topPicksEl.innerHTML = "";
    return;
  }

  const rank = gameDisplayRank(top);
  const prediction = top.prediction;
  const labelClass = prediction.confidenceLabel === "Strong pick" ? "label-strong" : prediction.confidenceLabel === "Lean" ? "label-lean" : "";
  const timeLabel = top.isLive
    ? `<span class="featured-pick-time featured-pick-live" data-featured-time="${top.eventId}">LIVE</span>`
    : `<span class="featured-pick-time" data-featured-time="${top.eventId}">${formatGameTimeShort(top.startDate)}</span>`;

  topPicksEl.classList.remove("hidden");
  topPicksEl.innerHTML = `
    <article class="featured-pick-card">
      <div class="featured-pick-rank ${rankTierClass(rank)}" aria-label="Rank ${rank}">#${rank}</div>
      <div class="featured-pick-matchup">
        <p class="featured-pick-teams">${escapeHtml(top.matchup || `${top.awayTeam} @ ${top.homeTeam}`)}</p>
        <p class="featured-pick-pick">Pick: <strong>${escapeHtml(prediction.outcomeLabel)}</strong> · ${formatConfidenceDisplay(prediction.confidence)}%</p>
        ${renderScoreboardProbBar(prediction)}
      </div>
      <div class="featured-pick-side">
        ${timeLabel}
        <div class="featured-pick-confidence ${labelClass}">${prediction.confidenceLabel || ""}</div>
      </div>
    </article>
  `;
}

function renderLineupColumn(teamName, lineup) {
  if (!lineup || !lineup.batters?.length) {
    return `<div class="lineup-column"><h5>${teamName || "Team"}</h5><p class="lineup-note">${lineup?.note || "Not available yet."}</p></div>`;
  }
  const rows = lineup.batters
    .map((batter) =>
      batter.order
        ? `<li><span class="lineup-order">${batter.order}</span> ${batter.name} <span class="lineup-pos">${batter.position || ""}</span></li>`
        : `<li>${batter.name} <span class="lineup-pos">${batter.statLine || batter.position || ""}</span></li>`
    )
    .join("");
  return `<div class="lineup-column"><h5>${teamName || "Team"}</h5><p class="lineup-note">${lineup.note || ""}</p><ol class="lineup-list">${rows}</ol></div>`;
}

function renderLineups(game, lineupLabel) {
  const hasLineup = game.homeLineup?.batters?.length || game.awayLineup?.batters?.length;
  if (!hasLineup && sportSelect.value !== "mlb") return "";
  return `<section class="detail-panel"><h4>${lineupLabel}</h4><div class="lineup-grid">${renderLineupColumn(game.awayTeam, game.awayLineup)}${renderLineupColumn(game.homeTeam, game.homeLineup)}</div></section>`;
}

function renderInjuryColumn(teamName, injuries) {
  if (!injuries?.length) {
    return `<div class="injury-column"><h5>${teamName || "Team"}</h5><p class="lineup-note">No major injuries listed.</p></div>`;
  }
  const rows = injuries
    .map((injury) => `<li><strong>${injury.player}</strong> — ${injury.status}${injury.detail ? `: ${injury.detail}` : ""}</li>`)
    .join("");
  return `<div class="injury-column"><h5>${teamName || "Team"}</h5><ul class="injury-list">${rows}</ul></div>`;
}

function renderMajorInjuries(game) {
  const hasInjuries = game.homeMajorInjuries?.length || game.awayMajorInjuries?.length;
  if (!hasInjuries && sportSelect.value !== "mlb") return "";
  return `<section class="detail-panel"><h4>Major injuries</h4><div class="lineup-grid">${renderInjuryColumn(game.awayTeam, game.awayMajorInjuries)}${renderInjuryColumn(game.homeTeam, game.homeMajorInjuries)}</div></section>`;
}

function formatPct(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(1)}%`;
}

function pickProbabilitySummary(prediction) {
  if (!prediction?.teamProbabilities) return "";
  const side = prediction.predictedSide;
  const team = prediction.teamProbabilities[side];
  if (!team?.truePct) return "";
  return `Model ${formatPct(team.truePct)}`;
}

function renderTeamProbabilityTable(prediction, game) {
  const teams = prediction.teamProbabilities;
  if (!teams) return renderProbabilityCompare(prediction, game);

  const rows = [
    { key: "away", label: game.awayTeam || "Away", favored: prediction.predictedSide === "away" },
    { key: "home", label: game.homeTeam || "Home", favored: prediction.predictedSide === "home" },
  ];
  if (teams.draw) {
    rows.push({ key: "draw", label: "Draw", favored: prediction.predictedSide === "draw" });
  }

  const trueP = prediction.probabilities?.true || {};
  const componentRows = (trueP.components || [])
    .map((item) => `<li><strong>${item.source}</strong> (${item.weightPct}% weight): ${item.homePct}% home — ${item.detail}</li>`)
    .join("");

  const body = rows
    .map((row) => {
      const stats = teams[row.key] || {};
      return `
        <tr class="${row.favored ? "prob-pick-row" : ""}">
          <td data-label="Team"><strong>${row.label}</strong>${row.favored ? ' <span class="rank-badge small">Pick</span>' : ""}</td>
          <td data-label="Model %" class="true-pct">${formatPct(stats.truePct ?? stats.blendedPct)}</td>
        </tr>
      `;
    })
    .join("");

  return `
    <section class="probability-compare">
      <h4>Win probability by team</h4>
      <table class="lines-table prob-pct-table prob-sheet-table">
        <thead>
          <tr>
            <th>Team</th>
            <th>Model %</th>
          </tr>
        </thead>
        <tbody>${body}</tbody>
      </table>
      ${componentRows ? `<ul class="prob-components">${componentRows}</ul>` : ""}
      <p class="lineup-note">Model % uses records, form, injuries, lineups, and advanced stats.</p>
    </section>
  `;
}

function renderProbabilityCompare(prediction, game) {
  const probs = prediction.probabilities;
  if (!probs) return "";

  const trueP = probs.true || {};
  const pick = probs.pick || probs.blended || {};

  const componentRows = (trueP.components || [])
    .map((item) => `<li><strong>${item.source}</strong> (${item.weightPct}% weight): ${item.homePct}% home — ${item.detail}</li>`)
    .join("");

  return `
    <section class="probability-compare">
      <h4>Model win probability</h4>
      <div class="prob-grid">
        <article class="prob-card true-card">
          <p class="prob-card-label">Model estimate</p>
          <p class="prob-card-values">${game.homeTeam}: <strong>${trueP.homePct ?? pick.homePct ?? prediction.homeWinPct ?? "—"}%</strong> · ${game.awayTeam}: <strong>${trueP.awayPct ?? pick.awayPct ?? prediction.awayWinPct ?? "—"}%</strong>${trueP.drawPct != null ? ` · Draw: <strong>${trueP.drawPct}%</strong>` : ""}</p>
          <p class="lineup-note">Records, form, injuries, advanced stats, and ESPN predictor.</p>
          ${componentRows ? `<ul class="prob-components">${componentRows}</ul>` : ""}
        </article>
      </div>
    </section>
  `;
}

function renderPrediction(game) {
  const prediction = game.prediction;
  if (!prediction) return "";

  const labelClass = prediction.confidenceLabel === "Strong pick" ? "label-strong" : prediction.confidenceLabel === "Lean" ? "label-lean" : "label-coin";

  const factors = (prediction.factors || [])
    .map((factor) => {
      const edgeClass = factor.edge === "home" ? "edge-home" : factor.edge === "away" ? "edge-away" : "";
      return `<span class="factor-chip ${edgeClass}" title="${factor.detail}">${factor.label}</span>`;
    })
    .join("");

  const reasons = (prediction.reasons || [])
    .map((reason) => `<li><strong>${reason.title}:</strong> ${reason.detail}${reason.source ? ` <em>(${reason.source})</em>` : ""}</li>`)
    .join("");

  const sources = (prediction.dataSources || []).map((source) => `<span class="source-chip">${source}</span>`).join("");

  const probabilityCompare = renderTeamProbabilityTable(prediction, game);
  const pickResult = renderModelPickBlock(game);

  return `
    <section class="prediction-panel prediction-panel-compact">
      <div class="details-status-row">
        ${renderGameScoreLine(game)}
        <div class="details-status-meta">
          <span class="confidence-label ${labelClass}">${prediction.confidenceLabel || ""}</span>
          ${renderPickStatusBadge(game)}
        </div>
      </div>
      <p class="prediction-detail-pick">${escapeHtml(prediction.outcomeLabel)} <span class="prediction-detail-pct">${formatConfidenceDisplay(prediction.confidence)}% model confidence</span></p>
      ${pickResult}
      ${probabilityCompare}
      <div class="why-panel">
        <h4>Why ${prediction.predictedSide === "draw" ? "draw" : prediction.predictedWinner}?</h4>
        <p class="why-summary">${prediction.whyTheyWin || "Analysis pending."}</p>
        ${reasons ? `<ul class="why-list">${reasons}</ul>` : ""}
        ${sources ? `<div class="source-list">${sources}</div>` : ""}
      </div>
      ${factors ? `<div class="factor-list">${factors}</div>` : ""}
    </section>
  `;
}

function renderGames(games) {
  if (sportSelect.value === "overview") {
    renderOverview();
    return;
  }

  const sport = sportSelect.value;
  const leagueLabel = SPORT_LABELS[sport] || "games";
  const refreshedGames = (games || []).map(refreshGameStatusFlags);
  const visible = prepareGamesForDisplay(refreshedGames);
  const removedCount = countRemovedGames(refreshedGames);

  if (!visible.length) {
    const scheduleOnly = lastPayload?.liveScheduleOnly;
    const buildError = lastPayload?.error;
    const displayDate = getSelectedDate();
    const tz = lastPayload?.scheduleTimezone || leagueMeta(sport)?.scheduleTimezone;
    const sourceHint = describeScheduleSource(lastPayload);
    const removedNote = removedCount
      ? ` ${removedCount} game${removedCount === 1 ? " was" : "s were"} washed out or postponed and removed from picks.`
      : "";
    gamesEl.innerHTML = `<div class="empty-state">No ${leagueLabel} picks available for ${displayDate}${tz ? ` (${tz})` : ""}.${removedNote} Source: ${sourceHint}.${buildError ? ` Build error: ${buildError}` : ""}${scheduleOnly ? " Live ESPN schedule loaded — predictions appear once GitHub Actions builds that date." : ""}</div>`;
    renderTopPicks([]);
    renderStats(lastPayload || {}, [], { gameCount: refreshedGames.length });
    renderModelDayResult(refreshedGames);
    if (removedCount) {
      showBanner(`${removedCount} game${removedCount === 1 ? "" : "s"} washed out or postponed — removed from picks.`, {
        autoHideMs: 10000,
        type: "success",
      });
    }
    return;
  }

  const topPickLabel = visible[0]?.prediction?.outcomeLabel;
  renderTopPicks(visible);
  renderStats(lastPayload || {}, visible, { topPick: topPickLabel, gameCount: refreshedGames.length });
  renderModelDayResult(refreshedGames);
  if (removedCount) {
    showBanner(`${removedCount} game${removedCount === 1 ? "" : "s"} washed out or postponed — removed from picks.`, {
      autoHideMs: 8000,
      type: "success",
    });
  }

  const lineupLabel = lineupLabelForSport(sport);
  const hash = window.location.hash;
  const hashGameId = hash.startsWith("#game-") ? hash.slice("#game-".length) : null;

  gamesEl.classList.remove("skeleton-list");
  gamesEl.setAttribute("aria-busy", "false");

  gamesEl.innerHTML = visible
    .map((game) => {
      const prediction = game.prediction;
      const labelClass = prediction?.confidenceLabel === "Strong pick" ? "label-strong" : prediction?.confidenceLabel === "Lean" ? "label-lean" : "label-coin";
      const records = game.awayRecord || game.homeRecord ? `${game.awayTeam} ${game.awayRecord || "—"} · ${game.homeTeam} ${game.homeRecord || "—"}` : null;
      const pitchers = sport === "mlb" && (game.awayPitcher?.name || game.homePitcher?.name) ? `SP: ${game.awayPitcher?.name || "TBD"} vs ${game.homePitcher?.name || "TBD"}` : null;
      const metaParts = [formatDateTime(game.startDate), game.venueName || "Venue TBD"];
      if (game.broadcast) metaParts.push(game.broadcast);
      if (records) metaParts.push(records);
      if (pitchers) metaParts.push(pitchers);

      const shareUrl = `${window.location.origin}${window.location.pathname}#game-${game.eventId}`;
      const logged = isBetLoggedForGame(game.eventId);
      const modelTracked = isModelPickTracked(game.eventId);
      const detailsOpen = hashGameId && String(game.eventId) === hashGameId;
      const timeLabel = game.isLive ? "LIVE" : gameStatusLabel(game);
      const pickShort = pickTeamAbbrev(game, prediction);
      const confLabel = prediction?.confidenceLabel === "Strong pick" ? "Strong" : prediction?.confidenceLabel === "Lean" ? "Lean" : "";
      const rowStateClass = game.isLive ? " game-live" : game.isFinal ? " game-final" : isGameVoided(game) ? " game-voided" : game.isDelayed ? " game-delayed" : "";

      return `
        <article class="scoreboard-row game-card${rowStateClass}" id="game-${game.eventId}" data-game-id="${game.eventId}">
          <details class="game-details"${detailsOpen ? " open" : ""}>
            <summary class="scoreboard-summary game-summary-bar">
              <span class="scoreboard-rank" aria-label="Rank ${gameDisplayRank(game)}">#${gameDisplayRank(game)}</span>
              <div class="scoreboard-matchup-block">
                ${renderScoreboardTeams(game)}
                ${renderScoreboardProbBar(prediction)}
                <div class="scoreboard-summary-chips">${renderCoverageChips(game)}</div>
              </div>
              <div class="scoreboard-pick-block">
                <span class="scoreboard-pick-label">Pick</span>
                <span class="scoreboard-pick-value" title="${escapeAttr(prediction?.outcomeLabel || "")}">${escapeHtml(pickShort)} <span class="tabular-nums">${formatConfidenceDisplay(prediction?.confidence || 0)}%</span></span>
                <span class="scoreboard-pick-conf">
                  ${confLabel ? `<span class="confidence-label ${labelClass}">${confLabel}</span>` : ""}
                  <span class="scoreboard-time${game.isLive ? " scoreboard-live-badge" : isGameVoided(game) ? " scoreboard-voided-badge" : game.isDelayed ? " scoreboard-delayed-badge" : ""}" data-live-time="${game.eventId}">${escapeHtml(timeLabel)}</span>
                </span>
              </div>
            </summary>
            <div class="scoreboard-details-body game-details-body">
              ${renderPrediction(game)}
              <div class="game-head game-head-compact">
                <p class="meta">${metaParts.join(" · ")}</p>
                <div class="game-actions">
                  <span class="status-pill ${game.isLive ? "live" : ""}${isGameVoided(game) ? " voided" : ""}${game.isDelayed ? " delayed" : ""}">${escapeHtml(game.gameStatusText || "Scheduled")}</span>
                  <button type="button" class="track-btn model-track-btn${modelTracked ? " tracked" : ""}" data-model-track-id="${game.eventId}"${!prediction?.outcomeLabel && !prediction?.predictedWinner ? " disabled" : ""}>${modelTracked ? '<span class="track-btn-label track-btn-label-long">Already tracked</span><span class="track-btn-label track-btn-label-short">Tracked</span>' : '<span class="track-btn-label track-btn-label-long">Track model pick</span><span class="track-btn-label track-btn-label-short">Track pick</span>'}</button>
                  <button type="button" class="track-btn${logged ? " tracked" : ""}" data-event-id="${game.eventId}">${logged ? '<span class="track-btn-label track-btn-label-long">Bet logged</span><span class="track-btn-label track-btn-label-short">Logged</span>' : "Log bet"}</button>
                  <button type="button" class="share-btn" data-share-url="${shareUrl}" data-share-title="${prediction?.outcomeLabel || game.matchup}">Share</button>
                </div>
              </div>
              ${renderLineups(game, lineupLabel)}
              ${renderMajorInjuries(game)}
            </div>
          </details>
        </article>
      `;
    })
    .join("");

  gamesEl.querySelectorAll(".track-btn[data-event-id]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const game = (lastPayload?.games || []).find((item) => String(item.eventId) === String(button.dataset.eventId));
      if (game) openBetFormFromGame(game);
    });
  });

  gamesEl.querySelectorAll(".model-track-btn").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (button.disabled) return;
      const game = (lastPayload?.games || []).find(
        (item) => String(item.eventId) === String(button.dataset.modelTrackId)
      );
      if (game) addModelTrackerFromGame(game);
    });
  });

  gamesEl.querySelectorAll(".share-btn").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      const url = button.dataset.shareUrl;
      const title = button.dataset.shareTitle || "Pick";
      try {
        if (navigator.share) {
          await navigator.share({ title, url });
        } else {
          await navigator.clipboard.writeText(url);
          showBanner("Link copied to clipboard.", { autoHideMs: 3000, type: "success" });
        }
      } catch {
        /* user cancelled */
      }
    });
  });

  if (hash.startsWith("#game-")) {
    const target = document.querySelector(hash);
    const details = target?.querySelector(".game-details");
    if (details) details.open = true;
    target?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function staticDataUrl(path, force) {
  const url = new URL(path, window.location.href);
  if (force) {
    url.searchParams.set("t", String(Date.now()));
  } else if (String(path).includes("/data/") && String(path).endsWith(".json")) {
    url.searchParams.set("v", String(Math.floor(Date.now() / 300000)));
  }
  return url.toString();
}

async function fetchManifest({ force = false } = {}) {
  try {
    const response = await fetch(staticDataUrl("data/manifest.json", force));
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

async function fetchOverview({ force = false } = {}) {
  try {
    const response = await fetch(staticDataUrl("data/overview.json", force));
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

function leagueMeta(league) {
  return (manifestData?.leagues || []).find((item) => item.id === league);
}

function parseEspnScoreboard(data, league) {
  const games = [];
  for (const event of data.events || []) {
    const competition = event.competitions?.[0];
    if (!competition) continue;

    let away;
    let home;
    for (const competitor of competition.competitors || []) {
      if (competitor.homeAway === "home") home = competitor;
      if (competitor.homeAway === "away") away = competitor;
    }

    const awayTeam = away?.team?.displayName;
    const homeTeam = home?.team?.displayName;
    const status = event.status?.type || {};
    const statusFlags = normalizeEspnStatus(status, {
      startDate: competition.date || event.date,
      attendance: competition.attendance,
      notes: competition.notes,
      homeScore: home?.score,
      awayScore: away?.score,
    });

    games.push({
      league,
      leagueLabel: SPORT_LABELS[league] || league,
      eventId: event.id,
      startDate: competition.date || event.date,
      awayTeam,
      homeTeam,
      matchup: awayTeam && homeTeam ? `${awayTeam} @ ${homeTeam}` : event.name,
      gameStatusText: statusFlags.gameStatusText,
      gameStatusDetail: statusFlags.gameStatusDetail,
      statusType: statusFlags.statusType,
      venueName: competition.venue?.fullName,
      awayScore: away?.score,
      homeScore: home?.score,
      isLive: statusFlags.isLive,
      isFinal: statusFlags.isFinal,
      isScheduled: statusFlags.isScheduled,
      isPostponed: statusFlags.isPostponed,
      isCanceled: statusFlags.isCanceled,
      isSuspended: statusFlags.isSuspended,
      isDelayed: statusFlags.isDelayed,
      isVoided: statusFlags.isVoided,
      isWashedOut: statusFlags.isWashedOut,
      attendance: statusFlags.attendance,
      lines: [],
      source: "espn-live",
    });
  }

  games.sort((a, b) => String(a.startDate || "").localeCompare(String(b.startDate || "")));
  return games;
}

async function fetchEspnSchedule(league, dateValue) {
  const espnPath = leagueMeta(league)?.espnPath || ESPN_PATHS[league];
  if (!espnPath || !dateValue) return null;

  const dates = dateValue.replace(/-/g, "");
  const url = `https://site.api.espn.com/apis/site/v2/sports/${espnPath}/scoreboard?dates=${dates}`;
  const response = await fetch(url);
  if (!response.ok) return null;

  const data = await response.json();
  const games = filterGamesForScheduleDate(parseEspnScoreboard(data, league), dateValue, league);
  return {
    league,
    leagueLabel: SPORT_LABELS[league] || league,
    scheduleDate: dateValue,
    games,
    gameCount: games.length,
    topPick: null,
    fetchedAt: new Date().toISOString(),
    source: "espn-live",
    liveScheduleOnly: true,
  };
}

async function fetchStaticPayloadForDate(league, dateValue, { force = false } = {}) {
  const meta = leagueMeta(league);
  const candidatePaths = [
    meta?.dateFiles?.[dateValue],
    `data/${league}_${dateValue}.json`,
  ].filter(Boolean);

  let emptyDatePayload = null;

  for (const filePath of candidatePaths) {
    const response = await fetch(staticDataUrl(String(filePath).replace(/^\//, ""), force));
    if (!response.ok) continue;
    const payload = await response.json();
    const gameCount = payload.gameCount ?? payload.games?.length ?? 0;
    const matchesDate = (payload.scheduleDate || dateValue) === dateValue;
    if (matchesDate && gameCount > 0) {
      return payload;
    }
    if (matchesDate && !emptyDatePayload) {
      emptyDatePayload = payload;
    }
  }

  const livePayload = await fetchEspnSchedule(league, dateValue);
  if (livePayload && (livePayload.gameCount ?? livePayload.games?.length ?? 0) > 0) {
    livePayload._requestedDate = dateValue;
    livePayload._liveFallback = true;
    return livePayload;
  }

  if (emptyDatePayload) {
    emptyDatePayload.scheduleDate = dateValue;
    if (livePayload) {
      emptyDatePayload._liveFallback = true;
      emptyDatePayload.games = livePayload.games || [];
      emptyDatePayload.gameCount = livePayload.gameCount ?? emptyDatePayload.games.length;
      emptyDatePayload.fetchedAt = livePayload.fetchedAt;
      emptyDatePayload.liveScheduleOnly = true;
    }
    return emptyDatePayload;
  }

  let fallbackPayload = null;
  const fallbackResponse = await fetch(staticDataUrl(`data/${league}.json`, force));
  if (fallbackResponse.ok) {
    fallbackPayload = await fallbackResponse.json();
    if (fallbackPayload.scheduleDate === dateValue) {
      return fallbackPayload;
    }
  }

  if (livePayload) {
    livePayload._requestedDate = dateValue;
    livePayload._liveFallback = true;
    return livePayload;
  }

  if (fallbackPayload) {
    fallbackPayload._requestedDate = dateValue;
    fallbackPayload._dateFallback = true;
    return fallbackPayload;
  }

  throw new Error(`Could not load ${league} data for ${dateValue}.`);
}

async function fetchStaticPayload(league, { force = false, dateValue = null } = {}) {
  if (dateValue) {
    return fetchStaticPayloadForDate(league, dateValue, { force });
  }
  const response = await fetch(staticDataUrl(`data/${league}.json`, force));
  if (!response.ok) throw new Error(`Could not load ${league} data (${response.status}).`);
  return response.json();
}

function updateFreshnessNote() {
  if (!IS_STATIC_HOST) {
    freshnessNote.classList.add("hidden");
    return;
  }
  freshnessNote.classList.remove("hidden");
  const note = manifestData?.snapshotNote || "Predictions refresh on GitHub Actions. Live scores refresh in your browser.";
  freshnessNote.textContent = note;
}

async function fetchLiveScoresFromEspn(league, dateValue) {
  const meta = leagueMeta(league);
  const espnPath = meta?.espnPath || ESPN_PATHS[league];
  if (!espnPath || !dateValue) return null;

  const dates = dateValue.replace(/-/g, "");
  const url = `https://site.api.espn.com/apis/site/v2/sports/${espnPath}/scoreboard?dates=${dates}`;
  const response = await fetch(url);
  if (!response.ok) return null;

  const data = await response.json();
  const scores = {};
  for (const event of data.events || []) {
    const competition = event.competitions?.[0];
    if (!competition) continue;
    let away;
    let home;
    for (const competitor of competition.competitors || []) {
      if (competitor.homeAway === "home") home = competitor;
      if (competitor.homeAway === "away") away = competitor;
    }
    const status = event.status?.type || {};
    const statusFlags = normalizeEspnStatus(status, {
      startDate: competition.date || event.date,
      attendance: competition.attendance,
      notes: competition.notes,
      homeScore: home?.score,
      awayScore: away?.score,
    });
    scores[String(event.id)] = {
      awayScore: away?.score,
      homeScore: home?.score,
      isLive: statusFlags.isLive,
      isFinal: statusFlags.isFinal,
      isScheduled: statusFlags.isScheduled,
      isPostponed: statusFlags.isPostponed,
      isCanceled: statusFlags.isCanceled,
      isSuspended: statusFlags.isSuspended,
      isDelayed: statusFlags.isDelayed,
      isVoided: statusFlags.isVoided,
      isWashedOut: statusFlags.isWashedOut,
      attendance: statusFlags.attendance,
      gameStatusText: statusFlags.gameStatusText,
      gameStatusDetail: statusFlags.gameStatusDetail,
      statusType: statusFlags.statusType,
    };
  }
  return scores;
}

function mergeLiveScores(games, liveScores) {
  if (!liveScores) return games;
  return (games || []).map((game) => {
    const live = liveScores[String(game.eventId)];
    if (!live) return game;
    return { ...game, ...live };
  });
}

async function refreshLiveScores() {
  if (!liveScoresToggle?.checked || !lastPayload?.games?.length || sportSelect.value === "overview") return;

  const league = sportSelect.value;
  const dateValue = lastPayload.scheduleDate || getSelectedDate();
  try {
    const liveScores = await fetchLiveScoresFromEspn(league, dateValue);
    if (!liveScores) return;
    lastLiveScoreAt = Date.now();
    const mergedGames = mergeLiveScores(lastPayload.games, liveScores);
    lastPayload = { ...lastPayload, games: mergedGames };
    saveMyBets(autoSettleMyBets(loadMyBets()));
    saveModelTracker(autoSettleModelBets(loadModelTracker()));
    if (activeView === "predictions") {
      const patched = patchLiveScoreDom(mergedGames);
      if (!patched) renderGames(mergedGames);
      else renderStats(lastPayload, prepareGamesForDisplay(mergedGames));
    } else if (activeView === "my-bets") {
      renderMyBetsView();
    } else if (activeView === "model-tracker") {
      renderModelTrackerView();
    }
  } catch {
    /* ESPN CORS or network blocked — keep snapshot scores */
  }
}

function resetLiveScorePolling() {
  if (liveScoresTimer) clearInterval(liveScoresTimer);
  liveScoresTimer = null;
  if (liveScoresToggle?.checked) {
    const intervalMs = (manifestData?.liveScoreRefreshSeconds || 90) * 1000;
    liveScoresTimer = setInterval(refreshLiveScores, intervalMs);
    refreshLiveScores();
  }
}

async function fetchAccuracy({ force = false } = {}) {
  try {
    const response = await fetch(staticDataUrl("data/accuracy.json", force));
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

async function fetchCalibration({ force = false } = {}) {
  try {
    const response = await fetch(staticDataUrl("data/calibration.json", force));
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

async function fetchDashboardPayload(params, { force = false } = {}) {
  const league = params.get("league") || sportSelect.value;

  if (IS_STATIC_HOST) {
    manifestData = manifestData || (await fetchManifest({ force }));
    if (manifestData?.leagues?.length) {
      syncDatePicker(league, activeScheduleDate || getSelectedDate());
    }
    accuracyData = await fetchAccuracy({ force });
    calibrationData = (await fetchCalibration({ force })) ?? calibrationData;

    if (league === "overview") {
      overviewData = await fetchOverview({ force });
      return overviewData || { games: [], gameCount: 0, leagueLabel: "All sports" };
    }

    const dateValue = params.get("date") || getSelectedDate();
    const payload = await fetchStaticPayload(league, { force, dateValue });

    if (payload.scheduleDate && payload.scheduleDate !== dateValue && !payload._liveFallback) {
      payload._requestedDate = dateValue;
      payload._dateFallback = true;
    }

    return payload;
  }

  if (window.location.protocol === "file:") {
    throw new Error("Run start-dashboard.bat and open http://127.0.0.1:8765/");
  }

  const API_PATHS = ["/api/games", "/api/odds"];
  for (const path of API_PATHS) {
    if (force) params.set("force", "1");
    const response = await fetch(`${path}?${params.toString()}`);
    const contentType = response.headers.get("content-type") || "";
    const body = await response.text();
    if (!contentType.includes("application/json")) continue;
    const payload = JSON.parse(body);
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
    if (league !== "overview") {
      syncDatePicker(league, activeScheduleDate || getSelectedDate());
    }
    return payload;
  }
  throw new Error("Could not reach the dashboard API.");
}

async function loadDashboard(force = false) {
  if (loadingDashboard && !force) return;
  loadingDashboard = true;
  refreshBtn.disabled = true;
  updateRefreshButtonLabel();
  if (sportSelect.value !== "overview" && activeView === "predictions") {
    showLoadingSkeletons();
  } else if (sportSelect.value !== "overview") {
    gamesEl.innerHTML = `<div class="empty-state loading-state">Loading ${SPORT_LABELS[sportSelect.value] || sportSelect.value}…</div>`;
  }

  const requestedDate = getSelectedDate();
  setActiveScheduleDate(requestedDate, { syncPicker: true });
  const params = new URLSearchParams({
    league: sportSelect.value,
    date: requestedDate,
  });

  try {
    const payload = stripBettingLinesFromPayload(await fetchDashboardPayload(params, { force }));
    if (sportSelect.value !== "overview") {
      accuracyData = (await fetchAccuracy({ force })) ?? accuracyData;
      calibrationData = (await fetchCalibration({ force })) ?? calibrationData;
    }

    if (sportSelect.value === "overview") {
      lastPayload = payload;
      updateFreshnessNote();
      hideBanner();
      renderOverview();
    } else {
      const displayDate = requestedDate;
      const league = sportSelect.value;
      let normalizedPayload = payload;
      if (!(payload._dateFallback && payload.scheduleDate !== displayDate)) {
        normalizedPayload = applyScheduleDateToPayload(payload, displayDate, league);
      } else {
        normalizedPayload = { ...payload, games: [], gameCount: 0 };
      }
      lastPayload = normalizedPayload;
      updateFreshnessNote();

      setActiveScheduleDate(displayDate, { syncPicker: true });
      syncDatePicker(league, displayDate);
      const tz = normalizedPayload.scheduleTimezone || leagueMeta(league)?.scheduleTimezone;
      statDate.textContent = `${displayDate}${tz ? ` (${tz})` : ""}`;

      const games = normalizedPayload.games || [];
      if (activeView === "predictions") {
        renderGames(games);
      } else if (activeView === "accuracy") {
        renderAccuracyView();
      }
      updateDateDisplayCount(games.length);

      if (games.length === 0 && normalizedPayload.error) {
        showBanner(`Data build error: ${normalizedPayload.error}. Try another date or re-run GitHub Actions.`);
      } else if (IS_STATIC_HOST) {
        const summaryParts = [`${displayDate}${tz ? ` (${formatTimezoneLabel(tz)})` : ""}`];
        if (normalizedPayload._liveFallback) {
          summaryParts.push("live ESPN schedule");
        } else if (normalizedPayload._dateFallback) {
          summaryParts.push("no snapshot for this date");
        } else {
          summaryParts.push(`${games.length} game${games.length === 1 ? "" : "s"}`);
        }
        if (games.length === 0) {
          summaryParts.push("no games on this date");
        }

        const detailParts = [
          "Fixtures come from GitHub snapshots when built, otherwise live ESPN in your browser.",
          `Predictions refresh every 30 minutes on GitHub Actions; live scores every ${manifestData?.liveScoreRefreshSeconds || 90}s.`,
        ];
        if (tz && US_SCHEDULE_SPORTS.has(league)) {
          detailParts.push(`US sports use ${formatTimezoneLabel(tz)} calendar dates (not your local day).`);
        }
        if (normalizedPayload._liveFallback) {
          detailParts.push(
            `Showing live ESPN for ${displayDate} — model picks appear after the next snapshot build.`
          );
        } else if (normalizedPayload._dateFallback) {
          detailParts.push(`No snapshot file for ${normalizedPayload._requestedDate || displayDate}.`);
        }
        if (games.length === 0) {
          detailParts.push(`No games on ${displayDate}. Try another date or Refresh.`);
        }

        const needsBanner =
          normalizedPayload._liveFallback ||
          normalizedPayload._dateFallback ||
          games.length === 0;

        if (needsBanner) {
          showBanner(summaryParts.join(" · "), { details: detailParts.join(" ") });
        } else {
          hideBanner();
        }
      } else if (games.length === 0) {
        showBanner(`No games for ${displayDate}${tz ? ` (${tz})` : ""}. Pick another date or try Refresh.`);
      } else {
        hideBanner();
      }
      resetLiveScorePolling();
      renderAccuracyPanel();
    }
  } catch (error) {
    if (sportSelect.value === "overview" && overviewData) {
      showBanner(`${error.message} Showing last overview.`);
      renderOverview();
    } else if (lastPayload?.games) {
      showBanner(`${error.message} Showing last loaded data.`);
      renderGames(lastPayload.games);
    } else {
      showBanner(error.message || "Failed to load dashboard.");
      gamesEl.innerHTML = `<div class="empty-state">Could not load games.${IS_STATIC_HOST ? " Wait for GitHub Actions to finish." : " Run start-dashboard.bat."}</div>`;
    }
  } finally {
    loadingDashboard = false;
    refreshBtn.disabled = false;
    updateRefreshButtonLabel();
    if (activeView === "my-bets") {
      saveMyBets(autoSettleMyBets(loadMyBets()));
      renderMyBetsView();
    } else if (activeView === "model-tracker") {
      saveModelTracker(autoSettleModelBets(loadModelTracker()));
      renderModelTrackerView();
    }
  }
}

function resetAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = null;
  if (autoRefresh.checked) refreshTimer = setInterval(() => loadDashboard(false), 120000);
}

function onSportChange() {
  if (sportSelect.value === "overview") {
    syncDatePicker("overview");
  } else {
    activeScheduleDate = defaultDateForSport(sportSelect.value);
    syncDatePicker(sportSelect.value, activeScheduleDate);
  }
  if (activeView === "accuracy") renderAccuracyView();
  loadDashboard(true);
}

function configureStaticMode() {
  /* Date select is always enabled for individual leagues. */
}

function registerServiceWorker() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }
}

async function initDashboard() {
  configureStaticMode();
  registerServiceWorker();
  syncOddsFormatSelect();
  bindMyBetsImportActions();

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
      flushModelTrackerEditsFromDom();
    }
  });

  if (IS_STATIC_HOST) {
    manifestData = await fetchManifest({ force: false });
  }

  sportSelect.addEventListener("change", onSportChange);
  viewTabsEl?.querySelectorAll(".view-tab").forEach((button) => {
    button.addEventListener("click", () => {
      const view = button.dataset.view;
      if (view && view !== activeView) switchView(view);
    });
  });
  window.addEventListener("hashchange", () => {
    const hashView = viewFromHash();
    if (hashView && hashView !== activeView) {
      switchView(hashView, { skipHashUpdate: true });
    }
  });
  oddsFormatSelect?.addEventListener("change", () => {
    const previousFormat = getOddsFormat();
    const nextFormat = oddsFormatSelect.value === "american" ? "american" : "decimal";
    if (previousFormat === nextFormat) return;

    if (activeView === "my-bets") {
      const form = myBetsViewEl?.querySelector("#my-bet-form");
      if (form) {
        betFormDraft = collectBetFormDraftFromForm(form, betFormDraft);
      }
    }

    setOddsFormat(nextFormat);
    convertBetFormDraftOddsFormat(nextFormat, previousFormat);
    if (activeView === "my-bets") {
      renderMyBetsView();
    } else if (activeView === "model-tracker") {
      renderModelTrackerView();
    } else if (lastPayload) {
      renderGames(lastPayload.games || []);
    }
  });
  confidenceFilter?.addEventListener("change", () => renderGames(lastPayload?.games || []));
  teamSearch?.addEventListener("input", () => renderGames(lastPayload?.games || []));
  filterOpenBtn?.addEventListener("click", openFilterSheet);
  filterCloseBtn?.addEventListener("click", closeFilterSheet);
  filterSheetBackdropEl?.addEventListener("click", closeFilterSheet);
  filterApplyBtn?.addEventListener("click", () => {
    applyFilterControlsFromMobile();
    closeFilterSheet();
    renderGames(lastPayload?.games || []);
  });
  refreshBtn.addEventListener("click", () => {
    if (activeView === "my-bets") {
      loadDashboard(true);
      return;
    }
    loadDashboard(true);
  });
  dateDisplayBtn?.addEventListener("click", openDatePicker);
  datePickerInput?.addEventListener("change", () => {
    const iso = datePickerInput.value;
    if (!iso) return;
    onDateSelected(iso);
  });
  datePrevBtn?.addEventListener("click", () => onDateNav(-1));
  dateNextBtn?.addEventListener("click", () => onDateNav(1));
  autoRefresh?.addEventListener("change", resetAutoRefresh);
  liveScoresToggle?.addEventListener("change", resetLiveScorePolling);

  syncDatePicker(sportSelect.value, defaultDateForSport(sportSelect.value));

  const initialView = viewFromHash();
  if (initialView && initialView !== activeView) {
    switchView(initialView, { skipHashUpdate: true });
  } else {
    updateRefreshButtonLabel();
  }

  await loadDashboard(true);
  resetAutoRefresh();
  resetLiveScorePolling();
}

initDashboard().catch((error) => {
  showBanner(error.message || "Failed to start dashboard.");
  gamesEl.innerHTML = `<div class="empty-state">Could not load dashboard. Tap Refresh or clear site data in your browser.</div>`;
});
