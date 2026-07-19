/**
 * Sports Predictions Dashboard - Apple Quality Edition
 * Modern, clean, performant JavaScript for the dashboard
 */

import { fetchDashboardData, staticDataUrl, stripBettingLinesForDisplay } from '../mlb_data.js';
import { applyPredictions, predictTotal, predictSpread, MIN_PUBLISHABLE_CONFIDENCE } from '../mlb_predictions.js';
import { parse_scoreboard, get_league, list_league_ids } from '../sports_config.js';
import { loadMyBets, saveMyBets, autoSettleMyBets, summarizeMyBets } from '../bet_tracker.js';
import { loadModelTracker, saveModelTracker, autoSettleModelBets, summarizeModelTracker, normalizeModelTrackerEntry } from '../model_tracker.js';
import { loadBankrollSettings, saveBankrollSettings } from '../bet_tracker.js';
import { loadModelTracker as loadModelTrackerSDK, exportModelTrackerToClipboard } from '../model_tracker.js';
import { getOddsFormat, setOddsFormat, parseOddsInput, formatOddsDisplay, formatDecimalOdds, americanToDecimal, decimalToAmerican, parseAmericanOddsInput, parseDecimalOddsInput, resolveOddsToDecimal, calcBetProfitDecimal, formatOddsInputPlaceholder, oddsFieldLabel, formatBetDateShort, parseBetDateInput, betDateInputFromIso, defaultBetFormDraft, collectBetFormDraftFromForm, syncBetFormDraftFromDom, resolveBetFormGameLink, openBetFormFromGame, normalizeBet, createBetId, loadModelPickCache, cacheModelPick, findGameForLeg, gradeLeg, autoSettleLeg, settleBetFromLegs, autoSettleBet, autoSettleMyBets, betLabel, renderBetLegsDetail, betGameLabel } from '../mlb_predictions.js';
import { formatDateTime, formatDateChipLabel, formatDateBarLabel, formatScheduleDayLabel, getSelectedDate, defaultDateForSport, shiftIsoDate, availableDatesForLeague, syncDatePicker, setActiveScheduleDate, syncDatePickerUI, openDatePicker, onDateSelected, onDateNav, leagueDateIso, defaultDateForSport, formatDateHint, formatDateChipLabel } from '../schedule_dates.js';
import { formatWinPctFromRecord, parse_record, win_pct_from_record, format_win_pct } from '../shared_utils.js';
import { formatMoney, formatPlainMoney, clamp, safe_get, escapeHtml, escapeAttr, formatPct } from '../shared_utils.js';
import { formatConfidencePct, formatConfidenceDisplay, renderConfidenceRing, confidenceLabel, confidence_label, renderConfidenceRing as renderConfidenceRingUI } from '../shared_utils.js';
import { formatPct, formatOddsDisplay, formatOddsInputPlaceholder, formatDecimalOdds, formatConfidenceDisplay as formatConfidenceDisplayUI, americanToDecimal as americanToDecimalUI, decimalToAmerican as decimalToAmericanUI, parseAmericanOddsInput, parseDecimalOddsInput, resolveOddsToDecimal, calcBetProfitDecimal, formatOddsInputPlaceholder, oddsFieldLabel, formatBetDateShort, parseBetDateInput, betDateInputFromIso, defaultBetFormDraft, collectBetFormDraftFromForm, syncBetFormDraftFromDom, resolveBetFormGameLink, openBetFormFromGame, normalizeBet, createBetId, loadModelPickCache, cacheModelPick, findGameForLeg, gradeLeg, autoSettleLeg, settleBetFromLegs, autoSettleBet, autoSettleMyBets, betLabel, renderBetLegsDetail, betGameLabel, extractPredictionFeatures, apply_predictions, predict_game, predict_total, predict_spread } from '../mlb_predictions.js';
import { enrich_games_with_providers, ensure_espn_odds_on_games } from '../data_providers/enrich.js';
import { merge_sbr_odds_into_games, ensure_espn_odds_on_games, merge_live_schedule, merge_live_scores, refresh_live_scores, reset_live_score_polling, reset_auto_refresh } from '../mlb_data.js';
import { loadMyBets, saveMyBets, autoSettleMyBets, summarizeMyBets, openBetFormFromGame, isBetLoggedForGame, loadBankrollSettings, saveBankrollSettings, summarizeMyBets, loadBankrollSettings, saveBankrollSettings } from '../bet_tracker.js';
import { loadModelTracker, saveModelTracker, autoSettleModelBets, summarizeModelTracker, normalizeModelTrackerEntry, addModelTrackerFromGame, deleteModelTrackerEntry, updateModelTrackerDate, updateModelTrackerStake, updateModelTrackerUserOdds, settleModelTrackerManual, exportModelTrackerToClipboard, loadModelTracker, addModelTrackerFromGame, deleteModelTrackerEntry, updateModelTrackerDate, updateModelTrackerStake, updateModelTrackerUserOdds, settleModelTrackerManual, exportModelTrackerToClipboard, loadModelTracker, autoSettleModelBets, summarizeModelTracker, normalizeModelTrackerEntry } from '../model_tracker.js';
import { loadBankrollSettings, saveBankrollSettings } from '../bet_tracker.js';
import { loadModelTracker as loadModelTrackerSDK, exportModelTrackerToClipboard } from '../model_tracker.js';

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
const themeToggleBtn = document.getElementById("theme-toggle");
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

const MIN_PUBLISHABLE_CONFIDENCE = 57;

function isPublishablePrediction(prediction) {
  if (!prediction?.predictedWinner && !prediction?.outcomeLabel) return false;
  if (prediction.publishable === false) return false;
  const confidence = Number(prediction?.confidence);
  if (Number.isNaN(confidence)) return Boolean(prediction?.predictedWinner);
  return confidence >= MIN_PUBLISHABLE_CONFIDENCE;
}

function coverageFromGame(game) {
  const fromPrediction = game?.prediction?.features?.dataCoverage;
  if (fromPrediction) return fromPrediction;
  const enrichment = game?.enrichment || {};
  const hasLineup = Boolean(game?.homeLineup?.batters?.length || game?.awayLineup?.batters?.length);
  const hasOdds = Boolean(game?.viewTypes?.some((viewType) => String(viewType).includes("MoneyLine")));
  return {
    lineup: hasLineup,
    injuries: Boolean(game?.homeMajorInjuries?.length || game?.awayMajorInjuries?.length),
    espnPredictor: enrichment.espnPredictorHome != null && enrichment.espnPredictorAway != null,
    advancedStats: Boolean(enrichment.homeAdvanced?.powerRating != null || enrichment.awayAdvanced?.powerRating != null),
    restData: enrichment.restDays?.home != null && enrichment.restDays?.away != null,
    scheduleFlags: Boolean(enrichment.homeScheduleFlags || enrichment.awayScheduleFlags),
    mlbPitching: Boolean(enrichment.mlbPitching),
    leagueMetrics: Boolean(enrichment.leagueMetrics && Object.keys(enrichment.leagueMetrics).length > 1),
    impliedOdds: hasOdds,
  };
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
  const coverage = coverageFromGame(game);
  if (!coverage) return "";
  const chips = Object.entries(COVERAGE_LABELS)
    .map(([key, label]) => {
      const on = Boolean(coverage[key]);
      return `<span class="coverage-chip${on ? " on" : ""}">${label}</span>`;
    })
    .join("");
  return `<div class="coverage-chips">${chips}</div>`;
}

function visiblePickIds(games) {
  return prepareGamesForDisplay(games).map((game) => String(game.eventId));
}

function pickVisibilityChanged(beforeGames, afterGames) {
  const before = visiblePickIds(beforeGames);
  const after = visiblePickIds(afterGames);
  if (before.length !== after.length) return true;
  return before.some((id, index) => id !== after[index]);
}

function hasMaterialStatusChange(beforeGames, afterGames) {
  const beforeMap = new Map((beforeGames || []).map((game) => [String(game.eventId), refreshGameStatusFlags(game)]));
  for (const game of afterGames || []) {
    const after = refreshGameStatusFlags(game);
    const before = beforeMap.get(String(game.eventId));
    if (!before) continue;
    if (before.isLive !== after.isLive) return true;
    if (before.isFinal !== after.isFinal) return true;
    if (before.isWashedOut !== after.isWashedOut) return true;
    if (before.isVoided !== after.isVoided) return true;
    if (before.isPostponed !== after.isPostponed) return true;
    if (before.isDelayed !== after.isDelayed) return true;
    if ((before.homeScore == null) !== (after.homeScore == null)) return true;
    if ((before.awayScore == null) !== (after.awayScore == null)) return true;
  }
  return false;
}

function patchLiveScoreDom(games) {
  if (!gamesEl || activeView !== "predictions") return false;
  const visible = prepareGamesForDisplay(games);
  let patched = false;
  for (const game of visible) {
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
    if (featuredTime) {
      const nextFeaturedTime = game.isLive ? "LIVE" : formatGameTimeShort(game.startDate);
      if (featuredTime.textContent !== nextFeaturedTime) {
        featuredTime.textContent = nextFeaturedTime;
        featuredTime.classList.toggle("featured-pick-live", game.isLive);
        patched = true;
      }
    }
  }
  if (patched && statLive) {
    const tiers = summarizePickTiers(visible);
    statLive.textContent = String(tiers.live);
  }
  return patched;
}

function filterGames(games) {
  const { minConfidence, query } = getFilterValues();
  return (games || []).filter((game) => {
    const confidence = game.prediction?.confidence;
    if (
      minConfidence > 0 &&
      confidence != null &&
      isPublishablePrediction(game?.prediction) &&
      confidence < minConfidence
    ) {
      return false;
    }
    if (!query) return true;
    const haystack = `${game.homeTeam || ""} ${game.awayTeam || ""} ${game.matchup || ""}`.toLowerCase();
    return haystack.includes(query);
  });
}

function gameDisplayRank(game) {
  if (game?.displayRank != null) return game.displayRank;
  return game?.predictionRank ?? "—";
}

function confidenceLabelFromConfidence(confidence) {
  const value = Number(confidence);
  if (Number.isNaN(value)) return "";
  if (value >= 68) return "Strong pick";
  if (value >= 57) return "Lean";
  return "Coin flip";
}

function recordToPrediction(record) {
  if (!record) return null;
  const predictedWinner = record.predictedWinner || record.predicted;
  const outcomeLabel = record.outcomeLabel || (predictedWinner ? `${predictedWinner} to win` : null);
  if (!outcomeLabel && !predictedWinner) return null;
  const confidence = record.confidence != null ? Number(record.confidence) : null;
  return {
    predictedWinner,
    predictedSide: record.predictedSide,
    outcomeLabel,
    confidence,
    confidenceLabel: confidenceLabelFromConfidence(confidence),
    publishable: record.publishable,
    features: record.features,
  };
}

function hydrateGamePredictions(games, { league, scheduleDate } = {}) {
  const records = new Map();
  for (const [eventId, pick] of Object.entries(accuracyData?.picksByEventId || {})) {
    records.set(String(eventId), pick);
  }
  for (const [eventId, pick] of Object.entries(predictionsLogData?.predictions || {})) {
    if (!records.has(String(eventId))) records.set(String(eventId), pick);
  }

  return (games || []).map((game) => {
    if (isPublishablePrediction(game?.prediction)) return game;
    const record = records.get(String(game.eventId));
    if (!record) return game;
    if (league && record.league && record.league !== league) return game;
    if (scheduleDate && record.scheduleDate && record.scheduleDate !== scheduleDate) return game;
    const prediction = recordToPrediction(record);
    if (!prediction || !isPublishablePrediction(prediction)) return game;
    return { ...game, prediction };
  });
}

function countGamesWithPredictions(games) {
  return (games || []).filter((game) => isPublishablePrediction(game?.prediction)).length;
}

function applyScheduleDateToPayload(payload, scheduleDate, league) {
  if (!payload?.games?.length) return payload;
  const games = apply_predictions(payload.games, { league, scheduleDate });
  const pickCount = countGamesWithPredictions(games);
  return {
    ...payload,
    games: apply_predictions(payload.games, { league, scheduleDate }),
    gameCount: payload.games.length,
    topPick: payload.topPick,
    liveScheduleOnly: pickCount > 0 ? false : payload.liveScheduleOnly,
    _liveFallback: pickCount > 0 ? false : payload._liveFallback,
  };
}

function gameHasPrediction(game) {
  return isPublishablePrediction(game?.prediction);
}

function prepareGamesForDisplay(games) {
  const refreshed = (games || []).map(refreshGameStatusFlags);
  const filtered = filterGames(refreshed);
  const playable = filtered.filter((game) => !isUnplayableGame(game));
  const publishable = playable
    .filter((game) => isPublishablePrediction(game?.prediction))
    .sort((left, right) => (right.prediction?.confidence ?? 0) - (left.prediction?.confidence ?? 0));
  const unpublished = playable
    .filter((game) => !isPublishablePrediction(game?.prediction))
    .sort((left, right) => String(left.startDate || "").localeCompare(String(right.startDate || "")));

  let rank = 1;
  const rankedPublishable = publishable.map((game) => ({ ...game, displayRank: rank++ }));
  const rankedUnpublished = unpublished.map((game) => ({ ...game, displayRank: null }));
  return [...rankedPublishable, ...rankedUnpublished];
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
      (league) => `\n      <article class="overview-card">\n        <h3>${league.label}</h3>\n        <p>${league.gameCount} games · ${league.scheduleDate || "—"}</p>\n        <p class="overview-pick">${league.topPick || "No pick yet"}${league.topConfidence ? ` · ${league.topConfidence}%` : ""}</p>\n        <button type="button" class="share-btn overview-jump" data-league="${league.id}">View league</button>\n      </article>\n    `
    )
    .join("");

  const topOverall = (overviewData.topPicksOverall || [])
    .map(
      (pick) => `\n      <article class="top-pick-card">\n        <span class="rank-badge">${pick.leagueLabel}</span>\n        <strong>${pick.pick || pick.matchup}</strong>\n        <span class="top-pick-meta">${pick.confidence}% · ${pick.confidenceLabel || ""}</span>\n      </article>\n    `
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

async function fetchDashboardPayload(params, { force = false } = {}) {
  const league = params.get("league") || sportSelect.value;

  if (IS_STATIC_HOST) {
    manifestData = await fetchManifest({ force: force || !manifestData });
    if (manifestData?.leagues?.length) {
      syncDatePicker(league, activeScheduleDate || getSelectedDate());
    }
    accuracyData = (await fetchAccuracy({ force: force || !accuracyData })) ?? accuracyData;
    predictionsLogData = (await fetchPredictionsLog({ force: force || !predictionsLogData })) ?? predictionsLogData;
    calibrationData = (await fetchCalibration({ force })) ?? calibrationData;

    if (league === "overview") {
      overviewData = await fetchOverview({ force });
      return overviewData || { games: [], gameCount: 0, leagueLabel: "All sports" };
    }

    const dateValue = params.get("date") || getSelectedDate();
    let payload = await fetchStaticPayload(league, { force, dateValue });

    if (payload.scheduleDate && payload.scheduleDate !== dateValue && !payload._liveFallback) {
      payload._requestedDate = dateValue;
      payload._dateFallback = true;
    }

    payload = enrichPayloadWithPredictions(payload, league, dateValue);
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
      const normalizedPayload = applyScheduleDateToPayload(payload, displayDate, league);
      lastPayload = normalizedPayload;
      updateFreshnessNote();

      setActiveScheduleDate(displayDate, { syncPicker: true });
      syncDatePicker(league, displayDate);
      const tz = normalizedPayload.scheduleTimezone || leagueMeta(league)?.scheduleTimezone;
      if (statDate) statDate.textContent = displayDate !== "—" && tz ? `${displayDate} (${tz})` : displayDate;

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
        const summaryParts = [`${displayDate}${tz ? ` (${tz})` : ""}`];
        const publishablePickCount = countGamesWithPredictions(games);
        if (normalizedPayload._liveFallback && publishablePickCount === 0) {
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
        if (normalizedPayload._liveFallback && publishablePickCount === 0) {
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
          (normalizedPayload._liveFallback && publishablePickCount === 0) ||
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
      gamesEl.innerHTML = `<div class="empty-state">Could not load dashboard.${IS_STATIC_HOST ? " Wait for GitHub Actions to finish." : " Run start-dashboard.bat."}</div>`;
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
  if (autoRefresh.checked) {
    refreshTimer = setInterval(() => loadDashboard(IS_STATIC_HOST), 120000);
  }
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
    manifestData = await fetchManifest({ force: true });
    predictionsLogData = await fetchPredictionsLog({ force: true });
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
  themeToggleBtn?.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
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
