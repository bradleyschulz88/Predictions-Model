const sportSelect = document.getElementById("sport-select");
const dateSelect = document.getElementById("date-select");
const datePrevBtn = document.getElementById("date-prev");
const dateNextBtn = document.getElementById("date-next");
const dateQuickEl = document.getElementById("date-quick");
const viewFilter = document.getElementById("view-filter");
const confidenceFilter = document.getElementById("confidence-filter");
const teamSearch = document.getElementById("team-search");
const refreshBtn = document.getElementById("refresh-btn");
const autoRefresh = document.getElementById("auto-refresh");
const gamesEl = document.getElementById("games");
const bannerEl = document.getElementById("banner");
const topPicksEl = document.getElementById("top-picks");
const dashboardTitle = document.getElementById("dashboard-title");

const statGames = document.getElementById("stat-games");
const statTopPick = document.getElementById("stat-top-pick");
const statAccuracy = document.getElementById("stat-accuracy");
const statLeague = document.getElementById("stat-league");
const statDate = document.getElementById("stat-date");
const statUpdated = document.getElementById("stat-updated");
const statFreshness = document.getElementById("stat-freshness");
const accuracyPanel = document.getElementById("accuracy-panel");
const freshnessNote = document.getElementById("freshness-note");
const liveScoresToggle = document.getElementById("live-scores");

const ESPN_PATHS = {
  mlb: "baseball/mlb",
  nfl: "football/nfl",
  nba: "basketball/nba",
  worldcup: "soccer/fifa.world",
  epl: "soccer/eng.1",
  afl: "australian-football/afl",
};

const GITHUB_PAGES_REPO = "bradleyschulz88/Predictions-Model";
const UPDATE_WORKFLOW_URL = `https://github.com/${GITHUB_PAGES_REPO}/actions/workflows/pages.yml`;
const IS_STATIC_HOST =
  window.location.hostname.endsWith("github.io") ||
  new URLSearchParams(window.location.search).has("static");

const LEAGUE_TIMEZONES = {
  mlb: "America/New_York",
  nfl: "America/New_York",
  nba: "America/New_York",
  worldcup: "America/New_York",
  epl: "Europe/London",
  afl: "Australia/Sydney",
};

const SPORT_LABELS = {
  mlb: "MLB Baseball",
  nfl: "NFL Football",
  nba: "NBA Basketball",
  worldcup: "FIFA World Cup",
  epl: "Premier League",
  afl: "AFL",
};

let refreshTimer = null;
let liveScoresTimer = null;
let loadingDashboard = false;
let lastPayload = null;
let accuracyData = null;
let manifestData = null;
let overviewData = null;
let lastLiveScoreAt = null;
let activeScheduleDate = null;
let dateOptionsCache = [];

function leagueTimezone(sport) {
  return LEAGUE_TIMEZONES[sport] || Intl.DateTimeFormat().resolvedOptions().timeZone;
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
  if (["mlb", "nfl", "nba"].includes(sport) && hour < 10) {
    return leagueDateIso(sport, -1);
  }
  return leagueDateIso(sport, 0);
}

function formatDateLabel(iso, sport = sportSelect.value) {
  const tz = leagueTimezone(sport);
  const today = leagueDateIso(sport, 0);
  const tomorrow = leagueDateIso(sport, 1);
  const yesterday = leagueDateIso(sport, -1);
  const date = new Date(`${iso}T12:00:00`);
  const formatted = date.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
  const tzShort = tz.split("/").pop()?.replace("_", " ") || tz;

  if (iso === today) return `Schedule today · ${formatted} (${tzShort})`;
  if (iso === tomorrow) return `Schedule tomorrow · ${formatted} (${tzShort})`;
  if (iso === yesterday) return `Schedule yesterday · ${formatted} (${tzShort})`;
  return `${formatted} (${tzShort})`;
}

function getSelectedDate() {
  return activeScheduleDate || dateSelect.value || defaultDateForSport(sportSelect.value);
}

function setActiveScheduleDate(iso, { syncSelect = true } = {}) {
  if (!iso) return;
  activeScheduleDate = iso;
  if (syncSelect) {
    ensureDateOption(iso);
    dateSelect.value = iso;
    renderDateQuickPicks();
    updateDateNavButtons();
  }
}

function shiftIsoDate(iso, days, sport = sportSelect.value) {
  const tz = leagueTimezone(sport);
  const base = new Date(`${iso}T12:00:00`);
  return leagueDateParts(tz, base, days).iso;
}

function ensureDateOption(iso) {
  if (!iso || !dateSelect) return;
  const exists = [...dateSelect.options].some((option) => option.value === iso);
  if (exists) return;
  const option = document.createElement("option");
  option.value = iso;
  option.textContent = formatDateLabel(iso, sportSelect.value);
  dateSelect.appendChild(option);
  dateOptionsCache = [...dateOptionsCache, iso].sort();
}

function availableDatesForLeague(league) {
  const meta = leagueMeta(league);
  if (meta?.availableDates?.length) {
    return [...new Set(meta.availableDates)].sort();
  }

  const dates = new Set([
    defaultDateForSport(league),
    leagueDateIso(league, 0),
    leagueDateIso(league, 1),
    leagueDateIso(league, -1),
  ]);

  if (["mlb", "nfl", "nba"].includes(league)) {
    dates.add(leagueDateIso(league, -2));
  }

  if (!IS_STATIC_HOST) {
    for (let offset = -3; offset <= 7; offset += 1) {
      dates.add(leagueDateIso(league, offset));
    }
  }

  return [...dates].sort();
}

function renderDateQuickPicks() {
  if (!dateQuickEl || sportSelect.value === "overview") {
    if (dateQuickEl) dateQuickEl.innerHTML = "";
    return;
  }

  const sport = sportSelect.value;
  const current = getSelectedDate();
  const quickDates = [leagueDateIso(sport, 0), leagueDateIso(sport, 1), leagueDateIso(sport, -1), defaultDateForSport(sport)]
    .filter((iso, index, list) => iso && list.indexOf(iso) === index)
    .sort();

  dateQuickEl.innerHTML = quickDates
    .map(
      (iso) =>
        `<button type="button" class="date-chip${iso === current ? " active" : ""}" data-date="${iso}">${formatDateLabel(iso, sport)}</button>`
    )
    .join("");

  dateQuickEl.querySelectorAll(".date-chip").forEach((button) => {
    button.addEventListener("click", () => onDateSelected(button.dataset.date));
  });
}

function updateDateNavButtons() {
  if (!datePrevBtn || !dateNextBtn) return;
  const disabled = sportSelect.value === "overview";
  datePrevBtn.disabled = disabled;
  dateNextBtn.disabled = disabled;
}

function populateDateSelect(league, preferredDate = null) {
  if (sportSelect.value === "overview") {
    dateSelect.disabled = true;
    dateSelect.innerHTML = `<option value="">All sports view</option>`;
    if (dateQuickEl) dateQuickEl.innerHTML = "";
    updateDateNavButtons();
    return;
  }

  const dates = availableDatesForLeague(league);
  const preferred = preferredDate || activeScheduleDate || getSelectedDate() || defaultDateForSport(league);
  const options = [...new Set([...dates, preferred])].sort();
  dateOptionsCache = options;

  const currentValue = options.includes(preferred) ? preferred : options[options.length - 1] || defaultDateForSport(league);
  const optionsKey = options.join("|");
  const existingKey = [...dateSelect.options].map((option) => option.value).join("|");

  if (optionsKey !== existingKey) {
  dateSelect.innerHTML = options
    .map((iso) => `<option value="${iso}">${formatDateLabel(iso, league)}</option>`)
    .join("");
  }

  dateSelect.disabled = false;
  setActiveScheduleDate(currentValue, { syncSelect: true });
}

function onDateSelected(iso) {
  if (!iso || sportSelect.value === "overview") return;
  setActiveScheduleDate(iso, { syncSelect: true });
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

function renderLineChips(line) {
  if (line == null) return "—";
  if (typeof line !== "object") return `<span class="line-chip">${line}</span>`;
  return Object.entries(line)
    .map(([key, value]) => `<span class="line-chip">${key}: ${value}</span>`)
    .join("");
}

function showBanner(message) {
  bannerEl.textContent = message;
  bannerEl.classList.remove("hidden");
}

function hideBanner() {
  bannerEl.classList.add("hidden");
}

function lineupLabelForSport(sport) {
  return sport === "mlb" ? "Batting lineup" : "Key players";
}

function filterGames(games) {
  const minConfidence = Number(confidenceFilter.value || 0);
  const query = (teamSearch.value || "").trim().toLowerCase();
  return (games || []).filter((game) => {
    const confidence = game.prediction?.confidence;
    if (confidence != null && confidence < minConfidence) return false;
    if (!query) return true;
    const haystack = `${game.homeTeam || ""} ${game.awayTeam || ""} ${game.matchup || ""}`.toLowerCase();
    return haystack.includes(query);
  });
}

function renderStats(payload, visibleCount) {
  statGames.textContent = visibleCount ?? payload.gameCount ?? 0;
  statDate.textContent = payload.scheduleDate || getSelectedDate() || "—";
  statTopPick.textContent = payload.topPick || "—";
  const sport = sportSelect.value;
  statLeague.textContent =
    sport === "overview" ? "All sports" : payload.leagueLabel || SPORT_LABELS[sport] || "—";
  statUpdated.textContent = formatDateTime(payload.fetchedAt);
  dashboardTitle.textContent =
    sport === "overview" ? "All Sports Predictions" : `${payload.leagueLabel || SPORT_LABELS[sport] || "Sports"} Predictions`;

  const acc = accuracyData?.summary?.last7Days;
  if (acc?.pct != null) {
    statAccuracy.textContent = `${acc.pct}% (${acc.correct}/${acc.total})`;
  } else {
    statAccuracy.textContent = "Building…";
  }

  if (lastLiveScoreAt) {
    const seconds = Math.round((Date.now() - lastLiveScoreAt) / 1000);
    statFreshness.textContent = liveScoresToggle?.checked ? `Scores ${seconds}s ago` : "Snapshot only";
  } else if (IS_STATIC_HOST) {
    statFreshness.textContent = "Hourly snapshot";
  } else {
    statFreshness.textContent = "Live server";
  }
}

function renderAccuracyPanel() {
  const results = accuracyData?.recentResults || [];
  if (!results.length) {
    accuracyPanel.classList.add("hidden");
    accuracyPanel.innerHTML = "";
    return;
  }

  const byLeague = accuracyData?.summary?.byLeague || {};
  const leagueRows = Object.entries(byLeague)
    .map(
      ([league, stats]) =>
        `<li><strong>${SPORT_LABELS[league] || league}</strong>: ${stats.pct != null ? `${stats.pct}% (${stats.correct}/${stats.total})` : "—"}</li>`
    )
    .join("");

  const recentRows = results
    .slice(0, 8)
    .map(
      (item) =>
        `<li class="${item.correct ? "acc-correct" : "acc-wrong"}"><strong>${item.matchup || item.league}</strong> — picked ${item.predicted}, result ${item.actual} ${item.correct ? "✓" : "✗"}</li>`
    )
    .join("");

  accuracyPanel.classList.remove("hidden");
  accuracyPanel.innerHTML = `
    <h2 class="accuracy-title">Prediction track record (7 days)</h2>
    <div class="accuracy-grid">
      <div>
        <h3>By league</h3>
        <ul class="accuracy-list">${leagueRows || "<li>No graded games yet.</li>"}</ul>
      </div>
      <div>
        <h3>Recent results</h3>
        <ul class="accuracy-list">${recentRows}</ul>
      </div>
    </div>
  `;
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
    totalGames
  );
  renderAccuracyPanel();

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
        ${pick.modelEdge ? `<span class="edge-chip">${pick.modelEdge}</span>` : ""}
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

function renderTopPicks(games) {
  const top = (games || []).slice(0, 3);
  if (!top.length) {
    topPicksEl.classList.add("hidden");
    topPicksEl.innerHTML = "";
    return;
  }
  topPicksEl.classList.remove("hidden");
  topPicksEl.innerHTML = `
    <h2 class="top-picks-title">Top picks today</h2>
    <div class="top-picks-grid">
      ${top
        .map(
          (game) => `
        <article class="top-pick-card">
          <span class="rank-badge">#${game.predictionRank}</span>
          <strong>${game.prediction?.outcomeLabel || game.matchup}</strong>
          <span class="top-pick-meta">${game.prediction?.confidence}% · ${game.prediction?.confidenceLabel || ""}</span>
          ${game.prediction?.modelEdge?.edgeLabel ? `<span class="edge-chip">${game.prediction.modelEdge.edgeLabel}</span>` : ""}
        </article>
      `
        )
        .join("")}
    </div>
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
  const implied = team.impliedPct != null ? ` · Implied ${formatPct(team.impliedPct)}` : "";
  return `True ${formatPct(team.truePct)}${implied}`;
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

  const body = rows
    .map((row) => {
      const stats = teams[row.key] || {};
      const edgeClass =
        stats.edgePct > 0 ? "edge-positive" : stats.edgePct < 0 ? "edge-negative" : "";
      return `
        <tr class="${row.favored ? "prob-pick-row" : ""}">
          <td><strong>${row.label}</strong>${row.favored ? ' <span class="rank-badge small">Pick</span>' : ""}</td>
          <td class="true-pct">${formatPct(stats.truePct)}</td>
          <td class="implied-pct">${stats.impliedPct != null ? formatPct(stats.impliedPct) : "—"}</td>
          <td class="blended-pct">${formatPct(stats.blendedPct)}</td>
          <td class="edge-pct ${edgeClass}">${stats.edgeLabel || "—"}</td>
        </tr>
      `;
    })
    .join("");

  const pickEdge = prediction.modelEdge;
  const edgeNote = pickEdge
    ? `<p class="prob-edge-line"><strong>Pick value edge:</strong> True ${formatPct(pickEdge.truePct)} vs implied ${formatPct(pickEdge.impliedPct)} <span class="edge-chip">${pickEdge.edgeLabel}</span></p>`
    : `<p class="lineup-note">Implied % appears when moneyline odds are available.</p>`;

  return `
    <section class="probability-compare">
      <h4>Probability by team (%)</h4>
      <table class="lines-table prob-pct-table">
        <thead>
          <tr>
            <th>Team</th>
            <th>True %</th>
            <th>Implied %</th>
            <th>Blended %</th>
            <th>Edge</th>
          </tr>
        </thead>
        <tbody>${body}</tbody>
      </table>
      ${edgeNote}
      <p class="lineup-note"><strong>True %</strong> = model data only (records, form, injuries, advanced stats). <strong>Implied %</strong> = devigged market odds. <strong>Blended %</strong> = final pick weighting.</p>
    </section>
  `;
}

function renderProbabilityCompare(prediction, game) {
  const probs = prediction.probabilities;
  if (!probs) return "";

  const trueP = probs.true || {};
  const implied = probs.implied || {};
  const blended = probs.blended || {};
  const consensus = implied.consensus || {};
  const edge = prediction.modelEdge;

  const bookRows = (implied.books || [])
    .map(
      (book) =>
        `<tr><td>${book.sportsbook}</td><td>${book.homePct}%</td><td>${book.awayPct}%</td><td>${book.drawPct != null ? `${book.drawPct}%` : "—"}</td><td>${book.vigPct}%</td></tr>`
    )
    .join("");

  const componentRows = (trueP.components || [])
    .map((item) => `<li><strong>${item.source}</strong> (${item.weightPct}% weight): ${item.homePct}% home — ${item.detail}</li>`)
    .join("");

  const edgeBlock = edge
    ? `<p class="prob-edge-line"><strong>Value edge:</strong> True ${edge.truePct}% vs implied ${edge.impliedPct}% <span class="edge-chip">${edge.edgeLabel}</span></p>`
    : "";

  return `
    <section class="probability-compare">
      <h4>True vs implied probability</h4>
      <div class="prob-grid">
        <article class="prob-card true-card">
          <p class="prob-card-label">True (all data)</p>
          <p class="prob-card-values">${game.homeTeam}: <strong>${trueP.homePct ?? "—"}%</strong> · ${game.awayTeam}: <strong>${trueP.awayPct ?? "—"}%</strong>${trueP.drawPct != null ? ` · Draw: <strong>${trueP.drawPct}%</strong>` : ""}</p>
          <p class="lineup-note">Records, form, injuries, advanced stats, ESPN predictor — no odds.</p>
          ${componentRows ? `<ul class="prob-components">${componentRows}</ul>` : ""}
        </article>
        <article class="prob-card implied-card">
          <p class="prob-card-label">Implied (market)</p>
          ${
            implied.available
              ? `<p class="prob-card-values">${game.homeTeam}: <strong>${consensus.homePct}%</strong> · ${game.awayTeam}: <strong>${consensus.awayPct}%</strong>${consensus.drawPct != null ? ` · Draw: <strong>${consensus.drawPct}%</strong>` : ""}</p>
                 <p class="lineup-note">Consensus from ${implied.booksUsed} book(s), ${consensus.avgVigPct}% avg vig removed.</p>`
              : `<p class="lineup-note">No moneyline odds published yet.</p>`
          }
        </article>
        <article class="prob-card blended-card">
          <p class="prob-card-label">Blended pick</p>
          <p class="prob-card-values">${game.homeTeam}: <strong>${blended.homePct ?? prediction.homeWinPct}%</strong> · ${game.awayTeam}: <strong>${blended.awayPct ?? prediction.awayWinPct}%</strong>${blended.drawPct != null ? ` · Draw: <strong>${blended.drawPct}%</strong>` : ""}</p>
          <p class="lineup-note">${blended.method || "Final pick probability."}</p>
        </article>
      </div>
      ${edgeBlock}
      ${
        bookRows
          ? `<table class="lines-table compact"><thead><tr><th>Book</th><th>Home</th><th>Away</th><th>Draw</th><th>Vig</th></tr></thead><tbody>${bookRows}</tbody></table>`
          : ""
      }
    </section>
  `;
}

function renderPrediction(game) {
  const prediction = game.prediction;
  if (!prediction) return "";

  const homeFavored = prediction.predictedSide === "home";
  const awayFavored = prediction.predictedSide === "away";
  const drawFavored = prediction.predictedSide === "draw";
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

  const edgeBlock = prediction.modelEdge
    ? `<div class="edge-panel"><strong>True vs implied:</strong> ${prediction.modelEdge.truePct ?? prediction.modelEdge.modelPct}% true · ${prediction.modelEdge.impliedPct ?? prediction.modelEdge.marketPct}% implied · <span class="edge-chip">${prediction.modelEdge.edgeLabel}</span></div>`
    : "";

  const probabilityCompare = renderTeamProbabilityTable(prediction, game);

  const totalBlock = prediction.totalPick
    ? `<div class="total-panel"><strong>Total pick:</strong> ${prediction.totalPick.pick} (${prediction.totalPick.confidence}% confidence)<br><span class="lineup-note">${prediction.totalPick.detail}</span>${
        prediction.totalPick.impliedOverPct != null
          ? `<br><span class="lineup-note">True: ${prediction.totalPick.trueOverPct}% over / ${prediction.totalPick.trueUnderPct}% under · Implied: ${prediction.totalPick.impliedOverPct}% / ${prediction.totalPick.impliedUnderPct}% (${prediction.totalPick.impliedBooksUsed} books)</span>`
          : ""
      }</div>`
    : "";

  const drawBlock = prediction.drawWinPct != null
    ? `<div class="probability-team ${drawFavored ? "favored" : ""}"><span class="probability-label">Draw</span><span class="probability-value">${prediction.drawWinPct}%</span></div>`
    : "";

  const liveBlock =
    game.isLive && game.homeScore != null
      ? `<div class="live-score">${game.awayTeam} ${game.awayScore} – ${game.homeScore} ${game.homeTeam}</div>`
      : game.isFinal && game.homeScore != null
        ? `<div class="final-score">Final: ${game.awayTeam} ${game.awayScore} – ${game.homeScore} ${game.homeTeam}</div>`
        : "";

  return `
    <section class="prediction-panel">
      <div class="prediction-head">
        <div>
          <span class="rank-badge">#${game.predictionRank || "?"}</span>
          <span class="confidence-label ${labelClass}">${prediction.confidenceLabel || ""}</span>
          <h3 class="prediction-title">${prediction.outcomeLabel}</h3>
          <p class="prediction-subtitle">${prediction.confidence}% confidence · sorted highest to lowest</p>
        </div>
        <div class="confidence-badge">${prediction.confidence}%</div>
      </div>
      ${liveBlock}
      ${probabilityCompare}
      <div class="probability-bar ${prediction.drawWinPct != null ? "three-way" : ""}">
        <div class="probability-team ${homeFavored ? "favored" : ""}"><span class="probability-label">${game.homeTeam || "Home"} (blended)</span><span class="probability-value">${prediction.homeWinPct}%</span></div>
        ${drawBlock}
        <div class="probability-team ${awayFavored ? "favored" : ""}"><span class="probability-label">${game.awayTeam || "Away"} (blended)</span><span class="probability-value">${prediction.awayWinPct}%</span></div>
      </div>
      ${edgeBlock}
      ${totalBlock}
      <div class="why-panel">
        <h4>Why ${prediction.predictedSide === "draw" ? "draw" : prediction.predictedWinner}?</h4>
        <p class="why-summary">${prediction.whyTheyWin || "Analysis pending."}</p>
        ${reasons ? `<ul class="why-list">${reasons}</ul>` : ""}
        ${sources ? `<div class="source-list">${sources}</div>` : ""}
      </div>
      <div class="factor-list">${factors}</div>
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
  const visible = filterGames(games);

  renderAccuracyPanel();

  if (!visible.length) {
    const scheduleOnly = lastPayload?.liveScheduleOnly;
    const buildError = lastPayload?.error;
    gamesEl.innerHTML = `<div class="empty-state">No ${leagueLabel} games match your filters.${buildError ? ` Build error: ${buildError}` : ""}${scheduleOnly ? " Live schedule loaded — predictions appear once GitHub Actions builds that date." : ""}</div>`;
    renderTopPicks([]);
    renderStats(lastPayload || {}, 0);
    return;
  }

  renderTopPicks(visible);
  renderStats(lastPayload || {}, visible.length);

  const lineupLabel = lineupLabelForSport(sport);
  gamesEl.innerHTML = visible
    .map((game) => {
      const lines = game.lines || [];
      const rows = lines.length
        ? lines
            .map(
              (line) => `<tr><td>${line.sportsbook || "—"}</td><td>${line.viewType || "—"}</td><td>${renderLineChips(line.openingLine)}</td><td>${renderLineChips(line.currentLine)}</td></tr>`
            )
            .join("")
        : `<tr><td colspan="4">No odds available yet.</td></tr>`;

      const records = game.awayRecord || game.homeRecord ? `${game.awayTeam} ${game.awayRecord || "—"} · ${game.homeTeam} ${game.homeRecord || "—"}` : null;
      const pitchers = sport === "mlb" && (game.awayPitcher?.name || game.homePitcher?.name) ? `SP: ${game.awayPitcher?.name || "TBD"} vs ${game.homePitcher?.name || "TBD"}` : null;
      const metaParts = [formatDateTime(game.startDate), game.venueName || "Venue TBD"];
      if (game.broadcast) metaParts.push(game.broadcast);
      if (records) metaParts.push(records);
      if (pitchers) metaParts.push(pitchers);

      const shareUrl = `${window.location.origin}${window.location.pathname}#game-${game.eventId}`;

      return `
        <article class="game-card" id="game-${game.eventId}">
          <details class="game-details" open>
            <summary class="game-summary-bar">
              <span class="rank-badge small">#${game.predictionRank || "?"}</span>
              <span class="summary-matchup">${game.matchup || "Unknown"}</span>
              <span class="summary-pick">${game.prediction?.outcomeLabel || ""} · ${pickProbabilitySummary(game.prediction) || `${game.prediction?.confidence || "?"}%`}</span>
            </summary>
            <div class="game-details-body">
              ${renderPrediction(game)}
              <div class="game-head">
                <div>
                  <h2 class="matchup">${game.matchup || "Unknown matchup"}</h2>
                  <p class="meta">${metaParts.join(" · ")}</p>
                </div>
                <div class="game-actions">
                  <span class="status-pill ${game.isLive ? "live" : ""}">${game.gameStatusText || "Scheduled"}</span>
                  <button type="button" class="share-btn" data-share-url="${shareUrl}" data-share-title="${game.prediction?.outcomeLabel || game.matchup}">Share</button>
                </div>
              </div>
              ${renderLineups(game, lineupLabel)}
              ${renderMajorInjuries(game)}
              <table class="lines-table">
                <thead><tr><th>Book</th><th>Market</th><th>Opening</th><th>Current</th></tr></thead>
                <tbody>${rows}</tbody>
              </table>
            </div>
          </details>
        </article>
      `;
    })
    .join("");

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
          showBanner("Link copied to clipboard.");
        }
      } catch {
        /* user cancelled */
      }
    });
  });

  const hash = window.location.hash;
  if (hash.startsWith("#game-")) {
    document.querySelector(hash)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function filterPayloadByView(payload, view) {
  if (!view || view === "Spread|MoneyLine|Total") {
    const allowed = ["Spread", "MoneyLine", "Total"];
    return {
      ...payload,
      games: (payload.games || []).map((game) => ({
        ...game,
        lines: (game.lines || []).filter((line) => allowed.some((name) => (line.viewType || "").includes(name))),
      })),
    };
  }
  return {
    ...payload,
    games: (payload.games || []).map((game) => ({
      ...game,
      lines: (game.lines || []).filter((line) => (line.viewType || "").includes(view)),
    })),
  };
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

    games.push({
      league,
      leagueLabel: SPORT_LABELS[league] || league,
      eventId: event.id,
      startDate: competition.date || event.date,
      awayTeam,
      homeTeam,
      matchup: awayTeam && homeTeam ? `${awayTeam} @ ${homeTeam}` : event.name,
      gameStatusText: status.description || status.shortDetail || "Scheduled",
      venueName: competition.venue?.fullName,
      awayScore: away?.score,
      homeScore: home?.score,
      isLive: status.state === "in",
      isFinal: Boolean(status.completed) || status.state === "post",
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
  const games = parseEspnScoreboard(data, league);
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

  for (const filePath of candidatePaths) {
    const response = await fetch(staticDataUrl(String(filePath).replace(/^\//, ""), force));
    if (!response.ok) continue;
    const payload = await response.json();
    const gameCount = payload.gameCount ?? payload.games?.length ?? 0;
    if ((payload.scheduleDate || dateValue) === dateValue || gameCount > 0) {
      return payload;
    }
  }

  let fallbackPayload = null;
  const fallbackResponse = await fetch(staticDataUrl(`data/${league}.json`, force));
  if (fallbackResponse.ok) {
    fallbackPayload = await fallbackResponse.json();
    if (fallbackPayload.scheduleDate === dateValue) {
      return fallbackPayload;
    }
  }

  const livePayload = await fetchEspnSchedule(league, dateValue);
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
    scores[String(event.id)] = {
      awayScore: away?.score,
      homeScore: home?.score,
      isLive: status.state === "in",
      isFinal: Boolean(status.completed) || status.state === "post",
      gameStatusText: status.description || status.shortDetail,
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
    lastPayload = { ...lastPayload, games: mergeLiveScores(lastPayload.games, liveScores) };
    renderGames(lastPayload.games);
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

async function fetchDashboardPayload(params, { force = false } = {}) {
  const league = params.get("league") || sportSelect.value;

  if (IS_STATIC_HOST) {
    manifestData = manifestData || (await fetchManifest({ force }));
    if (manifestData?.leagues?.length) {
      populateDateSelect(league, activeScheduleDate || getSelectedDate());
    }
    accuracyData = await fetchAccuracy({ force });

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

    return filterPayloadByView(payload, params.get("view") || viewFilter.value);
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
      populateDateSelect(league, activeScheduleDate || getSelectedDate());
    }
    return payload;
  }
  throw new Error("Could not reach the dashboard API.");
}

async function loadDashboard(force = false) {
  if (loadingDashboard && !force) return;
  loadingDashboard = true;
  refreshBtn.disabled = true;
  refreshBtn.textContent = "Loading…";
  if (sportSelect.value !== "overview") {
    gamesEl.innerHTML = `<div class="empty-state loading-state">Loading ${SPORT_LABELS[sportSelect.value] || sportSelect.value}…</div>`;
  }

  const requestedDate = getSelectedDate();
  setActiveScheduleDate(requestedDate, { syncSelect: true });
  const params = new URLSearchParams({
    league: sportSelect.value,
    date: requestedDate,
    view: viewFilter.value,
  });

  try {
    const payload = await fetchDashboardPayload(params, { force });
    lastPayload = payload;
    updateFreshnessNote();

    if (sportSelect.value === "overview") {
      hideBanner();
      renderOverview();
    } else {
      const scheduleDate = payload.defaultScheduleDate || payload.scheduleDate || requestedDate;
      setActiveScheduleDate(scheduleDate, { syncSelect: true });
      populateDateSelect(sportSelect.value, scheduleDate);
      statDate.textContent = `${scheduleDate}${payload.scheduleTimezone ? ` (${payload.scheduleTimezone})` : ""}`;

      if (IS_STATIC_HOST) {
        let note = `Predictions update every 30 min on GitHub. Live scores refresh every ${manifestData?.liveScoreRefreshSeconds || 90}s in your browser.`;
        if (payload.scheduleTimezone) {
          note += ` MLB/US sports use ${payload.scheduleTimezone} schedule dates (not your local calendar day).`;
        }
        if (payload._liveFallback) {
          note += ` Showing live ESPN schedule for ${requestedDate} (predictions load when snapshot is built).`;
        } else if (payload._dateFallback) {
          note += ` No snapshot for ${payload._requestedDate}; showing ${payload.scheduleDate}.`;
        }
        showBanner(note);
      } else if ((payload.sportsbookCount || 0) === 0) {
        showBanner("Games ranked by win probability. Odds appear when ESPN or sportsbooks publish them.");
      } else {
        hideBanner();
      }
      let games = payload.games || [];
      if (games.length === 0 && IS_STATIC_HOST && !payload._liveFallback) {
        const meta = leagueMeta(sportSelect.value);
        const retryDate = meta?.defaultDate;
        if (retryDate && retryDate !== requestedDate && retryDate !== payload.scheduleDate) {
          try {
            const retryPayload = await fetchStaticPayloadForDate(sportSelect.value, retryDate, { force });
            if ((retryPayload.games || []).length > 0) {
              lastPayload = { ...retryPayload, _requestedDate: requestedDate, _dateFallback: true };
              setActiveScheduleDate(retryPayload.scheduleDate || retryDate, { syncSelect: true });
              populateDateSelect(sportSelect.value, retryPayload.scheduleDate || retryDate);
              statDate.textContent = `${retryPayload.scheduleDate || retryDate}${retryPayload.scheduleTimezone ? ` (${retryPayload.scheduleTimezone})` : ""}`;
              showBanner(`No games for ${requestedDate}. Showing ${retryPayload.scheduleDate || retryDate} snapshot.`);
              games = retryPayload.games;
            }
          } catch {
            /* keep empty state */
          }
        }
      }

      renderGames(games);
      if (games.length === 0 && payload.error) {
        showBanner(`Data build error: ${payload.error}. Try another date or re-run GitHub Actions.`);
      } else if (games.length === 0 && !payload.error) {
        showBanner(`No games for ${requestedDate}. Pick another date from the dropdown or try Refresh.`);
      }
      resetLiveScorePolling();
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
    refreshBtn.textContent = "Refresh";
  }
}

function resetAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = null;
  if (autoRefresh.checked) refreshTimer = setInterval(() => loadDashboard(false), 120000);
}

function onSportChange() {
  if (sportSelect.value === "overview") {
    populateDateSelect("overview");
  } else {
    activeScheduleDate = defaultDateForSport(sportSelect.value);
    populateDateSelect(sportSelect.value, activeScheduleDate);
  }
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

  if (IS_STATIC_HOST) {
    manifestData = await fetchManifest({ force: false });
  }

  sportSelect.addEventListener("change", onSportChange);
  confidenceFilter.addEventListener("change", () => renderGames(lastPayload?.games || []));
  teamSearch.addEventListener("input", () => renderGames(lastPayload?.games || []));
  refreshBtn.addEventListener("click", () => loadDashboard(true));
  viewFilter.addEventListener("change", () => loadDashboard(true));
  dateSelect.addEventListener("change", () => onDateSelected(dateSelect.value));
  datePrevBtn?.addEventListener("click", () => onDateNav(-1));
  dateNextBtn?.addEventListener("click", () => onDateNav(1));
  autoRefresh.addEventListener("change", resetAutoRefresh);
  liveScoresToggle?.addEventListener("change", resetLiveScorePolling);

  populateDateSelect(sportSelect.value, defaultDateForSport(sportSelect.value));
  await loadDashboard(true);
  resetAutoRefresh();
  resetLiveScorePolling();
}

initDashboard().catch((error) => {
  showBanner(error.message || "Failed to start dashboard.");
  gamesEl.innerHTML = `<div class="empty-state">Could not load dashboard. Tap Refresh or clear site data in your browser.</div>`;
});
