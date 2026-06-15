const sportSelect = document.getElementById("sport-select");
const dateInput = document.getElementById("date-input");
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

const GITHUB_PAGES_REPO = "bradleyschulz88/Predictions-Model";
const UPDATE_WORKFLOW_URL = `https://github.com/${GITHUB_PAGES_REPO}/actions/workflows/pages.yml`;
const IS_STATIC_HOST =
  window.location.hostname.endsWith("github.io") ||
  new URLSearchParams(window.location.search).has("static");

const SPORT_DEFAULT_DAYS = { mlb: 1, nfl: 0, nba: 0, worldcup: 0, epl: 0, afl: 0 };
const SPORT_LABELS = {
  mlb: "MLB Baseball",
  nfl: "NFL Football",
  nba: "NBA Basketball",
  worldcup: "FIFA World Cup",
  epl: "Premier League",
  afl: "AFL",
};

let refreshTimer = null;
let loadingDashboard = false;
let lastPayload = null;
let accuracyData = null;

function offsetIso(daysAhead) {
  const now = new Date();
  now.setDate(now.getDate() + daysAhead);
  const offset = now.getTimezoneOffset();
  const local = new Date(now.getTime() - offset * 60 * 1000);
  return local.toISOString().slice(0, 10);
}

function defaultDateForSport(sport) {
  return offsetIso(SPORT_DEFAULT_DAYS[sport] ?? 0);
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
    const confidence = game.prediction?.confidence ?? 0;
    if (confidence < minConfidence) return false;
    if (!query) return true;
    const haystack = `${game.homeTeam || ""} ${game.awayTeam || ""} ${game.matchup || ""}`.toLowerCase();
    return haystack.includes(query);
  });
}

function renderStats(payload, visibleCount) {
  statGames.textContent = visibleCount ?? payload.gameCount ?? 0;
  statDate.textContent = payload.scheduleDate || dateInput.value || "—";
  statTopPick.textContent = payload.topPick || "—";
  statLeague.textContent = payload.leagueLabel || SPORT_LABELS[sportSelect.value] || "—";
  statUpdated.textContent = formatDateTime(payload.fetchedAt);
  dashboardTitle.textContent = `${payload.leagueLabel || SPORT_LABELS[sportSelect.value] || "Sports"} Predictions`;

  const acc = accuracyData?.summary?.last7Days;
  if (acc?.pct != null) {
    statAccuracy.textContent = `${acc.pct}% (${acc.correct}/${acc.total})`;
  } else {
    statAccuracy.textContent = "Building…";
  }
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
    ? `<div class="edge-panel"><strong>Model vs market:</strong> ${prediction.modelEdge.modelPct}% model · ${prediction.modelEdge.marketPct}% market · <span class="edge-chip">${prediction.modelEdge.edgeLabel}</span></div>`
    : "";

  const totalBlock = prediction.totalPick
    ? `<div class="total-panel"><strong>Total pick:</strong> ${prediction.totalPick.pick} (${prediction.totalPick.confidence}% confidence)<br><span class="lineup-note">${prediction.totalPick.detail}</span></div>`
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
      <div class="probability-bar ${prediction.drawWinPct != null ? "three-way" : ""}">
        <div class="probability-team ${homeFavored ? "favored" : ""}"><span class="probability-label">${game.homeTeam || "Home"}</span><span class="probability-value">${prediction.homeWinPct}%</span></div>
        ${drawBlock}
        <div class="probability-team ${awayFavored ? "favored" : ""}"><span class="probability-label">${game.awayTeam || "Away"}</span><span class="probability-value">${prediction.awayWinPct}%</span></div>
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
  const sport = sportSelect.value;
  const leagueLabel = SPORT_LABELS[sport] || "games";
  const visible = filterGames(games);

  if (!visible.length) {
    gamesEl.innerHTML = `<div class="empty-state">No ${leagueLabel} games match your filters.</div>`;
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
              <span class="summary-pick">${game.prediction?.outcomeLabel || ""} · ${game.prediction?.confidence || "?"}%</span>
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
  if (force) url.searchParams.set("t", String(Date.now()));
  return url.toString();
}

async function fetchStaticPayload(league, { force = false } = {}) {
  const response = await fetch(staticDataUrl(`data/${league}.json`, force));
  if (!response.ok) throw new Error(`Could not load ${league} data (${response.status}).`);
  return response.json();
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
    const payload = await fetchStaticPayload(league, { force });
    accuracyData = await fetchAccuracy({ force });
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
    return payload;
  }
  throw new Error("Could not reach the dashboard API.");
}

async function loadDashboard(force = false) {
  if (loadingDashboard) return;
  loadingDashboard = true;
  refreshBtn.disabled = true;
  refreshBtn.textContent = "Loading…";

  const params = new URLSearchParams({
    league: sportSelect.value,
    date: dateInput.value,
    view: viewFilter.value,
  });

  try {
    const payload = await fetchDashboardPayload(params, { force });
    lastPayload = payload;
    hideBanner();

    if (IS_STATIC_HOST) {
      showBanner(`Published snapshot · updates hourly. Rebuild: ${UPDATE_WORKFLOW_URL}`);
      if (payload.scheduleDate) dateInput.value = payload.scheduleDate;
    } else if ((payload.sportsbookCount || 0) === 0) {
      showBanner("Games ranked by win probability. Odds appear when ESPN or sportsbooks publish them.");
    }

    renderGames(payload.games || []);
  } catch (error) {
    if (lastPayload) {
      showBanner(`${error.message} Showing last loaded data.`);
      renderGames(lastPayload.games || []);
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
  dateInput.value = defaultDateForSport(sportSelect.value);
  loadDashboard(true);
}

function configureStaticMode() {
  if (!IS_STATIC_HOST) return;
  dateInput.disabled = true;
}

function registerServiceWorker() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }
}

configureStaticMode();
registerServiceWorker();
sportSelect.addEventListener("change", onSportChange);
confidenceFilter.addEventListener("change", () => renderGames(lastPayload?.games || []));
teamSearch.addEventListener("input", () => renderGames(lastPayload?.games || []));
dateInput.value = defaultDateForSport(sportSelect.value);
refreshBtn.addEventListener("click", () => loadDashboard(true));
viewFilter.addEventListener("change", () => loadDashboard(true));
dateInput.addEventListener("change", () => loadDashboard(true));
autoRefresh.addEventListener("change", resetAutoRefresh);

loadDashboard(true);
resetAutoRefresh();
