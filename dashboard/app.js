const sportSelect = document.getElementById("sport-select");
const dateInput = document.getElementById("date-input");
const viewFilter = document.getElementById("view-filter");
const refreshBtn = document.getElementById("refresh-btn");
const autoRefresh = document.getElementById("auto-refresh");
const gamesEl = document.getElementById("games");
const bannerEl = document.getElementById("banner");
const dashboardTitle = document.getElementById("dashboard-title");

const statGames = document.getElementById("stat-games");
const statTopPick = document.getElementById("stat-top-pick");
const statLeague = document.getElementById("stat-league");
const statDate = document.getElementById("stat-date");
const statUpdated = document.getElementById("stat-updated");

const API_PATHS = ["/api/games", "/api/odds"];
const SPORT_DEFAULT_DAYS = { mlb: 1, worldcup: 0, afl: 0 };
const SPORT_LABELS = {
  mlb: "MLB Baseball",
  worldcup: "FIFA World Cup",
  afl: "AFL",
};

let refreshTimer = null;
let loadingDashboard = false;
let lastPayload = null;

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

function renderStats(payload) {
  statGames.textContent = payload.gameCount ?? 0;
  statDate.textContent = payload.scheduleDate || dateInput.value || "—";
  statTopPick.textContent = payload.topPick || "—";
  statLeague.textContent = payload.leagueLabel || SPORT_LABELS[sportSelect.value] || "—";
  statUpdated.textContent = formatDateTime(payload.fetchedAt);
  dashboardTitle.textContent = `${payload.leagueLabel || SPORT_LABELS[sportSelect.value] || "Sports"} Dashboard`;
}

function renderLineupColumn(teamName, lineup) {
  if (!lineup || !lineup.batters?.length) {
    return `
      <div class="lineup-column">
        <h5>${teamName || "Team"}</h5>
        <p class="lineup-note">${lineup?.note || "Lineup not available yet."}</p>
      </div>
    `;
  }

  const rows = lineup.batters
    .map((batter) => {
      if (batter.order) {
        return `<li><span class="lineup-order">${batter.order}</span> ${batter.name} <span class="lineup-pos">${batter.position || ""}</span></li>`;
      }
      return `<li>${batter.name} <span class="lineup-pos">${batter.statLine || batter.position || ""}</span></li>`;
    })
    .join("");

  return `
    <div class="lineup-column">
      <h5>${teamName || "Team"}</h5>
      <p class="lineup-note">${lineup.note || ""}</p>
      <ol class="lineup-list">${rows}</ol>
    </div>
  `;
}

function renderLineups(game, lineupLabel) {
  const hasLineup =
    game.homeLineup?.batters?.length ||
    game.awayLineup?.batters?.length ||
    game.homeLineup?.status !== "unavailable" ||
    game.awayLineup?.status !== "unavailable";

  if (!hasLineup && sportSelect.value !== "mlb") {
    return "";
  }

  return `
    <section class="detail-panel">
      <h4>${lineupLabel || "Lineup"}</h4>
      <div class="lineup-grid">
        ${renderLineupColumn(game.awayTeam, game.awayLineup)}
        ${renderLineupColumn(game.homeTeam, game.homeLineup)}
      </div>
    </section>
  `;
}

function renderInjuryColumn(teamName, injuries) {
  if (!injuries?.length) {
    return `
      <div class="injury-column">
        <h5>${teamName || "Team"}</h5>
        <p class="lineup-note">No major injuries listed.</p>
      </div>
    `;
  }

  const rows = injuries
    .map(
      (injury) =>
        `<li><strong>${injury.player}</strong> — ${injury.status}${injury.detail ? `: ${injury.detail}` : ""}${injury.returnDate ? ` · return ${injury.returnDate}` : ""}</li>`
    )
    .join("");

  return `
    <div class="injury-column">
      <h5>${teamName || "Team"}</h5>
      <ul class="injury-list">${rows}</ul>
    </div>
  `;
}

function renderMajorInjuries(game) {
  const hasInjuries = game.homeMajorInjuries?.length || game.awayMajorInjuries?.length;
  if (!hasInjuries && sportSelect.value !== "mlb") {
    return "";
  }

  return `
    <section class="detail-panel">
      <h4>Major injuries</h4>
      <div class="lineup-grid">
        ${renderInjuryColumn(game.awayTeam, game.awayMajorInjuries)}
        ${renderInjuryColumn(game.homeTeam, game.homeMajorInjuries)}
      </div>
    </section>
  `;
}

function renderPrediction(game) {
  const prediction = game.prediction;
  if (!prediction) return "";

  const homeFavored = prediction.predictedSide === "home";
  const awayFavored = prediction.predictedSide === "away";
  const factors = (prediction.factors || [])
    .map((factor) => {
      const edgeClass =
        factor.edge === "home" ? "edge-home" : factor.edge === "away" ? "edge-away" : "";
      return `<span class="factor-chip ${edgeClass}" title="${factor.detail}">${factor.label}</span>`;
    })
    .join("");

  const reasons = (prediction.reasons || [])
    .map(
      (reason) =>
        `<li><strong>${reason.title}:</strong> ${reason.detail}${reason.source ? ` <em>(${reason.source})</em>` : ""}</li>`
    )
    .join("");

  const sources = (prediction.dataSources || [])
    .map((source) => `<span class="source-chip">${source}</span>`)
    .join("");

  const drawBlock =
    prediction.drawWinPct != null
      ? `
        <div class="probability-team">
          <span class="probability-label">Draw</span>
          <span class="probability-value">${prediction.drawWinPct}%</span>
        </div>
      `
      : "";

  return `
    <section class="prediction-panel">
      <div class="prediction-head">
        <div>
          <span class="rank-badge">#${game.predictionRank || "?"}</span>
          <h3 class="prediction-title">${prediction.outcomeLabel}</h3>
          <p class="prediction-subtitle">${prediction.confidence}% confidence · sorted highest to lowest</p>
        </div>
        <div class="confidence-badge">${prediction.confidence}%</div>
      </div>
      <div class="probability-bar">
        <div class="probability-team ${homeFavored ? "favored" : ""}">
          <span class="probability-label">${game.homeTeam || "Home"}</span>
          <span class="probability-value">${prediction.homeWinPct}%</span>
        </div>
        ${drawBlock}
        <div class="probability-team ${awayFavored ? "favored" : ""}">
          <span class="probability-label">${game.awayTeam || "Away"}</span>
          <span class="probability-value">${prediction.awayWinPct}%</span>
        </div>
      </div>
      <div class="why-panel">
        <h4>Why ${prediction.predictedWinner}?</h4>
        <p class="why-summary">${prediction.whyTheyWin || "Analysis pending."}</p>
        ${reasons ? `<ul class="why-list">${reasons}</ul>` : ""}
        ${sources ? `<div class="source-list">${sources}</div>` : ""}
      </div>
      <div class="factor-list">${factors}</div>
    </section>
  `;
}

function lineupLabelForSport(sport) {
  if (sport === "mlb") return "Batting lineup";
  return "Key players";
}

function renderGames(games) {
  const sport = sportSelect.value;
  const leagueLabel = SPORT_LABELS[sport] || "games";

  if (!games.length) {
    gamesEl.innerHTML = `<div class="empty-state">No ${leagueLabel} games scheduled for this date.</div>`;
    return;
  }

  const lineupLabel = lineupLabelForSport(sport);

  gamesEl.innerHTML = games
    .map((game) => {
      const lines = game.lines || [];
      const rows = lines.length
        ? lines
            .map(
              (line) => `
                <tr>
                  <td>${line.sportsbook || "—"}</td>
                  <td>${line.viewType || "—"}</td>
                  <td>${renderLineChips(line.openingLine)}</td>
                  <td>${renderLineChips(line.currentLine)}</td>
                </tr>
              `
            )
            .join("")
        : `<tr><td colspan="4">No odds available yet for this game.</td></tr>`;

      const records =
        game.awayRecord || game.homeRecord
          ? `${game.awayTeam} ${game.awayRecord || "—"} · ${game.homeTeam} ${game.homeRecord || "—"}`
          : null;

      const pitchers =
        sport === "mlb" && (game.awayPitcher?.name || game.homePitcher?.name)
          ? `SP: ${game.awayPitcher?.name || "TBD"} vs ${game.homePitcher?.name || "TBD"}`
          : null;

      const metaParts = [formatDateTime(game.startDate), game.venueName || "Venue TBD"];
      if (game.broadcast) metaParts.push(game.broadcast);
      if (records) metaParts.push(records);
      if (pitchers) metaParts.push(pitchers);

      return `
        <article class="game-card">
          ${renderPrediction(game)}
          <div class="game-head">
            <div>
              <h2 class="matchup">${game.matchup || "Unknown matchup"}</h2>
              <p class="meta">${metaParts.join(" · ")}</p>
            </div>
            <span class="status-pill">${game.gameStatusText || "Scheduled"}</span>
          </div>
          ${renderLineups(game, lineupLabel)}
          ${renderMajorInjuries(game)}
          <table class="lines-table">
            <thead>
              <tr>
                <th>Book</th>
                <th>Market</th>
                <th>Opening</th>
                <th>Current</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </article>
      `;
    })
    .join("");
}

async function fetchDashboardPayload(params, { force = false } = {}) {
  if (window.location.protocol === "file:") {
    throw new Error(
      "Open the dashboard through the local server, not the HTML file directly. Run start-dashboard.bat first."
    );
  }

  if (force) {
    params.set("force", "1");
  }

  let lastError = null;

  for (const path of API_PATHS) {
    let response;
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000);
      response = await fetch(`${path}?${params.toString()}`, { signal: controller.signal });
      clearTimeout(timeoutId);
    } catch (error) {
      if (error.name === "AbortError") {
        lastError = new Error("Request timed out while loading games. Try again in a moment.");
      } else {
        lastError = new Error(
          "Could not reach the dashboard server. Make sure start-dashboard.bat is still running."
        );
      }
      continue;
    }

    const contentType = response.headers.get("content-type") || "";
    const body = await response.text();

    if (!contentType.includes("application/json")) {
      lastError = new Error(
        "Dashboard server returned HTML instead of JSON. Close old dashboard windows, run start-dashboard.bat again, then hard-refresh the page (Ctrl+F5)."
      );
      continue;
    }

    let payload;
    try {
      payload = JSON.parse(body);
    } catch (error) {
      lastError = new Error("Dashboard server returned invalid JSON.");
      continue;
    }

    if (!response.ok) {
      throw new Error(payload.error || `Request failed (${response.status})`);
    }

    return payload;
  }

  throw lastError || new Error("Could not reach the dashboard API.");
}

async function loadDashboard(force = false) {
  if (loadingDashboard) {
    return;
  }
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

    if (payload.stale && payload.refreshError) {
      showBanner(`Live refresh failed. Showing cached data from ${payload.cacheAgeSeconds || "?"}s ago.`);
    } else if (payload.fromCache) {
      showBanner(`Loaded cached data (${payload.cacheAgeSeconds || "?"}s old). Click Refresh for a full update.`);
    } else if ((payload.sportsbookCount || 0) === 0) {
      showBanner("Games ranked by win probability. Odds will appear when ESPN or sportsbooks publish them.");
    }

    renderStats(payload);
    renderGames(payload.games || []);
  } catch (error) {
    if (lastPayload) {
      showBanner(`${error.message} Showing last loaded data.`);
      renderStats(lastPayload);
      renderGames(lastPayload.games || []);
    } else {
      showBanner(error.message || "Failed to load dashboard data.");
      gamesEl.innerHTML = `<div class="empty-state">Could not load games. Run start-dashboard.bat, keep that window open, then refresh this page.</div>`;
    }
  } finally {
    loadingDashboard = false;
    refreshBtn.disabled = false;
    refreshBtn.textContent = "Refresh";
  }
}

function resetAutoRefresh() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
  if (autoRefresh.checked) {
    refreshTimer = setInterval(() => loadDashboard(false), 120000);
  }
}

function onSportChange() {
  dateInput.value = defaultDateForSport(sportSelect.value);
  loadDashboard(true);
}

sportSelect.addEventListener("change", onSportChange);
dateInput.value = defaultDateForSport(sportSelect.value);
refreshBtn.addEventListener("click", () => loadDashboard(true));
viewFilter.addEventListener("change", () => loadDashboard(true));
dateInput.addEventListener("change", () => loadDashboard(true));
autoRefresh.addEventListener("change", resetAutoRefresh);

loadDashboard(true);
resetAutoRefresh();
