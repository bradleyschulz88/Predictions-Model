/* Edge Board front end.
 *
 * Four views, all fed from the JSON the build already writes:
 *
 *   data/overview.json    every published pick across every league, ranked by
 *                         expected value, with per-league summaries
 *   data/{league}[_date]  the full slate for one league, with features
 *   data/accuracy.json    the graded record, units, CLV, side markets
 *   data/evaluation.json  log loss / Brier / AUC / reliability / home bias
 *   data/model_weights.json  the fitted coefficients, for the decomposition
 *
 * Nothing here invents a number. Where a value is missing the view says so,
 * because "no data" and "zero" are different facts and only one of them is an
 * edge.
 */

/* ------------------------------------------------------------------ helpers */
const $ = (s, r = document) => r.querySelector(s);
const el = (t, c, h) => {
  const n = document.createElement(t);
  if (c) n.className = c;
  if (h != null) n.innerHTML = h;
  return n;
};
const esc = (s) =>
  String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
  );
const pct = (v, d = 1) => (v == null ? "—" : Number(v).toFixed(d) + "%");
const sgn = (v, d = 1) => (v == null ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(d));
const american = (o) => (o == null ? "—" : (o > 0 ? "+" : "") + o);

/* American odds -> implied probability. These must never be subtracted from
   one another: the scale is discontinuous at zero, so -112 to +100 differs by
   2.8 points of probability and not by "212" of anything. */
const impliedFrom = (o) => (o == null ? null : (o > 0 ? 100 / (o + 100) : -o / (-o + 100)) * 100);

const teamShort = (n) => {
  const parts = String(n || "").trim().split(" ");
  return parts[parts.length - 1] || String(n || "");
};

/* The overview labels leagues "MLB Baseball" and "WNBA Basketball", which wrap
   onto two lines in a tile heading and read oddly in a chip. The id is already
   the short form everyone uses. */
const leagueShort = (play) => String(play.league || "").toUpperCase();

/* The descriptive half of the label, or nothing when it adds nothing. */
function leagueSport(L) {
  const label = String(L.label || "").trim();
  const short = String(L.id || "").toUpperCase();
  const rest = label.replace(new RegExp(`^${short}\\s*`, "i"), "").trim();
  return rest && rest.toUpperCase() !== short ? rest : "";
}

const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

/* Kick-off in the league's own clock, not the viewer's, because a Melbourne
   game listed at 4am local reads as an error rather than a fixture. */
const LEAGUE_TZ = {
  mlb: "America/New_York",
  nfl: "America/New_York",
  nba: "America/New_York",
  wnba: "America/New_York",
  epl: "Europe/London",
  afl: "Australia/Melbourne",
  worldcup: "UTC",
};

function startTime(iso, league) {
  if (!iso) return null;
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return null;
  try {
    return new Intl.DateTimeFormat(undefined, {
      hour: "numeric",
      minute: "2-digit",
      timeZone: LEAGUE_TZ[league] || "UTC",
      timeZoneName: "short",
    }).format(when);
  } catch {
    return when.toISOString().slice(11, 16) + " UTC";
  }
}

async function getJson(path) {
  const url = new URL(path, window.location.href);
  url.searchParams.set("v", String(Math.floor(Date.now() / 300000)));
  const response = await fetch(url.toString());
  if (!response.ok) throw new Error(`${path} responded ${response.status}`);
  return response.json();
}

/* ------------------------------------------------------------------- state */
const S = {
  overview: null,
  accuracy: null,
  evaluation: null,
  weights: null,
  ablation: null,
  manifest: null,
  slates: {},          // `${league}:${date}` -> payload
  sport: null,
  dateByLeague: {},    // league -> ISO date currently showing
  sort: "ev",
  filter: "pub",
  failures: [],
};

/* --------------------------------------------------------------- dates */

function manifestLeague(id) {
  return (S.manifest?.leagues || []).find((x) => x.id === id) || null;
}

function shiftIsoDate(iso, days) {
  const d = new Date(iso + "T12:00:00Z");
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

/* Every date this league actually has a built slate for, oldest first. Falls
   back to just today's date when the manifest failed to load or carries
   nothing for this league, so date navigation degrades to "today only"
   instead of throwing. */
function datesFor(id) {
  const m = manifestLeague(id);
  if (m?.availableDates?.length) return m.availableDates.slice().sort();
  const O = (S.overview?.leagues || []).find((x) => x.id === id);
  return O?.scheduleDate ? [O.scheduleDate] : [];
}

function todayFor(id) {
  return manifestLeague(id)?.defaultDate ||
    (S.overview?.leagues || []).find((x) => x.id === id)?.scheduleDate || null;
}

function selectedDateFor(id) {
  return S.dateByLeague[id] || todayFor(id) || datesFor(id)[0] || null;
}

function setDate(iso) {
  S.dateByLeague[S.sport] = iso;
  renderSport();
}

/* "Today" and "Tomorrow" are read off the manifest's own defaultDate for that
   league -- computed server-side in the league's own timezone -- rather than
   compared against the viewer's local clock, which would drift near a
   timezone boundary (an AFL slate is built in Melbourne time; a viewer in
   Chicago is up to a day off if the comparison were done locally). */
function describeDate(iso, league) {
  if (!iso) return "";
  const today = todayFor(league);
  if (today) {
    if (iso === today) return "Today";
    if (iso === shiftIsoDate(today, 1)) return "Tomorrow";
    if (iso === shiftIsoDate(today, -1)) return "Yesterday";
  }
  const d = new Date(iso + "T12:00:00Z");
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat(undefined, { weekday: "short", month: "short", day: "numeric" }).format(d);
}

function renderDateBar() {
  const bar = $("#dateBar");
  const dates = datesFor(S.sport);
  if (dates.length <= 1) { bar.style.display = "none"; return; }
  bar.style.display = "";

  const selected = selectedDateFor(S.sport);
  const today = todayFor(S.sport);
  const idx = dates.indexOf(selected);

  $("#datePrev").disabled = idx <= 0;
  $("#dateNext").disabled = idx < 0 || idx >= dates.length - 1;

  const chips = $("#dateChips");
  chips.innerHTML = "";
  dates.forEach((iso) => {
    const d = new Date(iso + "T12:00:00Z");
    const b = el("button", "datechip" + (iso === today ? " istoday" : ""));
    b.type = "button";
    b.setAttribute("aria-pressed", String(iso === selected));
    b.innerHTML = `<span class="dow">${new Intl.DateTimeFormat(undefined, { weekday: "short" }).format(d)}</span>` +
      `<span>${new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(d)}</span>`;
    b.title = describeDate(iso, S.sport);
    b.addEventListener("click", () => setDate(iso));
    chips.appendChild(b);
  });
}

/* ------------------------------------------------- market and calibration */

/* Does the pick go against the side the market favours?
 *
 * Being more confident than the market on the SAME side is not fading it.
 * Conflating the two tags almost every pick a fade, which makes the label
 * useless -- on the graded record only about a fifth genuinely fade the price.
 */
function fadesMarket(play) {
  const market = play.marketPct;
  const confidence = play.confidence;
  if (market == null || confidence == null) return null;
  return market < 50;
}

/* Graded picks this league has of its own.
 *
 * The reliability curve every calibration reading comes from is pooled across
 * every league at once. That is the right way to build it -- no single league
 * has enough graded games to fit its own curve -- but it means a band reading
 * "n=139" can be almost entirely another sport. True as a prior, misleading if
 * presented as this league's record.
 *
 * NBA and NFL are the live case: both start seasons with zero graded games
 * here, so every calibration number shown against one of their picks is
 * borrowed from baseball until they build a record of their own. */
const MIN_LEAGUE_HISTORY = 30;

function leagueGraded(league) {
  const row = (S.accuracy?.summary?.byLeague || {})[league];
  return row?.total ?? 0;
}

/* Text for how far a league's own record backs a calibration reading, or null
   when it has enough history to speak for itself. */
function borrowedCalibrationNote(league) {
  const graded = leagueGraded(league);
  if (graded >= MIN_LEAGUE_HISTORY) return null;
  const label = String(league || "").toUpperCase();
  return graded === 0
    ? `No ${label} pick has been graded here yet, so this band is borrowed entirely from other leagues. Treat it as a prior about the model, not a record of its ${label} form.`
    : `Only ${plural(graded, "graded " + label + " pick", "graded " + label + " picks")} so far, so this band is mostly other leagues. It describes the model, not its ${label} form.`;
}

/* What games in this confidence band have actually done. More honest than a
   fabricated interval: it is measured, not modelled. */
function bucketFor(confidence) {
  if (confidence == null) return null;
  return (S.evaluation?.reliability || []).find((b) => {
    const [lo, hi] = String(b.range).split("-").map(Number);
    return confidence >= lo && confidence < hi;
  }) || null;
}

/* Rebuild the two terms the live model actually fits, so the decomposition
   reports the real decision rather than a plausible-looking one. Mirrors
   model_fit.build_feature_dict: record, centred home/road split and the power
   gap are three measurements of one thing, so they are averaged into a single
   input rather than stacked. */
function decompose(features, league) {
  const W = S.weights;
  if (!W || !features) return null;
  const anchored = W.anchored;
  if (!anchored || !anchored.weights) return null;
  const [w0, wStrength, wMarket] = anchored.weights;
  const centre = W.splitDiffCentre ?? 0;

  const num = (v) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };
  const parts = [];
  const record = num(features.recordDiff);
  if (record != null) parts.push(record);
  const split = num(features.splitDiff);
  if (split != null) parts.push(split - centre);
  const homePower = num(features.homePower);
  const awayPower = num(features.awayPower);
  if (homePower != null && awayPower != null) parts.push(homePower - awayPower);
  const strength = parts.length ? parts.reduce((a, b) => a + b, 0) / parts.length : null;

  const impliedHome = num(features.impliedHome);
  const marketLogit =
    impliedHome != null && impliedHome > 0 && impliedHome < 100
      ? Math.log(impliedHome / 100 / (1 - impliedHome / 100))
      : null;

  const means = anchored.means || {};
  const scales = anchored.scales || {};
  const term = (value, key, weight) =>
    value == null || !scales[key] ? 0 : weight * ((value - (means[key] ?? 0)) / scales[key]);

  const contribStrength = term(strength, "strengthDiff", wStrength);
  const contribMarket = term(marketLogit, "marketLogit", wMarket);
  const leagueIntercept = (W.leagueIntercepts || {})[league] ?? 0;
  const z = w0 + contribStrength + contribMarket + leagueIntercept;

  return {
    contribStrength,
    contribMarket,
    base: w0 + leagueIntercept,
    z,
    homePct: (1 / (1 + Math.exp(-z))) * 100,
    hasStrength: strength != null,
    hasMarket: marketLogit != null,
  };
}

/* --------------------------------------------------------------- tooltip */
const tip = el("div", "tip");
document.body.appendChild(tip);
function showTip(html, ev) {
  tip.innerHTML = html;
  tip.classList.add("on");
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14;
  let y = ev.clientY - r.height / 2;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  y = Math.max(8, Math.min(y, innerHeight - r.height - 8));
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
const hideTip = () => tip.classList.remove("on");

function readings(host, items) {
  host.innerHTML = "";
  items.forEach((it) => {
    const d = el("div", "reading");
    d.appendChild(el("div", "v num", it.v));
    d.appendChild(el("div", "k lbl", it.k));
    host.appendChild(d);
  });
}

/* ============================================================ VIEW: BOARD */
function renderBoard() {
  const O = S.overview;
  const host = $("#boardBody");
  host.innerHTML = "";
  if (!O) {
    host.appendChild(el("div", "failed", "Could not load data/overview.json."));
    return;
  }

  const sum = O.summary || {};
  const slate = (O.leagues || []).reduce((n, l) => n + (l.gameCount || 0), 0);
  $("#boardDate").textContent = new Date(O.builtAt || Date.now()).toLocaleString(undefined, {
    weekday: "long", day: "numeric", month: "long", hour: "2-digit", minute: "2-digit",
  });
  readings($("#readings"), [
    { v: slate, k: "Games" },
    { v: sum.picks ?? "—", k: "Published" },
    { v: sum.priced ?? "—", k: "Priced" },
    { v: sum.positiveEv ?? "—", k: "Positive edge" },
    { v: sum.bestEvPct == null ? "—" : sgn(sum.bestEvPct) + "%", k: "Best edge" },
  ]);

  /* ---- hero: the highest-EV published pick anywhere ---- */
  const best = (O.worthBacking || [])[0] || (O.unpriced || [])[0] || null;
  const heroPanel = el("div", "panel");
  heroPanel.appendChild(
    el("div", "phead",
      `<h2>Best play on the board</h2><span class="n">ranked by expected value, not confidence</span>`)
  );
  if (!best) {
    heroPanel.appendChild(el("div", "loading",
      "No published pick clears the bar on this slate. That is a decision, not a gap — every game came in too close to call."));
  } else {
    heroPanel.appendChild(heroFor(best));
  }
  host.appendChild(heroPanel);

  /* ---- league tiles, clickable through to the sport view ---- */
  host.appendChild(
    el("div", "phead", `<h2>By sport</h2><span class="n">tap a sport for its full slate</span>`)
  ).style.cssText = "border:0;padding:14px 0 0";
  const tiles = el("div", "tiles");
  (O.leagues || []).forEach((L) => tiles.appendChild(leagueTile(L)));
  host.appendChild(tiles);

  /* ---- what was passed on, and what has no price ---- */
  if ((O.passedOn || []).length) host.appendChild(listPanel(
    "Passed on", plural(O.passedOn.length, "priced pick with no edge", "priced picks with no edge"),
    O.passedOn,
    "Published, but the price already covers the probability. Shown rather than hidden: a pick you can see and reject is information, a pick that silently vanished is not."
  ));
  if ((O.unpriced || []).length) host.appendChild(listPanel(
    "No price found", plural(O.unpriced.length, "pick with no market", "picks with no market"),
    O.unpriced,
    "No moneyline reached the build for these, from either SportsBookReview or ESPN core odds. Without a price there is no expected value and no stake size, so they are never comparable with a priced return."
  ));

  host.appendChild(suggestions("board"));
}

/* The wager to place. Falls back to the moneyline winner when nothing on the
   game is priced, which is the only case where there is no bet to name. */
function betHeadline(play) {
  if (play.betLabel && play.betMarket && play.betMarket !== "moneyline") return play.betLabel;
  return play.pick || "";
}

function heroFor(play) {
  const wrap = el("div");
  const hero = el("div", "hero");
  const fades = fadesMarket(play);
  const gap = play.marketPct == null ? null : play.confidence - play.marketPct;
  const when = startTime(play.startDate, play.league);

  const left = el("div", "cell");
  left.innerHTML =
    `<div class="eyebrow">` +
      `<span class="chip mdl">${esc(leagueShort(play))}</span>` +
      (fades == null ? `<span class="chip warn">Unpriced</span>`
        : `<span class="chip ${fades ? "warn" : "good"}">${fades ? "Against the favourite" : "With the favourite"}</span>`) +
      (gap == null ? "" : `<span class="chip">${sgn(gap)}pts vs market</span>`) +
      (when ? `<span class="chip">${esc(when)}</span>` : "") +
    `</div>` +
    `<div class="match">${esc(play.matchup || "")}</div>` +
    // The headline is the bet, not the winner. Once the board ranks on the
    // best available market, printing the moneyline team above a total's EV
    // and price would attribute one market's edge to another.
    `<h3 class="pick">${esc(betHeadline(play))}</h3>` +
    (play.betMarket && play.betMarket !== "moneyline"
      ? `<div style="font-size:12.5px;color:var(--muted);margin:-4px 0 8px">` +
        `${esc(MARKET_LABEL[play.betMarket] || play.betMarket)} — the best-priced bet on this game. ` +
        `Model likes ${esc(play.pick || "")} to win at ${pct(play.confidence)}.</div>`
      : "") +
    `<div class="rail2">` +
      `<div class="barrow"><span class="lbl">Model</span><div class="bar m">` +
        `<i style="width:${Math.max(0, Math.min(100, play.confidence || 0))}%"></i>` +
        `<span>${pct(play.confidence)}</span></div></div>` +
      `<div class="barrow"><span class="lbl">Market</span><div class="bar k">` +
        `<i style="width:${play.marketPct == null ? 0 : Math.max(0, Math.min(100, play.marketPct))}%"></i>` +
        `<span>${pct(play.marketPct)}</span></div></div>` +
    `</div>`;

  const right = el("div", "cell sunk");
  right.innerHTML = play.evPct == null
    ? `<div class="lbl">Expected value</div><span class="big" style="color:var(--muted)">n/a</span>` +
      `<div style="margin-top:9px;font-size:12.5px;color:var(--muted)">No price for this game, so there is no edge to measure and no stake to size. The confidence still stands on its own.</div>`
    : `<div class="lbl">Expected value</div><span class="big">${sgn(play.evPct)}<span class="pc">%</span></span>` +
      `<div style="margin-top:9px;font-size:12.5px;color:var(--muted)">per unit staked at ${american(play.odds)}` +
      (play.breakEvenPct == null ? "" : `, against a break-even of ${pct(play.breakEvenPct)}`) + `</div>`;

  hero.appendChild(left);
  hero.appendChild(right);
  wrap.appendChild(hero);

  // Stake sizing defers to the calibration band once it has enough graded
  // picks to trust (see kelly_band_probability) -- when that overrode the
  // model's own number, kellyProbabilityPct no longer matches confidence.
  const bandSized = play.kellyProbabilityPct != null &&
    Math.abs(play.kellyProbabilityPct - play.confidence) >= 0.5;

  const meta = el("div", "metagrid");
  meta.innerHTML =
    /* Shown as a percentage of bankroll, which is literally what kellyPct is.
       "Units" means two different things in this codebase -- the tracker's
       roiPct divides units by the pick count, implying a flat one-unit stake,
       while build_overview's suggestedUnits divides kellyPct by 100. Rendering
       a bankroll fraction as "0.05u" invited exactly that confusion. */
    `<div class="m"><div class="lbl">Stake, &frac14; Kelly</div><div class="v">` +
      `${play.kellyPct == null ? "—" : pct(play.kellyPct)}` +
      `<span style="color:var(--muted);font-size:13px"> of bankroll</span></div>` +
      (bandSized
        ? `<div style="font-size:11px;color:var(--warn);margin-top:2px">sized off the ${pct(play.kellyProbabilityPct)} band, not the ${pct(play.confidence)} headline` +
          (borrowedCalibrationNote(play.league) ? ` — and that band is borrowed from other leagues` : "") + `</div>`
        : "") +
      `</div>` +
    `<div class="m"><div class="lbl">Price</div><div class="v">${american(play.odds)}</div></div>` +
    `<div class="m"><div class="lbl">Calibration band</div><div class="v">${calibrationShort(play.confidence, play.league)}</div></div>`;
  wrap.appendChild(meta);
  return wrap;
}

function calibrationShort(confidence, league) {
  const b = bucketFor(confidence);
  if (!b) return "—";
  // Say whose record it is. Unlabelled, a borrowed band reads as this
  // league's own form, which for a league with no graded games is a number
  // about baseball wearing an NBA badge.
  const borrowed = league != null && borrowedCalibrationNote(league);
  const tag = borrowed ? "borrowed" : "actual";
  return pct(b.actualWinPct) +
    `<span style="font-size:12px;color:var(--muted)"> ${tag}</span>`;
}

function leagueTile(L) {
  const b = el("button", "tile");
  const empty = !L.gameCount;
  b.disabled = empty && !L.error;
  if (empty) {
    const why = L.error
      ? `The build could not reach the schedule feed for this league. This is a data failure, not an empty slate.`
      : `No games scheduled. Out of season — nothing is being withheld.`;
    b.innerHTML =
      `<div class="trow"><span class="tname">${esc(String(L.id).toUpperCase())}</span>` +
        (leagueSport(L) ? `<span class="tsub">${esc(leagueSport(L))}</span>` : "") +
        `<span class="go">&rarr;</span></div>` +
      (L.error ? `<div><span class="chip bad">Feed error</span></div>` : "") +
      `<div class="empty">${esc(why)}</div>`;
    if (L.error) b.addEventListener("click", () => { setSport(L.id); go("sport"); });
    return b;
  }
  const best = L.best;
  b.innerHTML =
    `<div class="trow"><span class="tname">${esc(String(L.id).toUpperCase())}</span>` +
      `<span class="tsub">${L.pickCount} of ${L.gameCount} published</span>` +
      `<span class="go">&rarr;</span></div>` +
    // "No prices" has two causes that look the same. Say which one, when the
    // build actually established it: a league no feed covers is permanently
    // unpriced, where a feed that failed today will be back.
    (L.pricedCount
      ? ""
      : L.priceCoverage?.noSourceFound
        ? `<div><span class="chip warn">No odds feed covers this league</span></div>`
        : `<div><span class="chip warn">No prices</span></div>`) +
    // A league with no graded record of its own is publishing picks whose
    // confidence has never been checked against that sport. Say so here
    // rather than only inside an expanded game card.
    (leagueGraded(L.id) === 0 ? `<div><span class="chip warn">No graded record yet</span></div>` : "") +
    (best
      ? `<div class="tbest"><b>${esc(best.pick || "")}</b><br>` +
        `<span style="color:var(--muted)">${esc(best.matchup || "")}</span></div>` +
        `<div class="tnums">` +
          `<div><div class="lbl">Conf</div><div class="v">${pct(best.confidence)}</div></div>` +
          `<div><div class="lbl">Edge</div><div class="v" style="color:${
            best.evPct == null ? "var(--muted)" : best.evPct > 0 ? "var(--good)" : "var(--bad)"
          }">${best.evPct == null ? "—" : sgn(best.evPct) + "%"}</div></div>` +
          `<div><div class="lbl">Price</div><div class="v">${american(best.odds)}</div></div>` +
        `</div>`
      : `<div class="empty">Every game on this slate came in under the publish bar.</div>`);
  b.addEventListener("click", () => { setSport(L.id); go("sport"); });
  return b;
}

function listPanel(title, note, plays, footer) {
  const p = el("div", "panel");
  p.appendChild(el("div", "phead", `<h2>${esc(title)}</h2><span class="n">${esc(note)}</span>`));
  const box = el("div");
  box.style.padding = "4px 16px 14px";
  plays.forEach((play) => {
    const row = el("div");
    row.style.cssText =
      "display:flex;gap:12px;align-items:baseline;padding:9px 0;border-bottom:1px solid var(--rule);font-size:13.5px;flex-wrap:wrap";
    row.innerHTML =
      `<span class="chip">${esc(leagueShort(play))}</span>` +
      `<span style="color:var(--muted)">${esc(play.matchup || "")}</span>` +
      `<b style="margin-left:auto">${esc(play.pick || "")}</b>` +
      `<span class="num">${pct(play.confidence)}</span>` +
      `<span class="num" style="color:${play.evPct == null ? "var(--muted)" : play.evPct > 0 ? "var(--good)" : "var(--bad)"}">` +
        `${play.evPct == null ? "no price" : sgn(play.evPct) + "%"}</span>`;
    box.appendChild(row);
  });
  box.appendChild(el("p", "", esc(footer)));
  box.lastChild.style.cssText = "font-size:12.5px;color:var(--muted);margin:11px 0 0";
  p.appendChild(box);
  return p;
}

/* ============================================================ VIEW: SPORT */
function setSport(id) { S.sport = id; }

let _sportRenderToken = 0;

async function renderSport() {
  const token = ++_sportRenderToken;
  const O = S.overview;
  const leagues = (O?.leagues || []).filter((L) => L.gameCount > 0);
  if (!S.sport) S.sport = (leagues[0] || (O?.leagues || [])[0] || {}).id || "mlb";
  const L = (O?.leagues || []).find((x) => x.id === S.sport) || { id: S.sport, label: S.sport };
  const date = selectedDateFor(S.sport);

  $("#sportTitle").textContent = String(L.id || "").toUpperCase();
  $("#sportWhen").textContent = [leagueSport(L), describeDate(date, S.sport)].filter(Boolean).join(" \u00b7 ");

  const seg = $("#sportSeg");
  seg.innerHTML = "";
  (O?.leagues || []).forEach((x) => {
    const b = el("button", null, esc(String(x.id).toUpperCase()));
    b.setAttribute("aria-pressed", String(x.id === S.sport));
    if (!x.gameCount) b.style.opacity = ".45";
    b.addEventListener("click", () => { S.sport = x.id; renderSport(); });
    seg.appendChild(b);
  });
  renderDateBar();

  const host = $("#games");
  host.innerHTML = `<div class="g"><div class="loading">Loading the ${esc(L.label || L.id)} slate for ${esc(describeDate(date, S.sport))}\u2026</div></div>`;

  const cacheKey = `${S.sport}:${date}`;
  let payload = S.slates[cacheKey];
  if (!payload) {
    const dateFile = manifestLeague(S.sport)?.dateFiles?.[date];
    const candidates = [
      dateFile,
      date ? `data/${S.sport}_${date}.json` : null,
      date && date === todayFor(S.sport) ? `data/${S.sport}.json` : null,
    ].filter(Boolean);
    for (const path of candidates) {
      try { payload = await getJson(path); break; } catch { /* try the next */ }
    }
    if (payload) S.slates[cacheKey] = payload;
  }

  // A slower, superseded fetch must not clobber whatever the user has since
  // clicked to -- date and sport can both change again before this resolves.
  if (token !== _sportRenderToken) return;

  if (!payload) {
    host.innerHTML = `<div class="g"><div class="failed">Could not load the ${esc(describeDate(date, S.sport))} slate for ${esc(L.label || L.id)}.</div></div>`;
    readings($("#sportReadings"), [
      { v: "\u2014", k: "Games" }, { v: "\u2014", k: "Published" }, { v: "\u2014", k: "Priced" }, { v: "\u2014", k: "Positive edge" },
    ]);
    return;
  }

  let games = (payload.games || []).slice();
  // Computed from the loaded slate itself, not the overview's per-league
  // summary -- that summary describes only today, so reusing it once the
  // date changes would show today's published/priced counts beside
  // tomorrow's games.
  const publishedCount = games.filter((g) => isPublished(g.prediction || {}, S.sport)).length;
  const pricedCount = games.filter((g) => ((g.prediction || {}).value || {}).evPct != null).length;
  readings($("#sportReadings"), [
    { v: games.length, k: "Games" },
    { v: publishedCount, k: "Published" },
    { v: pricedCount, k: "Priced" },
    { v: games.filter((g) => ((g.prediction || {}).value || {}).evPct > 0).length, k: "Positive edge" },
  ]);

  const rows = games.map((g) => toPlay(g, S.sport, L.label));
  const shown = S.filter === "pub" ? rows.filter((r) => r.published) : rows;
  shown.sort((a, b) => {
    if (S.sort === "conf") return (b.confidence || 0) - (a.confidence || 0);
    if (S.sort === "time") return String(a.startDate || "").localeCompare(String(b.startDate || ""));
    if (S.sort === "gap") {
      const ga = a.marketPct == null ? -1 : Math.abs(a.confidence - a.marketPct);
      const gb = b.marketPct == null ? -1 : Math.abs(b.confidence - b.marketPct);
      return gb - ga;
    }
    return (b.evPct == null ? -99 : b.evPct) - (a.evPct == null ? -99 : a.evPct);
  });

  host.innerHTML = "";
  if (!shown.length) {
    host.appendChild(el("div", "g",
      `<div class="loading">Nothing published for this sport on this slate.</div>`));
  }
  shown.forEach((play) => host.appendChild(gameRow(play)));
}

/* One game payload -> the flat shape every view reads. */
function toPlay(game, league, label) {
  const prediction = game.prediction || {};
  const value = prediction.value || {};
  const probabilities = prediction.probabilities || {};
  const consensus = (probabilities.implied || {}).consensus || {};
  const side = prediction.predictedSide;
  const marketPct =
    side === "home" ? consensus.homePct : side === "away" ? consensus.awayPct : consensus.drawPct;
  return {
    league,
    leagueLabel: label,
    eventId: game.eventId,
    matchup: game.matchup,
    homeTeam: game.homeTeam,
    awayTeam: game.awayTeam,
    startDate: game.startDate,
    pick: prediction.predictedWinner,
    pickSide: side,
    confidence: prediction.confidence,
    marketPct: marketPct == null ? null : marketPct,
    evPct: value.evPct == null ? null : value.evPct,
    kellyPct: value.kellyPct,
    // Equal to modelPct unless a reliable calibration band overrode the
    // stake sizing -- see kelly_band_probability in mlb_predictions.py.
    kellyProbabilityPct: value.kellyProbabilityPct,
    odds: value.odds,
    breakEvenPct: value.breakEvenPct,
    published: isPublished(prediction, league),
    features: prediction.features || {},
    enrichment: game.enrichment || {},
    total: prediction.total || null,
    spread: prediction.spread || null,
    // Every priced market ranked, and which one is worth backing.
    bestBet: prediction.bestBet || null,
    betMarket: (prediction.bestBet?.pick || {}).market || null,
    betLabel: (prediction.bestBet?.pick || {}).pick || null,
    statusLabel: game.statusLabel || (game.isFinal ? "Final" : null),
    isFinal: !!game.isFinal,
  };
}

/* Mirrors calibration_params.is_publishable_pick. A pick below the bar is
   withheld from the board but still logged, so the two must agree or the site
   shows picks the record does not contain. */
const MIN_PUBLISHABLE_CONFIDENCE = 55;
function isPublished(prediction, league) {
  if (!prediction) return false;
  if (prediction.published != null) return !!prediction.published;
  const c = prediction.confidence;
  return c != null && c >= MIN_PUBLISHABLE_CONFIDENCE;
}

function gameRow(play) {
  const row = el("div", "g");
  row.dataset.open = "0";
  const gap = play.marketPct == null ? null : play.confidence - play.marketPct;
  const fades = fadesMarket(play);
  const when = startTime(play.startDate, play.league);
  const result = resultFor(play.eventId);
  const score = finalScore(result);
  // The headline verdict follows the bet that was actually recommended, not
  // always the moneyline -- otherwise a card headlining a total would report
  // won or lost against a different market than the one it told you to back.
  const verdict = marketOutcome(result, play.betMarket || "moneyline");

  const head = el("button", "ghead");
  head.innerHTML =
    `<div class="mu"><div class="t">${esc(play.matchup || "")}</div><div class="s">` +
      (when ? `<span class="tm">${esc(when)}</span>` : "") +
      (play.statusLabel ? `<span class="chip ${play.isFinal ? "final" : "live"}">${esc(play.statusLabel)}</span>` : "") +
      // A played game should say what happened, not just that it happened.
      (score ? `<span class="chip num">${esc(score.text)}</span>` : "") +
      (verdict ? `<span class="chip ${OUTCOME_TONE[verdict]}">${OUTCOME_LABEL[verdict]}</span>` : "") +
      (play.published ? "" : `<span class="chip">Withheld</span>`) +
      (play.odds == null ? `<span class="chip warn">No price</span>` : "") +
      (fades === true ? `<span class="chip warn">Against the favourite</span>` : "") +
      (gap == null ? "" : `<span class="chip">${sgn(gap)}pts vs market</span>`) +
    `</div></div>` +
    `<div class="pk"><span class="who">${esc(betHeadline(play))}</span>` +
      (play.betMarket && play.betMarket !== "moneyline"
        ? `<span class="chip" style="margin-left:6px">${esc(MARKET_LABEL[play.betMarket] || play.betMarket)}</span>`
        : "") +
      `<div class="rail2" style="margin-top:6px">` +
        `<div class="bar m" style="height:9px"><i style="width:${Math.max(0, Math.min(100, play.confidence || 0))}%"></i></div>` +
        `<div class="bar k" style="height:9px"><i style="width:${play.marketPct == null ? 0 : Math.max(0, Math.min(100, play.marketPct))}%"></i></div>` +
      `</div></div>` +
    `<div class="cf">${pct(play.confidence)}</div>` +
    `<div class="evc" style="color:${play.evPct == null ? "var(--muted)" : play.evPct > 0 ? "var(--good)" : "var(--bad)"}">` +
      `${play.evPct == null ? "—" : sgn(play.evPct) + "%"}</div>` +
    `<div class="caret">&#9656;</div>`;
  head.addEventListener("click", () => {
    row.dataset.open = row.dataset.open === "1" ? "0" : "1";
  });
  row.appendChild(head);
  row.appendChild(whyPanel(play));
  return row;
}

const MARKET_LABEL = { moneyline: "Moneyline", total: "Total", spread: "Spread / runline" };

/* What actually happened, for a game that has been played.
 *
 * accuracy.json has carried the final score and the graded outcome of all
 * three markets since side markets were scored, keyed by the same eventId the
 * board already has -- and nothing on the board read it. A finished game said
 * "Final" and stopped, which is the least interesting true thing about it.
 */
function resultFor(eventId) {
  if (eventId == null) return null;
  const row = (S.accuracy?.picksByEventId || {})[String(eventId)];
  return row && row.status === "graded" ? row : null;
}

/* Scores arrive as strings from the scoreboard feed, so coerce rather than
   trusting them, and say nothing at all when either side is unreadable. */
function finalScore(result) {
  const home = Number(result?.homeScore);
  const away = Number(result?.awayScore);
  if (!Number.isFinite(home) || !Number.isFinite(away)) return null;
  // Away first, matching the "Away @ Home" order of every matchup label.
  return { away, home, text: `${away}–${home}` };
}

/* Won, lost or pushed, per market. A push is neither: the stake comes back,
   and calling it a win would flatter the record. */
function marketOutcome(result, market) {
  if (!result) return null;
  if (market === "moneyline") {
    return result.correct == null ? null : (result.correct ? "win" : "loss");
  }
  const block = market === "total" ? result.totalResult : result.spreadResult;
  return block?.outcome || null;
}

const OUTCOME_LABEL = { win: "Won", loss: "Lost", push: "Push" };
const OUTCOME_TONE = { win: "good", loss: "bad", push: "" };

/* Which of the three markets to actually back.
 *
 * The card used to name one bet -- the moneyline -- and print the total and
 * spread beside it as inert text with no pick, price or edge, so "is the
 * moneyline even the best bet here" had no answer on the page. All three are
 * now priced through the same assess_price, because percentage points of edge
 * are no more comparable across markets than across prices.
 *
 * The winner is gated, not a bare argmax: the moneyline is fitted and
 * calibrated against every graded game, while the side markets are heuristics
 * that have only recently started carrying prices at all. An unvalidated
 * market still shows its edge -- hiding it would be its own dishonesty -- but
 * it cannot take the headline on the strength of a number nothing has checked.
 */
function marketsPanel(play) {
  const best = play.bestBet;
  const options = best?.options || [];
  if (!options.length) {
    // No priced market at all. Fall back to naming whatever picks exist, so a
    // game with a total but no odds still shows it.
    const bare = [["total", play.total], ["spread", play.spread]]
      .filter(([, m]) => m)
      .map(([k, m]) => `<div class="k"><div class="kv">${esc(m.pick || m.pickSide || "—")}</div>` +
        `<div class="kk lbl">${MARKET_LABEL[k]}</div></div>`).join("");
    if (!bare) return null;
    const p = el("div", "");
    p.innerHTML = `<div class="subh" style="margin:15px 0 8px"><h4>Other markets</h4>` +
      `<span class="note">no price reached the build, so no edge to compare</span></div>` +
      `<div class="ctx">${bare}</div>`;
    return p;
  }

  const headline = best.pick;
  // Once a game is played the table can settle itself: every market it ranked
  // has a graded outcome sitting in accuracy.json, so the panel that said
  // which bet to back can also say whether backing it worked.
  const result = resultFor(play.eventId);
  const rows = options.map((o) => {
    const isBest = headline && o.market === headline.market;
    const good = o.evPct > 0;
    const outcome = marketOutcome(result, o.market);
    return `<tr style="${isBest ? "background:color-mix(in srgb, var(--good) 9%, transparent)" : ""}">` +
      `<td style="padding:7px 9px;white-space:nowrap">` +
        (isBest ? `<span class="chip good" style="margin-right:6px">Back this</span>` : "") +
        `<b>${esc(MARKET_LABEL[o.market] || o.market)}</b>` +
        (o.validated ? "" : `<span class="chip warn" style="margin-left:6px">unproven</span>`) +
      `</td>` +
      // board.css sets `th, td { text-align:right }` with only :first-child
      // left, so this needs saying explicitly or the column's values sit
      // right-aligned under a left-aligned heading.
      `<td style="padding:7px 9px;text-align:left">${esc(o.pick || "—")}</td>` +
      `<td class="num" style="padding:7px 9px;text-align:right">${american(o.odds)}</td>` +
      `<td class="num" style="padding:7px 9px;text-align:right">${pct(o.confidence)}</td>` +
      `<td class="num" style="padding:7px 9px;text-align:right;color:${good ? "var(--good)" : "var(--bad)"}">` +
        `${sgn(o.evPct)}%</td>` +
      (result
        ? `<td style="padding:7px 9px;text-align:right">` +
          (outcome
            ? `<span class="chip ${OUTCOME_TONE[outcome]}">${OUTCOME_LABEL[outcome]}</span>`
            : `<span style="color:var(--muted)">—</span>`) +
          `</td>`
        : "") +
      `</tr>`;
  }).join("");

  const held = best.heldBack;
  const notes = [];
  if (!headline) {
    notes.push("No market on this game clears zero expected value, so there is nothing here worth backing. " +
      "That is a decision, not a gap.");
  } else if (held) {
    notes.push(`${MARKET_LABEL[held.market]} shows the bigger edge at ${sgn(held.evPct)}%, but it has only ` +
      `${plural(held.gradedPriced ?? 0, "priced graded pick", "priced graded picks")} behind it — not enough to ` +
      `back ahead of the moneyline, which is fitted and calibrated against every graded game. It is shown ` +
      `because the edge is real data, not because it is a recommendation.`);
  } else {
    notes.push("All priced markets on this game, ranked by expected value per unit staked — the only figure " +
      "comparable across different prices.");
  }
  // A recommended side market has to carry its error bar. Its record is a few
  // dozen picks, where a hit rate moves several points on noise alone -- the
  // spread record fell from 67.9% to 58.7% inside a single afternoon's
  // grading. Printing the rate without the interval states a settled fact
  // that the sample cannot support.
  const caveat = headline && recordCaveat(headline);
  if (caveat) notes.push(caveat);

  const p = el("div", "");
  p.innerHTML =
    `<div class="subh" style="margin:15px 0 8px"><h4>Which bet</h4>` +
      `<span class="note">${result && finalScore(result)
        ? `final ${esc(finalScore(result).text)} — settled below`
        : "ranked by edge, gated by record"}</span></div>` +
    `<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12.5px">` +
      `<thead><tr style="color:var(--muted);font-size:11px;letter-spacing:.06em">` +
        `<th style="text-align:left;padding:0 9px 5px">MARKET</th>` +
        `<th style="text-align:left;padding:0 9px 5px">PICK</th>` +
        `<th style="text-align:right;padding:0 9px 5px">PRICE</th>` +
        `<th style="text-align:right;padding:0 9px 5px">MODEL</th>` +
        `<th style="text-align:right;padding:0 9px 5px">EDGE</th>` +
        (result ? `<th style="text-align:right;padding:0 9px 5px">RESULT</th>` : "") +
      `</tr></thead><tbody>${rows}</tbody></table></div>` +
    notes.map((n, i) => `<p style="font-size:12px;color:${i && /not yet/.test(n) ? "var(--warn)" : "var(--muted)"};margin:10px 0 0">${esc(n)}</p>`).join("");
  return p;
}

/* How thin the evidence behind a recommended side market actually is.
   Null for the moneyline, which is fitted and calibrated, and for any market
   with no graded record to describe. */
function recordCaveat(option) {
  const r = option?.record;
  if (!r || r.pct == null) return null;
  const label = MARKET_LABEL[option.market] || option.market;
  const err = r.stdErrPct == null ? "" : ` ±${r.stdErrPct}`;
  const base = `${label} has hit ${pct(r.pct)}${err} across ${plural(r.decided ?? 0, "decided pick", "decided picks")}, ` +
    `against a ${pct(r.breakEvenPct)} break-even at the prices actually taken.`;
  return r.beatsBreakEven
    ? `${base} That clears break-even by more than the sample's own error.`
    : `${base} That is not yet distinguishable from break-even — the edge above is the model's estimate, ` +
      `not a demonstrated return.`;
}

function whyPanel(play) {
  const body = el("div", "gbody");
  const why = el("div", "why");
  const d = decompose(play.features, play.league);

  /* ---- left: what actually decided it ---- */
  const left = el("div");
  left.innerHTML = `<div class="subh"><h4>What decided it</h4><span class="note">log-odds, home side positive</span></div>`;
  if (!d) {
    left.appendChild(el("div", "loading",
      "The fitted weights or this game's features are unavailable, so the decomposition cannot be shown. It is not being approximated."));
  } else {
    const terms = [
      { nm: "Team strength", v: d.contribStrength, has: d.hasStrength,
        why: "record, home/road split and power rating averaged into one input, because they measure the same thing" },
      { nm: "Market anchor", v: d.contribMarket, has: d.hasMarket,
        why: "the de-vigged price, in log-odds" },
      { nm: "League + home", v: d.base, has: true,
        why: "home-field advantage, carried exactly once" },
    ];
    const max = Math.max(0.25, ...terms.map((t) => Math.abs(t.v || 0)));
    const list = el("div", "terms");
    terms.forEach((t) => {
      const w = (Math.abs(t.v || 0) / max) * 50;
      const r = el("div", "term");
      r.innerHTML =
        `<span class="nm">${t.nm}</span>` +
        `<span class="track"><span class="mid"></span>` +
          (t.has ? `<i class="${(t.v || 0) >= 0 ? "pos" : "neg"}" style="${
            (t.v || 0) >= 0 ? `left:50%;width:${w}%` : `right:50%;width:${w}%`
          }"></i>` : "") +
        `</span>` +
        `<span class="val">${t.has ? sgn(t.v, 2) : "no data"}</span>`;
      const track = r.querySelector(".track");
      track.addEventListener("mousemove", (e) =>
        showTip(`<b>${t.nm}</b><div style="font-size:11.5px;color:var(--muted)">${t.why}</div>` +
          (t.has ? `<div class="r"><span>contribution</span><span>${sgn(t.v, 3)}</span></div>` : ""), e));
      track.addEventListener("mouseleave", hideTip);
      list.appendChild(r);
    });
    left.appendChild(list);
    left.appendChild(el("div", "termaxis",
      `<span></span><span class="ends">` +
      `<span class="lbl">&larr; favours ${esc(teamShort(play.awayTeam))}</span>` +
      `<span class="lbl">favours ${esc(teamShort(play.homeTeam))} &rarr;</span></span><span></span>`));
    const pPick = play.pickSide === "home" ? d.homePct : 100 - d.homePct;
    left.appendChild(el("p", "", ""));
    left.lastChild.style.cssText = "font-size:12px;color:var(--muted);margin:11px 0 0";
    left.lastChild.innerHTML =
      `These three are the whole decision. They sum to ${sgn(d.z, 3)} log-odds, which is ` +
      `<b class="num" style="color:var(--ink)">${pct(pPick)} ${esc(teamShort(play.pick))}</b> ` +
      `(${pct(d.homePct)} to the home side). Everything on the right is context the model ` +
      `either folded into team strength already, or is still measuring.`;
  }

  /* ---- calibration check: the record, not a forecast ---- */
  const b = bucketFor(play.confidence);
  if (b) {
    const thin = b.picks < 30;
    const bad = Math.abs(b.overconfidencePct) > 8 && !thin;
    const borrowed = borrowedCalibrationNote(play.league);
    left.appendChild(el("div", "",
      `<div class="subh" style="margin:17px 0 8px"><h4>Calibration check</h4>` +
        `<span class="note">not a forecast — the record</span></div>` +
      `<div style="border:1px solid ${bad ? "color-mix(in srgb, var(--warn) 45%, var(--rule))" : "var(--rule)"};` +
        `border-radius:3px;padding:11px 12px;background:var(--panel)">` +
        `<div style="font-size:12.5px">Games this model rated <b class="num">${esc(b.range)}%</b> have won ` +
          `<b class="num" style="color:${bad ? "var(--warn)" : "var(--ink)"}">${pct(b.actualWinPct)}</b> ` +
          `of the time (n=${b.picks}, &plusmn;${b.stdErrPct}).</div>` +
        (bad
          ? `<div style="font-size:12px;color:var(--muted);margin-top:6px">That is ${sgn(b.overconfidencePct)} points of ` +
            `overconfidence on a sample big enough to believe. Discount this number before you stake on it.</div>`
          : thin
            ? `<div style="font-size:12px;color:var(--muted);margin-top:6px">Too few picks in this band to conclude much.</div>`
            : "") +
        (borrowed
          ? `<div style="font-size:12px;color:var(--warn);margin-top:6px">${esc(borrowed)}</div>`
          : "") +
      `</div>`));
  }

  /* ---- right: context, plus the side markets ---- */
  const right = el("div");
  right.innerHTML = `<div class="subh"><h4>Context</h4><span class="note">known, not yet trusted as its own input</span></div>`;
  const f = play.features || {};
  /* Number(null) is 0 and Number("") is 0, so a feature the model logged as
     absent was arriving here as a real zero -- "no rating gap", "no travel",
     "no injuries" -- rather than as missing. Python writes None, JSON carries
     null, and every one of those became a confident 0 on the card. */
  const n = (v) => {
    if (v === null || v === undefined || v === "") return null;
    const x = Number(v);
    return Number.isFinite(x) ? x : null;
  };
  /* Every one of these was a bare signed number against a jargon label, which
     is unreadable unless you already know the sign convention AND the sport.
     "Head to head -1.00" tells you nothing; "Boston has won every meeting"
     tells you the same thing. So each tile now carries a unit and a sentence
     naming the side it favours, rather than expecting the reader to decode a
     minus sign. */
  const HOME = teamShort(play.homeTeam) || "the home side";
  const AWAY = teamShort(play.awayTeam) || "the visitors";
  const favours = (v) => (v > 0 ? HOME : AWAY);
  /* Four of these are fed by baseball-only tables: PARK_FACTORS and TEAM_HOME
     hold 30 MLB clubs each, bullpen workload comes from the MLB pitching
     pipeline, and handedness is gated on league === "mlb" outright. On an NBA
     or NFL card they can never resolve, and rendering them as "no data" says a
     feed is broken when nothing is. Half a card of false alarms, on every game,
     once those seasons start.

     This tracks which tables exist today, not a fact about the sports. Fill in
     TEAM_HOME for the other leagues and travel stops being baseball-only. */
  /* A tile carries `leagues` when its underlying table only covers some of
     them, and is hidden where it could never resolve. This was a boolean
     `baseballOnly` until the travel table grew to cover basketball and
     football -- at which point a single flag could no longer say the truth,
     because travel resolves for four leagues and the park factors for one.
     Leaving it a boolean would have put an empty "Travel burden" tile back on
     AFL and EPL cards, which is the exact false alarm the flag exists to
     prevent. Absent means the feature is available everywhere. */
  const league = String(play.league || "").toLowerCase();
  const covers = (i) => !i.leagues || i.leagues.includes(league);
  const items = [
    {
      kk: "Team rating gap",
      kv: n(f.eloEdge) == null ? null : `${Math.abs(n(f.eloEdge)).toFixed(0)} pts`,
      hint: n(f.eloEdge) == null ? null
        : n(f.eloEdge) === 0 ? "Both clubs rated level."
        : `${favours(n(f.eloEdge))} rated higher, on form to date. Home advantage is not in this number.`,
    },
    {
      kk: "Ballpark scoring",
      leagues: ["mlb"],
      kv: n(f.parkEdge) == null ? null : `${sgn(n(f.parkEdge), 0)}%`,
      /* The only tile here that is not about either team. Worth saying so --
         a reader reasonably assumes every number in this grid picks a side. */
      hint: n(f.parkEdge) == null ? null
        : n(f.parkEdge) >= 8 ? "This ground yields well above average runs. Favours neither team — it lifts the total."
        : n(f.parkEdge) >= 3 ? "Slightly high-scoring ground. Affects the total, not the winner."
        : n(f.parkEdge) <= -3 ? "Low-scoring ground. Affects the total, not the winner."
        : "An average ground for scoring.",
    },
    {
      kk: "Bullpen freshness",
      leagues: ["mlb"],
      kv: n(f.bullpenDiff) == null ? null : `${sgn(n(f.bullpenDiff), 1)} inn`,
      hint: n(f.bullpenDiff) == null ? null
        : n(f.bullpenDiff) === 0 ? "Both relief corps equally worked."
        : `${favours(n(f.bullpenDiff))} has the fresher relief pitchers — the other side's have thrown more lately.`,
    },
    {
      kk: "This season's meetings",
      /* Shown as the home side's share of meetings won rather than the signed
         difference the model stores. The two shares sum to 1, so
         share = (diff + 1) / 2 loses nothing -- and "-1.00" was exactly the
         number that read as meaningless. */
      kv: n(f.h2hDiff) == null ? null
        : `${HOME} ${Math.round(((n(f.h2hDiff) + 1) / 2) * 100)}%`,
      /* Phrased about the same side the value names, or the tile reads as two
         different facts. */
      hint: n(f.h2hDiff) == null ? null
        : n(f.h2hDiff) === 0 ? `${HOME} and ${AWAY} have split their meetings this season.`
        : n(f.h2hDiff) <= -1 ? `${HOME} has lost every meeting with ${AWAY} this season.`
        : n(f.h2hDiff) >= 1 ? `${HOME} has won every meeting with ${AWAY} this season.`
        : `Share of this season's meetings ${HOME} has won.`,
    },
    {
      // Not baseball-only any more: TEAM_HOME covers MLB, NBA, NFL and the
      // WNBA, so this resolves for all four. A league still outside the table
      // renders it empty like any other missing feature.
      kk: "Travel burden",
      leagues: ["mlb", "nba", "nfl", "wnba"],
      kv: n(f.travelDiff) == null ? null : sgn(n(f.travelDiff), 2),
      hint: n(f.travelDiff) == null ? null
        : n(f.travelDiff) === 0 ? "No meaningful trip for the visitors."
        : `${AWAY} travelled — distance plus time-zone change. Higher means a harder trip.`,
    },
    {
      kk: "Left-handed starter",
      leagues: ["mlb"],
      kv: n(f.handednessDiff) == null ? null
        : n(f.handednessDiff) === 0 ? "neither" : favours(n(f.handednessDiff)),
      /* 0 here genuinely means "both or neither", not missing data, and the
         bare 0 read as an absent value. */
      hint: n(f.handednessDiff) == null ? null
        : n(f.handednessDiff) === 0 ? "Both starters throw the same hand, or neither is a lefty."
        : `Only ${favours(n(f.handednessDiff))} starts a left-hander, which is the rarer matchup.`,
    },
    {
      kk: "Injuries out",
      kv: n(f.homeInjuryLoad) == null && n(f.awayInjuryLoad) == null
        ? null : `${n(f.homeInjuryLoad) ?? "—"} / ${n(f.awayInjuryLoad) ?? "—"}`,
      hint: n(f.homeInjuryLoad) == null && n(f.awayInjuryLoad) == null ? null
        : `${HOME} / ${AWAY}. Weighted by how important the missing players are; lower is healthier.`,
    },
    {
      kk: "Days off",
      kv: n(f.homeRest) == null && n(f.awayRest) == null
        ? null : `${n(f.homeRest) ?? "—"} / ${n(f.awayRest) ?? "—"}`,
      hint: n(f.homeRest) == null && n(f.awayRest) == null ? null
        : `${HOME} / ${AWAY}. Rest since each side last played; 0 means they played yesterday.`,
    },
  ];
  /* Dropped rather than greyed out: an empty tile still reads as a gap in the
     data. The note below says what is missing and why, once, instead of four
     times per card. */
  const hidden = items.filter((i) => !covers(i));
  const shown = items.filter(covers);
  const ctx = el("div", "ctx");
  shown.forEach((i) => {
    const k = el("div", "k" + (i.kv == null ? " off" : ""));
    k.innerHTML =
      `<div class="kv">${i.kv == null ? "no data" : esc(String(i.kv))}</div>` +
      `<div class="kk lbl">${esc(i.kk)}</div>` +
      (i.kv == null
        ? `<div class="khint">Not available for this game.</div>`
        : i.hint ? `<div class="khint">${esc(i.hint)}</div>` : "");
    ctx.appendChild(k);
  });
  right.appendChild(ctx);
  if (hidden.length) {
    /* Names the sport rather than saying "baseball only", which stopped being
       true once travel covered four leagues and the park factors one. */
    right.appendChild(el("div", "", `<div class="khint" style="margin-top:7px">` +
      `${hidden.map((i) => esc(i.kk.toLowerCase())).join(", ")} ` +
      `${hidden.length === 1 ? "is" : "are"} not tracked for ` +
      `${esc(league.toUpperCase() || "this sport")}, so ` +
      `${hidden.length === 1 ? "it is" : "they are"} not shown here.</div>`));
  }

  const p = f.mlbPitching;
  if (p) {
    right.appendChild(el("div", "",
      `<div class="subh" style="margin:15px 0 8px"><h4>Pitching</h4>` +
      `<span class="note">runs allowed per nine innings — lower is better</span></div>` +
      `<div class="ctx">` +
        `<div class="k"><div class="kv">${p.homePitcherApiEra ?? "—"} / ${p.homePitcherRecentEra ?? "—"}</div>` +
          `<div class="kk lbl">${esc(HOME)} starter</div>` +
          `<div class="khint">Season so far / last few outings. The starter is the single biggest per-game factor in baseball.</div></div>` +
        `<div class="k"><div class="kv">${p.awayPitcherApiEra ?? "—"} / ${p.awayPitcherRecentEra ?? "—"}</div>` +
          `<div class="kk lbl">${esc(AWAY)} starter</div>` +
          `<div class="khint">Season so far / last few outings.</div></div>` +
        `<div class="k"><div class="kv">${p.homeBullpenEra ?? "—"}</div>` +
          `<div class="kk lbl">${esc(HOME)} bullpen</div>` +
          `<div class="khint">The relief pitchers who finish the game once the starter comes out.</div></div>` +
        `<div class="k"><div class="kv">${p.awayBullpenEra ?? "—"}</div>` +
          `<div class="kk lbl">${esc(AWAY)} bullpen</div>` +
          `<div class="khint">Relief pitchers.</div></div>` +
      `</div>`));
  }


  const cov = f.dataCoverage || {};
  const covNames = {
    lineup: "Lineup", injuries: "Injuries", espnPredictor: "ESPN predictor",
    advancedStats: "Advanced stats", restData: "Rest", scheduleFlags: "Schedule",
    mlbPitching: "Pitching", impliedOdds: "Price",
  };
  const cb = el("div", "cov");
  cb.style.marginTop = "13px";
  Object.keys(covNames).forEach((k) => {
    if (!(k in cov)) return;
    cb.appendChild(el("span", "c" + (cov[k] ? " on" : ""), `<i></i>${covNames[k]}`));
  });
  if (cb.children.length) {
    right.appendChild(el("div", "", `<div class="subh" style="margin:15px 0 8px"><h4>Inputs present</h4></div>`));
    right.appendChild(cb);
  }

  why.appendChild(left);
  why.appendChild(right);
  body.appendChild(why);

  // Full width, below the two columns, rather than inside the narrow context
  // sidebar. It is the actual recommendation now, not sidebar trivia -- and a
  // five-column table in that column pushed the model and edge figures off the
  // right edge, hiding the two numbers the whole panel exists to show.
  const markets = marketsPanel(play);
  if (markets) body.appendChild(markets);
  return body;
}

/* ========================================================= VIEW: ACCURACY */
function renderAccuracy() {
  const A = S.accuracy;
  const E = S.evaluation;
  const host = $("#accBody");
  host.innerHTML = "";
  if (!A) {
    host.appendChild(el("div", "failed", "Could not load data/accuracy.json."));
    return;
  }
  const sum = A.summary || {};
  const all = sum.allTime || {};
  $("#accWhen").textContent = "graded through " +
    new Date(A.updatedAt || Date.now()).toLocaleDateString(undefined, { day: "numeric", month: "long" });

  const topForecaster = ((E?.overall || {}).forecasters || [])[0] || {};
  readings($("#accReadings"), [
    { v: all.total ?? "—", k: "Graded picks" },
    { v: pct(all.pct), k: "Hit rate" },
    { v: all.units == null ? "—" : sgn(all.units, 1) + "u", k: "Units" },
    { v: all.roiPct == null ? "—" : sgn(all.roiPct) + "%", k: "ROI" },
    { v: topForecaster.logLoss == null ? "—" : topForecaster.logLoss.toFixed(4), k: "Log loss" },
  ]);

  /* ---- verdict, written from the numbers rather than around them ---- */
  host.appendChild(verdictPanel(A, E));

  if (!E) {
    host.appendChild(el("div", "panel",
      `<div class="loading">data/evaluation.json is unavailable, so log loss, Brier, AUC, reliability and home bias cannot be shown. The record above still stands.</div>`));
  } else {
    host.appendChild(reliabilityPanel(E));
    host.appendChild(forecasterPanel(E));
  }
  host.appendChild(leaguePanel(sum.byLeague || {}));
  if (E) host.appendChild(chartsPanel(A, E));
  host.appendChild(cardsPanel(A, E));
  host.appendChild(suggestions("accuracy"));
}

function verdictPanel(A, E) {
  const p = el("div", "panel");
  p.appendChild(el("div", "phead",
    `<h2>Where this actually stands</h2><span class="n">read this before the charts</span>`));
  const overall = (E?.overall || {}).forecasters || [];
  const priced = (E?.vsMarket || {}).forecasters || [];
  const model = overall.find((f) => f.name === "model (published)");
  const constant = overall.find((f) => f.name.includes("constant"));
  const pricedModel = priced.find((f) => f.name === "model (published)");
  const pricedMarket = priced.find((f) => f.name.startsWith("market"));
  const rel = (E?.reliability || []).filter((b) => b.picks >= 30);
  const worst = rel.slice().sort((a, b) => b.overconfidencePct - a.overconfidencePct)[0];

  const bits = [];
  if (model && constant) {
    bits.push(`<p><b>Established:</b> the model beats every naive baseline — log loss ` +
      `${model.logLoss.toFixed(4)} against ${constant.logLoss.toFixed(4)} for a constant base rate, ` +
      `and it ranks winners above losers (AUC ${model.auc.toFixed(3)}).</p>`);
  }
  if (pricedModel && pricedMarket) {
    const marketWins = pricedMarket.logLoss < pricedModel.logLoss;
    bits.push(`<p><b>${marketWins ? "Not established" : "Also established"}:</b> ` +
      (marketWins
        ? `that it beats the market. On the ${pricedMarket.n} games where both have a price, the market scores a ` +
          `<b>better</b> log loss — ${pricedMarket.logLoss.toFixed(4)} against the model's ${pricedModel.logLoss.toFixed(4)}. ` +
          `Any advantage on the full set comes partly from games the market never priced.`
        : `on the ${pricedMarket.n} shared priced games the model now edges the market, ` +
          `${pricedModel.logLoss.toFixed(4)} against ${pricedMarket.logLoss.toFixed(4)}.`) + `</p>`);
  }
  if (worst && worst.overconfidencePct > 6) {
    bits.push(`<p><b>Known problem:</b> the ${worst.range}% band is ${worst.overconfidencePct.toFixed(1)} points ` +
      `overconfident across ${worst.picks} picks. On that sample it is not an artefact.</p>`);
  }
  const clv = (A.summary || {}).closingLineValue;
  if (clv && clv.picks) {
    bits.push(`<p><b>Watch:</b> closing line value is ${sgn(clv.avgPct, 2)}% over ${clv.picks} picks, ` +
      `beating the close ${pct(clv.beatCloseP)} of the time. CLV tracks long-run profit better than hit rate does.</p>`);
  }
  const nGraded = (A.summary?.allTime || {}).total;
  const nEval = (E?.overall || {}).n;
  if (nGraded && nEval && nGraded !== nEval) {
    bits.push(`<p style="color:var(--muted);font-size:12.5px"><b>On the two counts:</b> the strip above reads ` +
      `${nGraded} graded picks from the live tracker; the model-quality panels read n=${nEval}, because the ` +
      `evaluation report is rebuilt on its own cadence. Both are labelled with their own n rather than averaged ` +
      `into one number that is true of neither.</p>`);
  }
  const v = el("div", "verdict", `<span class="mark">!</span><div>${bits.join("")}</div>`);
  p.appendChild(v);
  return p;
}

function reliabilityPanel(E) {
  const p = el("div", "panel");
  p.appendChild(el("div", "phead",
    `<h2>Reliability</h2><span class="n">n=${(E.overall || {}).n ?? "—"}, all-time</span>` +
    `<span class="spacer"></span><span class="legend">` +
      `<span class="lbl"><i style="background:var(--model)"></i>Bucket, &plusmn;1 s.e.</span>` +
      `<span class="lbl"><i style="background:var(--muted)"></i>Under 30 picks</span></span>`));
  const box = el("div");
  box.style.padding = "18px 16px 16px";
  const fig = el("figure");
  const chart = el("div");
  fig.appendChild(chart);
  fig.appendChild(el("figcaption", "",
    "Each point is a confidence bucket: where it said it would land on the horizontal, where it actually landed on the vertical. On the dashed line the model is honest; below it, overconfident. Bars are one binomial standard error, and grey points hold too few picks to conclude from."));
  box.appendChild(fig);
  p.appendChild(box);
  relChart(chart, E.reliability || []);
  return p;
}

function relChart(host, rel) {
  if (!rel.length) { host.innerHTML = `<div class="loading">No graded buckets yet.</div>`; return; }
  const W = 660, H = 380, P = { l: 44, r: 16, t: 14, b: 40 };
  const x = (v) => P.l + ((v - 45) / 50) * (W - P.l - P.r);
  const y = (v) => H - P.b - ((v - 30) / 70) * (H - P.t - P.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Reliability curve"><g class="grid">`;
  for (let v = 50; v <= 90; v += 10) s += `<line x1="${x(v)}" y1="${P.t}" x2="${x(v)}" y2="${H - P.b}"/>`;
  for (let v = 40; v <= 100; v += 20) s += `<line x1="${P.l}" y1="${y(v)}" x2="${W - P.r}" y2="${y(v)}"/>`;
  s += `</g><line class="diag" x1="${x(45)}" y1="${y(45)}" x2="${x(95)}" y2="${y(95)}"/><g class="axis">`;
  for (let v = 50; v <= 90; v += 10) s += `<text x="${x(v)}" y="${H - P.b + 16}" text-anchor="middle">${v}%</text>`;
  for (let v = 40; v <= 100; v += 20) s += `<text x="${P.l - 8}" y="${y(v) + 3}" text-anchor="end">${v}%</text>`;
  s += `</g>`;
  s += `<text x="${(P.l + W - P.r) / 2}" y="${H - 4}" text-anchor="middle" font-size="11" fill="var(--muted)" font-family="var(--display)" letter-spacing=".08em">SAID IT WOULD WIN</text>`;
  s += `<text x="12" y="${(P.t + H - P.b) / 2}" text-anchor="middle" font-size="11" fill="var(--muted)" font-family="var(--display)" letter-spacing=".08em" transform="rotate(-90 12 ${(P.t + H - P.b) / 2})">ACTUALLY WON</text>`;
  rel.forEach((b, i) => {
    const thin = b.picks < 30;
    const cx = x(b.avgPredictedPct), cy = y(b.actualWinPct);
    const lo = y(Math.max(30, b.actualWinPct - b.stdErrPct));
    const hi = y(Math.min(100, b.actualWinPct + b.stdErrPct));
    const cls = `ebar${thin ? " thin" : ""}`;
    s += `<line class="${cls}" x1="${cx}" y1="${lo}" x2="${cx}" y2="${hi}"/>`;
    s += `<line class="${cls}" x1="${cx - 4}" y1="${lo}" x2="${cx + 4}" y2="${lo}"/>`;
    s += `<line class="${cls}" x1="${cx - 4}" y1="${hi}" x2="${cx + 4}" y2="${hi}"/>`;
    s += `<circle class="pt${thin ? " thin" : ""}" cx="${cx}" cy="${cy}" r="${Math.max(5, Math.min(11, Math.sqrt(b.picks)))}" data-i="${i}"/>`;
    if (!thin) s += `<text x="${cx}" y="${lo + 15}" text-anchor="middle" font-size="10.5" font-family="var(--mono)" fill="var(--muted)">${b.range}</text>`;
  });
  s += `</svg>`;
  host.innerHTML = s;
  host.querySelectorAll(".pt").forEach((c) => {
    const b = rel[+c.dataset.i];
    c.addEventListener("mousemove", (e) => showTip(
      `<b>${b.range}% bucket</b>` +
      `<div class="r"><span>picks</span><span>${b.picks}</span></div>` +
      `<div class="r"><span>said</span><span>${pct(b.avgPredictedPct)}</span></div>` +
      `<div class="r"><span>actual</span><span>${pct(b.actualWinPct)} &plusmn;${b.stdErrPct}</span></div>` +
      `<div class="r"><span>overconfident by</span><span>${sgn(b.overconfidencePct)}pts</span></div>` +
      (b.picks < 30 ? `<div style="font-size:11px;color:var(--muted);margin-top:4px">Under 30 picks — too thin to conclude from.</div>` : ""), e));
    c.addEventListener("mouseleave", hideTip);
  });
}

function forecasterPanel(E) {
  const p = el("div", "panel");
  p.appendChild(el("div", "phead",
    `<h2>Against the alternatives</h2><span class="n">n=${(E.overall || {}).n ?? "—"}</span>`));
  const rows = [];
  ((E.overall || {}).forecasters || []).forEach((f) => rows.push({ slate: "all graded", ...f }));
  ((E.vsMarket || {}).forecasters || [])
    .filter((f) => f.name === "model (published)" || f.name.startsWith("market"))
    .forEach((f) => rows.push({ slate: "priced only", ...f }));
  let html = `<thead><tr><th>Forecaster</th><th>Slate</th><th>n</th><th>Log loss</th><th>Brier</th><th>AUC</th><th>Accuracy</th></tr></thead><tbody>`;
  rows.forEach((f) => {
    const cls = f.name === "model (published)" ? "hi" : f.name.startsWith("baseline") ? "dim" : "";
    html += `<tr class="${cls}"><td>${esc(f.name)}</td><td style="color:var(--muted)">${f.slate}</td>` +
      `<td class="n">${f.n}</td>` +
      `<td class="n">${f.logLoss == null ? "—" : f.logLoss.toFixed(4)}</td>` +
      `<td class="n">${f.brier == null ? "—" : f.brier.toFixed(4)}</td>` +
      `<td class="n">${f.auc == null ? "—" : f.auc.toFixed(3)}</td>` +
      `<td class="n">${f.accuracy == null ? "—" : pct(f.accuracy * 100)}</td></tr>`;
  });
  html += `</tbody>`;
  const t = el("div", "tscroll");
  t.appendChild(el("table", "", html));
  p.appendChild(t);
  p.appendChild(el("div", "",
    "Lower log loss and Brier are better; both punish confident mistakes. AUC is rank quality, where 0.5 is noise. The market row covers only the games it priced, which is why the comparison is shown twice: once on its own slate, once on everything."));
  p.lastChild.style.cssText = "padding:12px 16px 15px;font-size:12.5px;color:var(--muted);border-top:1px solid var(--rule)";
  return p;
}

function leaguePanel(byLeague) {
  const p = el("div", "panel");
  p.appendChild(el("div", "phead",
    `<h2>By league</h2><span class="n">priced and unpriced kept apart</span>`));
  let html = `<thead><tr><th>League</th><th>Picks</th><th>Hit rate</th><th>Priced</th><th>Units</th><th>ROI</th></tr></thead><tbody>`;
  Object.entries(byLeague).forEach(([k, v]) => {
    const mostlyUnpriced = (v.pricedPct ?? 0) < 50;
    html += `<tr><td>${esc(k.toUpperCase())}` +
      (mostlyUnpriced ? ` <span class="chip warn" style="margin-left:6px">mostly unpriced</span>` : "") +
      `</td><td class="n">${v.total}</td><td class="n">${pct(v.pct)}</td>` +
      `<td class="n">${v.priced ?? "—"}/${v.total}</td>` +
      `<td class="n">${v.units == null ? "—" : sgn(v.units, 2)}</td>` +
      `<td class="n" style="color:${v.roiPct == null ? "var(--muted)" : v.roiPct > 0 ? "var(--good)" : "var(--bad)"}">` +
        `${v.roiPct == null ? "not measurable" : sgn(v.roiPct) + "%"}</td></tr>`;
  });
  html += `</tbody>`;
  const t = el("div", "tscroll");
  t.appendChild(el("table", "", html));
  p.appendChild(t);
  p.appendChild(el("div", "",
    "A hit rate without a price is not comparable to one with a price, so leagues whose slates are mostly unpriced are flagged. Stacking them in one column invites exactly that comparison."));
  p.lastChild.style.cssText = "padding:12px 16px 15px;font-size:12.5px;color:var(--muted);border-top:1px solid var(--rule)";
  return p;
}

function chartsPanel(A, E) {
  const wrap = el("div", "hero");
  wrap.style.cssText = "border:1px solid var(--rule);border-radius:3px;overflow:hidden";
  const equity = equitySeries(A);
  const left = el("div", "cell");
  left.innerHTML = `<div class="subh"><h4>Running units</h4><span class="note">last ${equity.length} graded</span></div>`;
  const eqHost = el("div");
  const eqFig = el("figure");
  eqFig.appendChild(eqHost);
  eqFig.appendChild(el("figcaption", "", "Quarter-Kelly units on priced picks only. A flat stretch is unpriced games, not a pause in the model."));
  left.appendChild(eqFig);

  const right = el("div", "cell sunk");
  right.innerHTML = `<div class="subh"><h4>Home bias</h4><span class="note">pick rate minus actual home win rate</span></div>`;
  const biasHost = el("div");
  const biasFig = el("figure");
  biasFig.appendChild(biasHost);
  biasFig.appendChild(el("figcaption", "", "Positive means the model backs the home side more often than the home side wins."));
  right.appendChild(biasFig);

  wrap.appendChild(left);
  wrap.appendChild(right);
  setTimeout(() => { eqChart(eqHost, equity); biasChart(biasHost, E.homeBias || {}); }, 0);
  return wrap;
}

function equitySeries(A) {
  const graded = (A.recentResults || [])
    .filter((r) => r.status === "graded" && r.units != null)
    .sort((a, b) => String(a.gradedAt || a.date || "").localeCompare(String(b.gradedAt || b.date || "")));
  let cum = 0;
  return graded.map((r) => {
    cum += r.units;
    return { date: r.date, league: r.league, units: Math.round(cum * 1000) / 1000, correct: r.correct };
  });
}

function eqChart(host, eq) {
  if (eq.length < 2) { host.innerHTML = `<div class="loading">Not enough graded priced picks yet.</div>`; return; }
  const W = 520, H = 210, P = { l: 38, r: 12, t: 12, b: 26 };
  const vals = eq.map((d) => d.units);
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  const x = (i) => P.l + (i / (eq.length - 1)) * (W - P.l - P.r);
  const y = (v) => H - P.b - ((v - lo) / ((hi - lo) || 1)) * (H - P.t - P.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Running units">` +
    `<g class="grid"><line x1="${P.l}" y1="${y(hi)}" x2="${W - P.r}" y2="${y(hi)}"/></g>` +
    `<line class="zero" x1="${P.l}" y1="${y(0)}" x2="${W - P.r}" y2="${y(0)}"/>`;
  let d = `M ${x(0)} ${y(eq[0].units)}`, a = `M ${x(0)} ${y(0)} L ${x(0)} ${y(eq[0].units)}`;
  eq.forEach((pt, i) => { if (i) { d += ` L ${x(i)} ${y(pt.units)}`; a += ` L ${x(i)} ${y(pt.units)}`; } });
  a += ` L ${x(eq.length - 1)} ${y(0)} Z`;
  const last = vals[vals.length - 1];
  s += `<path class="eqfill" d="${a}"/><path class="eq" d="${d}"/>` +
    `<circle cx="${x(eq.length - 1)}" cy="${y(last)}" r="4" fill="var(--model)" stroke="var(--panel)" stroke-width="2"/>` +
    `<text x="${x(eq.length - 1) - 6}" y="${y(last) - 10}" text-anchor="end" font-size="12" font-family="var(--mono)" fill="var(--ink)">${sgn(last, 2)}u</text>` +
    `<g class="axis"><text x="${P.l - 7}" y="${y(hi) + 3}" text-anchor="end">${hi.toFixed(1)}</text>` +
    `<text x="${P.l - 7}" y="${y(lo) + 3}" text-anchor="end">${lo.toFixed(1)}</text></g></svg>`;
  host.innerHTML = s;
  const svg = host.querySelector("svg");
  svg.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const i = Math.round((((e.clientX - r.left) / r.width) * W - P.l) / (W - P.l - P.r) * (eq.length - 1));
    const pt = eq[Math.max(0, Math.min(eq.length - 1, i))];
    if (!pt) return;
    showTip(`<b>${pt.date}</b><div class="r"><span>${String(pt.league).toUpperCase()}</span>` +
      `<span>${pt.correct ? "won" : "lost"}</span></div>` +
      `<div class="r"><span>running</span><span>${sgn(pt.units, 2)}u</span></div>`, e);
  });
  svg.addEventListener("mouseleave", hideTip);
}

function biasChart(host, bias) {
  const rows = Object.entries(bias).filter(([k]) => k !== "ALL").map(([k, v]) => ({ k, ...v }));
  if (!rows.length) { host.innerHTML = `<div class="loading">No per-league bias yet.</div>`; return; }
  rows.sort((a, b) => b.biasPct - a.biasPct);
  const W = 460, rh = 30, H = rows.length * rh + 26, P = { l: 62, r: 44 };
  const max = Math.max(6, ...rows.map((r) => Math.abs(r.biasPct)));
  const mid = P.l + (W - P.l - P.r) / 2;
  const scale = (v) => (v / max) * ((W - P.l - P.r) / 2);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Home bias by league">` +
    `<line class="zero" x1="${mid}" y1="6" x2="${mid}" y2="${rows.length * rh + 6}"/>`;
  rows.forEach((r, i) => {
    const yy = 6 + i * rh, w = Math.abs(scale(r.biasPct));
    s += `<rect class="hbar${r.biasPct < 0 ? " neg" : ""}" x="${r.biasPct >= 0 ? mid : mid - w}" y="${yy + 6}" width="${w}" height="14" rx="2" data-i="${i}"/>` +
      `<text x="${P.l - 8}" y="${yy + 17}" text-anchor="end" font-size="11.5" font-family="var(--display)" letter-spacing=".05em" fill="var(--ink2)">${esc(r.k.toUpperCase())}</text>` +
      `<text x="${r.biasPct >= 0 ? mid + w + 6 : mid - w - 6}" y="${yy + 17}" text-anchor="${r.biasPct >= 0 ? "start" : "end"}" font-size="11.5" font-family="var(--mono)" fill="var(--muted)">${sgn(r.biasPct)}</text>`;
  });
  s += `<text x="${mid}" y="${H - 3}" text-anchor="middle" font-size="10" font-family="var(--mono)" fill="var(--muted)">0 = honest</text></svg>`;
  host.innerHTML = s;
  host.querySelectorAll("rect").forEach((rect) => {
    const r = rows[+rect.dataset.i];
    rect.style.cursor = "pointer";
    rect.addEventListener("mousemove", (e) => showTip(
      `<b>${r.k.toUpperCase()}</b><div class="r"><span>picks home</span><span>${pct(r.pickHomePct)}</span></div>` +
      `<div class="r"><span>home actually wins</span><span>${pct(r.actualHomeWinPct)}</span></div>` +
      `<div class="r"><span>bias</span><span>${sgn(r.biasPct)}pts</span></div>` +
      `<div class="r"><span>n</span><span>${r.n}</span></div>`, e));
    rect.addEventListener("mouseleave", hideTip);
  });
}

function cardsPanel(A, E) {
  const cards = el("div", "tiles");
  const sum = A.summary || {};
  const card = (title, big, sub, tone) => {
    const d = el("button", "tile");
    d.disabled = true;
    d.style.minHeight = "0";
    d.innerHTML = `<div class="lbl">${title}</div>` +
      `<div class="num" style="font-size:31px;font-weight:600;line-height:1;color:${tone || "inherit"}">${big}</div>` +
      `<div style="font-size:12.5px;color:var(--muted)">${sub}</div>`;
    return d;
  };
  /* CLV is only measured on picks whose price was frozen when the game
     started. Everything else is the latest quote seen, which is a different
     number, and mixing them is how this read -0.52% over 96 picks while
     measuring nothing. A card that says "still filling" is information; a
     card that vanishes looks like a bug. */
  const clv = sum.closingLineValue;
  if (clv && clv.picks) {
    const beat = clv.beatCloseP;
    const err = clv.beatCloseStdErrPct;
    const verdict = clv.beatsCoinFlip
      ? "clears a coin flip"
      : "not yet distinguishable from a coin flip";
    cards.appendChild(card("Closing line value", sgn(clv.avgPct, 2) + "%",
      `${clv.picks} confirmed closes, beat the close ${pct(beat)}` +
      (err != null ? ` ±${err}` : "") + ` of the time -- ${verdict}.`,
      clv.avgPct > 0 ? "var(--good)" : "var(--bad)"));
  } else if (clv && clv.provisionalPicks) {
    cards.appendChild(card("Closing line value", "--",
      `${clv.provisionalPicks} picks priced, none with a confirmed closing line yet. ` +
      `A price only counts once it is frozen at first pitch.`));
  }
  const wf = E?.fittedWalkForward;
  if (wf) cards.appendChild(card("Walk-forward, out of sample", wf.logLoss.toFixed(4),
    `${wf.folds} folds, n=${wf.n}. Every build fails if this regresses against the checked-in baseline.`));
  /* The only fair model-vs-market comparison. The accuracy tables score the
     confidence that was published at the time, which pools every model version
     this log has carried -- so a model corrected last week still reads as
     losing until its own history washes out. This is the model running now,
     out of sample, on the games where a price existed. */
  const head = wf?.vsMarket;
  if (head?.n) cards.appendChild(card(
    "Live model vs market",
    (head.edge > 0 ? "+" : "") + head.edge.toFixed(4),
    `Out of sample on ${head.n} priced games: model ${head.modelLogLoss}, market ` +
    `${head.marketLogLoss}. Positive means the model prices those games better.`,
    head.edge > 0 ? "var(--good)" : "var(--bad)"));
  /* Totals/spreads only carry a price when ESPN core odds embedded one --
     SBR has nowhere to put it -- so the card reads "hit rate only" until
     that happens, then switches to real ROI, mirroring _market_summary. */
  const marketCard = (title, m) => {
    if (!m?.graded) return null;
    const record = `${m.wins}-${m.losses}` + (m.pushes ? `-${m.pushes}` : "");
    const roi = m.priced ? `${sgn(m.roiPct, 1)}% ROI. ` : "";
    // The error bar belongs beside the hit rate, not in a footnote. A few
    // dozen picks moves several points on noise alone, so a bare "61.6%"
    // reads as settled when the interval still spans break-even.
    const err = m.stdErrPct == null ? "" : ` ±${m.stdErrPct}`;
    // The break-even bar is derived from prices, so it can only be read
    // against the picks that carried one. Showing the all-graded rate next to
    // it made totals look like they beat break-even (53.2% vs 52.4%) while
    // losing 7.2% -- the two numbers covered different picks. When the priced
    // rate differs, it goes on screen beside the blended one.
    const pricedRate = (m.pricedPct != null && m.pricedPct !== m.pct)
      ? ` ${pct(m.pricedPct)}${m.pricedStdErrPct == null ? "" : ` ±${m.pricedStdErrPct}`}`
        + ` on the ${m.pricedDecided} priced.`
      : "";
    const verdict = m.beatsBreakEven === true
      ? " Clears break-even by more than the sample's error."
      : m.beatsBreakEven === false
        ? ` Not yet distinguishable from the ${pct(m.breakEvenPct)} break-even.`
        : "";
    return card(title, record,
      `${pct(m.pct)}${err} on ${m.decided ?? m.graded} decided.${pricedRate} ${roi}${verdict} ${m.note}`,
      m.priced && m.roiPct != null ? (m.roiPct > 0 ? "var(--good)" : m.roiPct < 0 ? "var(--bad)" : undefined) : undefined);
  };
  const totalsCard = marketCard("Totals", sum.totals);
  if (totalsCard) cards.appendChild(totalsCard);
  const spreadsCard = marketCard("Spreads / runline", sum.spreads);
  if (spreadsCard) cards.appendChild(spreadsCard);
  const dv = E?.divergence;
  if (dv) {
    cards.appendChild(card("Divergence from market", dv.medianGapPct.toFixed(1) + "pts",
      `Median gap. ${dv.shareOver15Pct}% of games differ by more than 15 points.`));
    if (dv.fadesMarket) cards.appendChild(card("Against the favourite", pct(dv.fadesMarket.winPct),
      `${dv.fadesMarket.picks} picks against the price, break-even ${pct(dv.fadesMarket.breakEvenPct)}.`,
      dv.fadesMarket.winPct > dv.fadesMarket.breakEvenPct ? "var(--good)" : "var(--bad)"));
  }
  if (sum.streak) cards.appendChild(card("Streak", `${sum.streak.current} ${sum.streak.type}`,
    `Best run ${sum.streak.bestWin} wins, worst ${sum.streak.bestLoss} losses. Streaks are noise at this sample size and are shown for context only.`));
  return cards;
}

/* Whether a queued candidate feature (h2h, handedness, bullpen, elo, ...)
   now beats the shipped set. Rebuilt on the same cadence as everything else
   in data/, so this is a rerun on a schedule rather than something someone
   has to remember to check by running model_fit.py --ablate by hand. */
function ablationPanel(ablation) {
  const p = el("div", "panel");
  p.appendChild(el("div", "phead",
    `<h2>Queued candidate features</h2><span class="n">rechecked every build, not from memory</span>`));
  const box = el("div");
  box.style.padding = "4px 16px 16px";
  const rows = ablation?.rows || [];
  if (!rows.length) {
    box.appendChild(el("div", "loading",
      "data/ablation.json is unavailable or too little has graded yet, so there is nothing to recheck against."));
    p.appendChild(box);
    return p;
  }
  const shippedRow = rows[ablation.shippedSize - 1] || rows[rows.length - 1];
  const queued = rows.slice(ablation.shippedSize);
  const when = new Date(ablation.generatedAt).toLocaleString(undefined,
    { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  box.appendChild(el("p", "",
    `As of ${esc(when)}, walk-forward on ${ablation.nSamples} graded games. Shipped: ` +
    `<b class="num" style="color:var(--ink)">${esc(shippedRow.features.join(" + "))}</b> at ` +
    `<b class="num" style="color:var(--ink)">${shippedRow.logLoss.toFixed(4)}</b> log loss (n=${shippedRow.n}).`));
  box.lastChild.style.cssText = "font-size:12.5px;color:var(--muted);margin:0 0 12px";

  if (!queued.length) {
    box.appendChild(el("div", "loading", "Every candidate feature this build knows how to test is already shipped."));
  } else {
    const ctx = el("div", "ctx");
    queued.forEach((row) => {
      const added = row.features[row.features.length - 1];
      const delta = row.logLoss - shippedRow.logLoss;
      const beats = delta < 0;
      const k = el("div", "k");
      k.innerHTML =
        `<div class="kv" style="color:${beats ? "var(--good)" : "var(--muted)"}">${beats ? "" : "+"}${delta.toFixed(4)}</div>` +
        `<div class="kk lbl">${esc(added)}</div>`;
      ctx.appendChild(k);
    });
    box.appendChild(ctx);
    box.appendChild(el("p", "",
      "Log loss added on top of the shipped set, cumulative through that feature -- lower is better, and none of " +
      "these ship until one actually beats it out of sample. A positive number here is not a bug; it is the record " +
      "saying this candidate is not earning its place yet."));
    box.lastChild.style.cssText = "font-size:12px;color:var(--muted);margin:13px 0 0";
  }
  p.appendChild(box);
  return p;
}

/* ============================================================ VIEW: DIG */
function renderDig() {
  const O = S.overview;
  const host = $("#digBody");
  host.innerHTML = "";
  if (!O) { host.appendChild(el("div", "failed", "Could not load data/overview.json.")); return; }

  const picks = (O.worthBacking || []).slice(0, 6);
  const pool = picks.length ? picks : (O.unpriced || []).slice(0, 6);
  const blindTotal = pool.reduce((n, p) => n + blindOf(p).length, 0);
  readings($("#digReadings"), [
    { v: pool.length, k: "Picks briefed" },
    { v: blindTotal, k: "Blind spots" },
    { v: (O.unpriced || []).length, k: "No price" },
    { v: (O.summary || {}).suggestedUnits ?? "—", k: "Suggested units" },
  ]);

  host.appendChild(el("div", "panel",
    `<div class="verdict" style="align-items:center"><span class="mark" style="color:var(--model)">&rarr;</span><div>` +
    `<p>Built for the handoff from this model to your own research: for each pick worth backing, what the model actually had, ` +
    `what it could not see, and the checks that would move the number. Everything is derived from the real coverage flags on ` +
    `each game rather than a generic checklist.</p></div></div>`));

  host.appendChild(ablationPanel(S.ablation));

  if (!pool.length) {
    host.appendChild(el("div", "panel", `<div class="loading">Nothing on the board to brief.</div>`));
  } else {
    const list = el("div", "dig");
    pool.forEach((p) => list.appendChild(digCard(p)));
    host.appendChild(list);
  }
  host.appendChild(suggestions("dig"));
}

/* The overview play carries no feature blob, so coverage is read from the
   league slate when it has been loaded and inferred conservatively otherwise.
   A blind spot is only claimed when the data says so. */
function featuresFor(play) {
  // Dig Deeper only ever briefs today's board (from overview.json), so the
  // cache key is always the league's today-date, same convention the Sports
  // view uses now that slates are cached per date.
  const payload = S.slates[`${play.league}:${todayFor(play.league)}`] ||
    S.slates[`${play.league}:${play.startDate ? play.startDate.slice(0, 10) : ""}`];
  if (!payload) return null;
  const game = (payload.games || []).find((g) => String(g.eventId) === String(play.eventId));
  return game ? (game.prediction || {}).features || null : null;
}

function blindOf(play) {
  const out = [];
  if (play.evPct == null) out.push(["price",
    "No moneyline reached the build from either source, so there is no edge figure and no stake size."]);
  const f = featuresFor(play);
  if (!f) return out;
  const cov = f.dataCoverage || {};
  const missing = (v) => v == null;
  if (missing(f.bullpenDiff) && play.league === "mlb") out.push(["bullpen",
    "No relief-innings reading for one or both sides, so recent bullpen workload is unknown."]);
  if (missing(f.handednessDiff) && play.league === "mlb") out.push(["handedness",
    "Starter handedness unresolved, so the lineup-versus-hand matchup is not accounted for."]);
  if (missing(f.h2hDiff)) out.push(["head to head", "No season series between these two yet."]);
  if (cov.lineup === false) out.push(["lineup", "Lineups were not posted when the build ran."]);
  if (cov.espnPredictor === false) out.push(["second opinion",
    "ESPN had not published its own win probability, so there is no independent check."]);
  if (missing(f.parkEdge) && play.league === "mlb") out.push(["park",
    "Venue not matched to a park factor — likely a neutral site or alternate ground."]);
  return out;
}

function knownOf(play) {
  const out = [];
  const f = featuresFor(play);
  if (play.odds != null) out.push(["price", `Taken at ${american(play.odds)}.`]);
  if (play.marketPct != null) out.push(["market", `The de-vigged price makes this ${pct(play.marketPct)}.`]);
  if (!f) {
    out.push(["note", "Open this game on the Sports view to load its full feature set."]);
    return out;
  }
  const cov = f.dataCoverage || {};
  /* Number(null) is 0 and Number("") is 0, so a feature the model logged as
     absent was arriving here as a real zero -- "no rating gap", "no travel",
     "no injuries" -- rather than as missing. Python writes None, JSON carries
     null, and every one of those became a confident 0 on the card. */
  const n = (v) => {
    if (v === null || v === undefined || v === "") return null;
    const x = Number(v);
    return Number.isFinite(x) ? x : null;
  };
  if (cov.lineup) out.push(["lineup", "Confirmed batting order for both sides."]);
  if (cov.mlbPitching && f.mlbPitching) out.push(["starters",
    `Home ${f.mlbPitching.homePitcherApiEra ?? "—"} ERA against away ${f.mlbPitching.awayPitcherApiEra ?? "—"}, plus both bullpens.`]);
  if (cov.injuries) out.push(["injuries",
    `Severity-weighted load ${n(f.homeInjuryLoad) ?? "—"} home / ${n(f.awayInjuryLoad) ?? "—"} away.`]);
  if (n(f.eloEdge) != null) out.push(["elo",
    `Pre-game rating gap of ${Math.abs(n(f.eloEdge)).toFixed(0)} points to ` +
    `${n(f.eloEdge) >= 0 ? teamShort(play.homeTeam) : teamShort(play.awayTeam)}.`]);
  if (n(f.parkEdge) != null) out.push(["park", `Run index ${sgn(n(f.parkEdge), 0)} against a neutral ballpark.`]);
  if (n(f.travelDiff) != null) out.push(["travel",
    `Travel edge ${sgn(n(f.travelDiff), 2)} to ${n(f.travelDiff) >= 0 ? teamShort(play.homeTeam) : teamShort(play.awayTeam)}.`]);
  return out;
}

function checksOf(play) {
  const out = [];
  const f = featuresFor(play);
  /* Number(null) is 0 and Number("") is 0, so a feature the model logged as
     absent was arriving here as a real zero -- "no rating gap", "no travel",
     "no injuries" -- rather than as missing. Python writes None, JSON carries
     null, and every one of those became a confident 0 on the card. */
  const n = (v) => {
    if (v === null || v === undefined || v === "") return null;
    const x = Number(v);
    return Number.isFinite(x) ? x : null;
  };
  if (fadesMarket(play) === true) {
    const dv = S.evaluation?.divergence?.fadesMarket;
    out.push(["against the favourite",
      `The market favours the other side and the model takes this one.` +
      (dv ? ` On the record these win ${pct(dv.winPct)} against a ${pct(dv.breakEvenPct)} break-even across ${dv.picks} picks — the thinnest margin on the board.` : "")]);
  }
  if (f) {
    if (n(f.bullpenDiff) == null && play.league === "mlb") out.push(["bullpen",
      "Pull up the last three games for both clubs and count relief innings. This is the input the model most wants and least often has."]);
    else if (n(f.bullpenDiff) != null && Math.abs(n(f.bullpenDiff)) >= 2) out.push(["bullpen",
      `A ${sgn(n(f.bullpenDiff), 1)}-inning workload gap is large. Check who is actually available tonight rather than who is on the roster.`]);
    if (n(f.handednessDiff) == null && play.league === "mlb") out.push(["handedness",
      "Confirm both starters' throwing hand and how each lineup splits against it."]);
  }
  const b = bucketFor(play.confidence);
  const borrowed = borrowedCalibrationNote(play.league);
  if (b && b.picks >= 30 && b.overconfidencePct > 8) out.push(["calibration",
    `This sits in the ${b.range}% band, which has actually won ${pct(b.actualWinPct)} across ${b.picks} picks. ` +
    `Treat the headline number as ${b.overconfidencePct.toFixed(0)} points optimistic.` +
    (borrowed ? ` Those picks are almost all other leagues, so read it as a prior about the model rather than about ${leagueShort(play)}.` : "")]);
  else if (borrowed) out.push(["unproven league", borrowed]);
  if (play.evPct == null) out.push(["price",
    "Find a price yourself before staking. Without one there is no edge and no sizing, only a probability."]);
  return out.slice(0, 4);
}

function digCard(play) {
  const d = el("div", "dg");
  d.innerHTML =
    `<div class="dh"><span class="chip mdl">${esc(leagueShort(play))}</span>` +
      `<span class="who">${esc(play.pick || "")}</span>` +
      `<span class="mu">${esc(play.matchup || "")}</span>` +
      `<span style="margin-left:auto" class="num">${pct(play.confidence)}</span>` +
      (play.evPct == null
        ? `<span class="chip warn">unpriced</span>`
        : `<span class="num" style="color:${play.evPct > 0 ? "var(--good)" : "var(--bad)"}">${sgn(play.evPct)}%</span>`) +
    `</div>`;
  const cols = el("div", "dcols");
  const mk = (cls, title, items, empty) => {
    const c = el("div", "dcol " + cls);
    c.innerHTML = `<h5>${title}</h5>`;
    const ul = el("ul");
    if (!items.length) ul.appendChild(el("li", "", `<span style="color:var(--muted)">${empty}</span>`));
    items.forEach(([b, t]) => ul.appendChild(el("li", "", `<span class="b">${esc(b)}</span><span>${esc(t)}</span>`)));
    c.appendChild(ul);
    return c;
  };
  cols.appendChild(mk("know", "What the model had", knownOf(play), "Very little — treat this pick as weak."));
  cols.appendChild(mk("blind", "What it could not see", blindOf(play), "Full coverage on every input it tracks."));
  cols.appendChild(mk("check", "Worth twenty minutes", checksOf(play), "Nothing obvious outstanding."));
  d.appendChild(cols);
  return d;
}

/* ==================================================== shared: suggestions */
const SUGGESTIONS = {
  board: [
    ["Ranked by edge, not confidence",
      "A 93% pick at a price that already assumes 93% is not a bet. The board leads with expected value, and confidence rides alongside it."],
    ["Against the favourite is flagged separately from a wide gap",
      "Being more confident than the market on the same side is not fading it. Only genuine fades carry the warning, because those are the picks the record says are barely above break-even."],
    ["Nothing is silently dropped",
      "Picks with no edge and picks with no price both stay on the page, in their own sections. A pick you can see and reject is information; one that vanished looks like a bug."],
    ["Out of season is distinguished from broken",
      "An empty slate and a failed feed both render as zero games. The tile says which, because only one of them needs looking at."],
  ],
  accuracy: [
    ["Two decision terms, shown as two decision terms",
      "The live model fits strengthDiff and marketLogit over a league intercept. Listing fifteen factors implies fifteen inputs move the number."],
    ["The calibration band travels with every pick",
      "Rather than a fabricated interval, each game carries what games in its band have actually done. That is the number to stake on."],
    ["Thin buckets are marked, not hidden",
      "A bucket under thirty picks is grey and excluded from the headline. The alternative is a 38-point miss on sixteen picks reading as a catastrophe."],
    ["Closing line value is promoted, not buried",
      "It is currently the weakest number on the page, which is exactly why it belongs in view."],
  ],
  dig: [
    ["It makes the model's ignorance legible",
      "Head-to-head, handedness, bullpen workload and Elo are logged but not yet trusted, because none has enough graded coverage to beat its own absence. Here that fact tells you which games are thinnest."],
    ["It converts a percentage into a next action",
      "A bare percentage tells you nothing about where to spend twenty minutes. A missing bullpen reading and a line that moved against you does."],
  ],
};

function suggestions(key) {
  const items = SUGGESTIONS[key] || [];
  const p = el("div", "panel");
  p.appendChild(el("div", "phead",
    `<h2>How to read this page</h2><span class="n">${items.length} notes</span>`));
  const box = el("div", "suglist");
  box.style.padding = "16px";
  items.forEach(([h, t]) => box.appendChild(el("div", "sug", `<h3>${esc(h)}</h3><p>${esc(t)}</p>`)));
  p.appendChild(box);
  return p;
}

/* ==================================================================== nav */
function go(view) {
  document.querySelectorAll(".view").forEach((n) => n.classList.toggle("on", n.id === "v-" + view));
  document.querySelectorAll("#nav button, #mbar button").forEach((b) => {
    if (b.dataset.v === view) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  });
  if (view === "sport") renderSport();
  try { history.replaceState(null, "", "#" + view); } catch { /* file:// */ }
  window.scrollTo({ top: 0, behavior: "instant" });
}

function wireNav() {
  document.querySelectorAll("#nav button, #mbar button").forEach((b) =>
    b.addEventListener("click", () => go(b.dataset.v)));
  addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea, select")) return;
    const map = { 1: "board", 2: "sport", 3: "accuracy", 4: "dig" };
    if (map[e.key]) go(map[e.key]);
    if ($("#v-sport").classList.contains("on")) {
      if (e.key === "ArrowLeft" && !$("#datePrev").disabled) setDate(datesFor(S.sport)[datesFor(S.sport).indexOf(selectedDateFor(S.sport)) - 1]);
      if (e.key === "ArrowRight" && !$("#dateNext").disabled) setDate(datesFor(S.sport)[datesFor(S.sport).indexOf(selectedDateFor(S.sport)) + 1]);
    }
  });
  $("#datePrev").addEventListener("click", () => {
    const dates = datesFor(S.sport);
    const idx = dates.indexOf(selectedDateFor(S.sport));
    if (idx > 0) setDate(dates[idx - 1]);
  });
  $("#dateNext").addEventListener("click", () => {
    const dates = datesFor(S.sport);
    const idx = dates.indexOf(selectedDateFor(S.sport));
    if (idx >= 0 && idx < dates.length - 1) setDate(dates[idx + 1]);
  });
  $("#sortSeg").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    S.sort = b.dataset.s;
    $("#sortSeg").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderSport();
  });
  $("#showSeg").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    S.filter = b.dataset.f;
    $("#showSeg").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderSport();
  });
  $("#theme").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const systemDark = matchMedia("(prefers-color-scheme: dark)").matches;
    const now = current || (systemDark ? "dark" : "light");
    const next = now === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("edgeboard-theme", next); } catch { /* private mode */ }
    renderAccuracy();
  });
  try {
    const saved = localStorage.getItem("edgeboard-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
  } catch { /* private mode */ }
}

/* =================================================================== boot */
async function boot() {
  wireNav();
  const [overview, accuracy, evaluation, weights, ablation, manifest] = await Promise.all([
    getJson("data/overview.json").catch((e) => { S.failures.push("overview.json"); return null; }),
    getJson("data/accuracy.json").catch(() => { S.failures.push("accuracy.json"); return null; }),
    getJson("data/evaluation.json").catch(() => { S.failures.push("evaluation.json"); return null; }),
    getJson("data/model_weights.json").catch(() => { S.failures.push("model_weights.json"); return null; }),
    // Whether a queued candidate feature (h2h, handedness, bullpen, elo, ...)
    // now beats the shipped set. Absent on a build that predates it -- the
    // Dig panel says so rather than showing a stale or fabricated table.
    getJson("data/ablation.json").catch(() => null),
    // Not fetched with the others' bare `catch(() => null)` because losing
    // this file has a specific, easy-to-miss consequence: the Sports view
    // silently degrades to today-only with no prev/next and no chips, and
    // nothing else on the page looks wrong. That is exactly the failure mode
    // this session exists to fix, so it gets its own line in the banner
    // rather than disappearing the way it did before this was noticed.
    getJson("data/manifest.json").catch(() => { S.failures.push("manifest.json (date navigation will be limited to today)"); return null; }),
  ]);
  S.overview = overview;
  S.accuracy = accuracy;
  S.evaluation = evaluation;
  S.weights = weights;
  S.ablation = ablation;
  S.manifest = manifest;

  $("#foot").innerHTML = accuracy
    ? `Built ${new Date(accuracy.updatedAt).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}<br>` +
      `${(accuracy.summary?.allTime || {}).total ?? 0} graded picks`
    : "Data unavailable";

  renderBoard();
  renderAccuracy();
  renderDig();

  if (S.failures.length) {
    const b = el("div", "banner",
      `<b>Some data did not load</b>${esc(S.failures.join(", "))}. The views below show what is available and say ` +
      `explicitly where a number is missing rather than substituting a placeholder.`);
    // Prepended INSIDE #boardBody, after renderBoard() has already run --
    // renderBoard() clears #boardBody's innerHTML at its own top, so
    // inserting the banner any earlier just gets wiped out by that clear.
    // It must also land inside .wrap rather than before it: a sibling of
    // .wrap sits outside its padding and max-width entirely, which is why the
    // first version ran full-bleed, 104px wider than every panel beneath it,
    // flush against the first one with no gap. As a first child it shares
    // .wrap's measure and picks up the standard `* + *` spacing for free.
    $("#boardBody").prepend(b);
  }

  const hash = (location.hash || "").replace("#", "");
  go(["board", "sport", "accuracy", "dig"].includes(hash) ? hash : "board");
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
else boot();
