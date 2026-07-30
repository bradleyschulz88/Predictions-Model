"""Turn subscribed YouTube channels into pre-game team news, as an ablation candidate.

Beat reporters and preview shows say things the box score does not: who is a
game-time decision, who is being rested, which bullpen is spent. This module
collects that from the channels you already subscribe to, extracts it into a
number per team, and offers it to the ablation as `videoIntelDiff`.

It is deliberately NOT wired into the live model. Like `eloDiff` and
`injurySeverityDiff` before it, it ships only if it beats its own absence out of
sample, and the honest prior is that it will not for priced leagues -- see the
"Why this probably will not help" note at the bottom of this docstring.

Why this runs locally rather than in CI
---------------------------------------
YouTube's official `captions.download` only works for videos you own, so the
sanctioned API cannot read other people's transcripts. The unofficial timedtext
endpoint can, but YouTube blocks datacenter IP ranges -- and GitHub Actions
runners are Azure. Running this in the Pages workflow returns empty transcripts
for everything. So it runs on your machine, on a residential IP, and commits
`video_intel.json`; the build just reads the file.

Leakage
-------
A recap video published after the final whistle describes who won. Fed to a
pre-game model that is a spectacular-looking improvement and complete fiction.
Every record therefore carries the video's `publishedAt`, and `intel_edge()`
refuses any video published at or after the game start. This is the same
discipline as Elo's pre-game edge.

Why this probably will not help (priced leagues)
------------------------------------------------
The model anchors to `marketLogit`, and a de-vigged closing line already
contains public information. A YouTube preview show is public. By the time a
pundit says it, it is captioned, and this job scrapes it, the line has moved.
Information already in the line contributes nothing by construction.

The exception is AFL, which has no odds source at all -- `STANDALONE_FEATURES`
is a single feature there, with no market to anchor to, so extra signal has
somewhere to go. That is the case worth testing first.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

INTEL_FILE = "video_intel.json"

YOUTUBE_API = "https://www.googleapis.com/youtube/v3"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
TIMEDTEXT_URL = "https://www.youtube.com/api/timedtext"
WATCH_URL = "https://www.youtube.com/watch"

# Quota is 10,000 units/day. subscriptions.list and playlistItems.list cost 1
# unit each, channels.list 1 -- so a hundred channels costs a few hundred units.
# search.list costs 100 and is deliberately never used.
MAX_CHANNELS = 200
MAX_VIDEOS_PER_CHANNEL = 5
LOOKBACK_HOURS = 36

HTTP_TIMEOUT = 20.0

# Transcripts are long; this is roughly the front of a preview segment, which is
# where availability news lands. Sending 60k characters to the model would blow
# the free tier's token budget for no extra signal.
MAX_TRANSCRIPT_CHARS = 12000

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
LLM_TIMEOUT_SECONDS = 45.0
MIN_SECONDS_BETWEEN_CALLS = 1.6
MAX_CALLS_PER_RUN = 120

_last_call_at = 0.0
_calls_made = 0


def reset_llm_budget() -> None:
    global _last_call_at, _calls_made
    _last_call_at = 0.0
    _calls_made = 0


# --------------------------------------------------------------------------
# YouTube discovery. Cheap, reliable, and the only part that uses the real API.
# --------------------------------------------------------------------------


class YouTubeAuthError(RuntimeError):
    """Credentials are absent or rejected."""


def _get_json(url: str, token: str, params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def access_token_from_refresh(
    client_id: str, client_secret: str, refresh_token: str
) -> str:
    """Exchange a long-lived refresh token for a short-lived access token.

    Only the refresh token is stored, and only ever outside the repo -- the
    access token lives for an hour and is never written down.
    """
    payload = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OAUTH_TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise YouTubeAuthError(f"token refresh rejected ({error.code})") from error
    except (urllib.error.URLError, ValueError, OSError) as error:
        raise YouTubeAuthError(f"token refresh failed: {error}") from error

    token = body.get("access_token")
    if not token:
        raise YouTubeAuthError("token response carried no access_token")
    return str(token)


def subscribed_channels(token: str, limit: int = MAX_CHANNELS) -> list[dict[str, str]]:
    """Every channel the authenticated account subscribes to. 1 quota unit per page."""
    channels: list[dict[str, str]] = []
    page: str | None = None

    while len(channels) < limit:
        params = {"part": "snippet", "mine": "true", "maxResults": "50"}
        if page:
            params["pageToken"] = page
        body = _get_json(f"{YOUTUBE_API}/subscriptions", token, params)

        for item in body.get("items") or []:
            snippet = item.get("snippet") or {}
            channel_id = ((snippet.get("resourceId") or {}).get("channelId"))
            if channel_id:
                channels.append({"id": str(channel_id), "title": snippet.get("title") or ""})

        page = body.get("nextPageToken")
        if not page:
            break

    return channels[:limit]


def uploads_playlists(token: str, channel_ids: Iterable[str]) -> dict[str, str]:
    """Map channel id -> uploads playlist id, 50 channels per quota unit.

    Listing a playlist costs 1 unit; searching a channel costs 100. This is the
    difference between a few hundred units a day and blowing the whole quota.
    """
    ids = [cid for cid in channel_ids if cid]
    playlists: dict[str, str] = {}

    for start in range(0, len(ids), 50):
        batch = ids[start : start + 50]
        body = _get_json(
            f"{YOUTUBE_API}/channels",
            token,
            {"part": "contentDetails", "id": ",".join(batch), "maxResults": "50"},
        )
        for item in body.get("items") or []:
            uploads = (
                ((item.get("contentDetails") or {}).get("relatedPlaylists") or {}).get("uploads")
            )
            if uploads:
                playlists[str(item.get("id"))] = str(uploads)

    return playlists


def _parse_published(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def recent_videos(
    token: str,
    uploads_playlist: str,
    *,
    channel_title: str = "",
    since: datetime | None = None,
    limit: int = MAX_VIDEOS_PER_CHANNEL,
) -> list[dict[str, Any]]:
    """Newest uploads from one channel, newest first. 1 quota unit."""
    body = _get_json(
        f"{YOUTUBE_API}/playlistItems",
        token,
        {"part": "snippet", "playlistId": uploads_playlist, "maxResults": str(min(limit, 50))},
    )

    videos: list[dict[str, Any]] = []
    for item in body.get("items") or []:
        snippet = item.get("snippet") or {}
        video_id = ((snippet.get("resourceId") or {}).get("videoId"))
        published = _parse_published(snippet.get("publishedAt"))
        if not video_id or published is None:
            continue
        if since and published < since:
            continue
        videos.append(
            {
                "videoId": str(video_id),
                "title": snippet.get("title") or "",
                "channel": channel_title or snippet.get("channelTitle") or "",
                "publishedAt": published.isoformat(),
            }
        )
    return videos[:limit]


# --------------------------------------------------------------------------
# Transcripts. The part YouTube does not want automated, hence the local run.
# --------------------------------------------------------------------------


def _caption_tracks(video_id: str) -> list[dict[str, str]]:
    """Available caption tracks, read off the watch page's player config."""
    request = urllib.request.Request(
        f"{WATCH_URL}?{urllib.parse.urlencode({'v': video_id})}",
        headers={
            # Without a browser UA the watch page returns a consent interstitial
            # with no player config in it.
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            html = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return []

    match = re.search(r'"captionTracks":(\[.*?\])', html)
    if not match:
        return []
    try:
        tracks = json.loads(match.group(1).replace("\\u0026", "&"))
    except json.JSONDecodeError:
        return []

    return [
        {"url": track.get("baseUrl", ""), "lang": (track.get("languageCode") or "")}
        for track in tracks
        if isinstance(track, dict) and track.get("baseUrl")
    ]


def fetch_transcript(video_id: str, *, prefer_lang: str = "en") -> str | None:
    """Plain-text transcript for a video, or None if it has no usable captions.

    Returns None rather than raising on a blocked request: from a datacenter IP
    every call fails, and the caller's job is to record that the intel is
    missing, not to crash the run.
    """
    tracks = _caption_tracks(video_id)
    if not tracks:
        return None

    chosen = next((t for t in tracks if t["lang"].startswith(prefer_lang)), tracks[0])
    try:
        with urllib.request.urlopen(chosen["url"], timeout=HTTP_TIMEOUT) as response:
            xml_body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None

    try:
        root = ET.fromstring(xml_body)
    except ET.ParseError:
        return None

    parts = [(node.text or "").strip() for node in root.iter("text")]
    text = " ".join(part for part in parts if part)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


# --------------------------------------------------------------------------
# Extraction. Turns prose into one number per team.
# --------------------------------------------------------------------------

_PROMPT = (
    "You extract pre-game team news from a sports video transcript.\n"
    "For every team discussed, rate how the news affects that team's chance of "
    "winning its NEXT game, on a -3 to +3 scale: -3 = a star is ruled out or "
    "the team is in crisis, -1 = minor concern, 0 = neutral or no real news, "
    "+1 = mildly positive, +3 = a key player returns or the team is in "
    "outstanding form.\n"
    "Only rate teams the transcript actually discusses. Ignore speculation "
    "about games already played.\n"
    'Reply ONLY with a JSON object mapping full team name to its number, e.g. '
    '{{"New York Yankees": -2}}. No prose.\n\n'
    "League: {league}\nTranscript:\n{transcript}"
)


def llm_enabled() -> bool:
    return _api_key() is not None


def _api_key() -> str | None:
    """The key, stripped. See data_providers.injury_severity.api_key for why.

    A trailing newline from pasting the key into the GitHub secrets field makes
    urllib raise `Invalid header value` and the request never goes out, which
    reads as an unreachable API rather than a formatting problem.
    """
    raw = os.environ.get("NVIDIA_API_KEY")
    if raw is None:
        return None
    key = raw.strip()
    if not key or any(character.isspace() for character in key):
        return None
    return key


def _throttle() -> None:
    global _last_call_at
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def _call_nvidia(prompt: str, api_key: str) -> str | None:
    global _calls_made
    if _calls_made >= MAX_CALLS_PER_RUN:
        return None

    payload = json.dumps(
        {
            "model": NVIDIA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            # Deterministic: re-running the same day's videos must not produce a
            # different number, or the feature becomes unmeasurable.
            "temperature": 0.0,
            "max_tokens": 500,
        }
    ).encode("utf-8")

    for attempt in range(2):
        _throttle()
        _calls_made += 1
        request = urllib.request.Request(
            NVIDIA_BASE_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt == 0:
                time.sleep(4.0)
                continue
            return None
        except (urllib.error.URLError, ValueError, TimeoutError, OSError):
            return None

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None

    return None


def _parse_scores(reply: str | None) -> dict[str, float]:
    """Pull the JSON object out of a reply that may be wrapped in prose or fences."""
    if not reply:
        return {}
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}

    scores: dict[str, float] = {}
    for team, value in raw.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        # The scale is -3..+3; anything outside is the model ignoring the rubric.
        scores[str(team).strip()] = max(-3.0, min(3.0, number))
    return scores


def extract_team_news(transcript: str, league: str) -> dict[str, float]:
    """Team -> news impact in [-3, 3]. Empty when the model is off or unhelpful."""
    key = _api_key()
    if not key or not transcript:
        return {}
    prompt = _PROMPT.format(league=league, transcript=transcript[:MAX_TRANSCRIPT_CHARS])
    return _parse_scores(_call_nvidia(prompt, key))


# --------------------------------------------------------------------------
# Read side. This is the only part the prediction build touches.
# --------------------------------------------------------------------------


def load_intel(data_dir: Path | None = None) -> dict[str, Any]:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "docs" / "data"
    path = data_dir / INTEL_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _team_score(
    records: list[dict[str, Any]], team: str | None, cutoff: datetime | None
) -> float | None:
    """Mean news impact for one team across videos published before the cutoff."""
    if not team:
        return None
    wanted = team.strip().casefold()
    values: list[float] = []

    for record in records:
        published = _parse_published(record.get("publishedAt"))
        # The leakage guard. A video published at or after first pitch may be
        # describing the result.
        if cutoff is not None and (published is None or published >= cutoff):
            continue
        for name, score in (record.get("teams") or {}).items():
            if str(name).strip().casefold() == wanted:
                try:
                    values.append(float(score))
                except (TypeError, ValueError):
                    continue

    if not values:
        return None
    return sum(values) / len(values)


def intel_edge(
    intel: dict[str, Any],
    league: str,
    home: str | None,
    away: str | None,
    start: str | datetime | None,
) -> float | None:
    """Home news impact minus away, from videos published before the game starts.

    None when either side has no coverage -- a one-sided score would read as an
    edge when it only means one team was talked about.
    """
    records = (intel.get("leagues") or {}).get(league) or []
    if not records:
        return None

    cutoff = start if isinstance(start, datetime) else _parse_published(start)
    home_score = _team_score(records, home, cutoff)
    away_score = _team_score(records, away, cutoff)
    if home_score is None or away_score is None:
        return None
    return home_score - away_score


# --------------------------------------------------------------------------
# The local job.
# --------------------------------------------------------------------------


def _credentials() -> tuple[str, str, str]:
    client_id = os.environ.get("YOUTUBE_CLIENT_ID", "")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET", "")
    refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN", "")
    if not (client_id and client_secret and refresh_token):
        raise YouTubeAuthError(
            "Set YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET and YOUTUBE_REFRESH_TOKEN "
            "(a gitignored .env is the intended home; never commit them)."
        )
    return client_id, client_secret, refresh_token


def classify_league(text: str) -> str | None:
    """Which league a video is about, from its title and channel.

    Deliberately conservative: an unrecognised video is skipped rather than
    guessed into a league, because a mislabelled record pollutes that league's
    feature for every game that day.
    """
    haystack = text.casefold()
    markers = {
        "mlb": ("mlb", "baseball", "world series"),
        "nba": ("nba", "basketball"),
        "wnba": ("wnba",),
        "nfl": ("nfl", "football", "super bowl"),
        "epl": ("premier league", "epl", "football club"),
        "afl": ("afl", "aussie rules", "australian football"),
    }
    # WNBA before NBA: "wnba" contains "nba" and would otherwise mislabel.
    for league in ("wnba", "afl", "epl", "mlb", "nba", "nfl"):
        if any(marker in haystack for marker in markers[league]):
            return league
    return None


def build_intel(data_dir: Path, *, lookback_hours: int = LOOKBACK_HOURS) -> dict[str, Any]:
    """Full local pass: subscriptions -> new videos -> transcripts -> team news."""
    client_id, client_secret, refresh_token = _credentials()
    token = access_token_from_refresh(client_id, client_secret, refresh_token)
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    channels = subscribed_channels(token)
    playlists = uploads_playlists(token, [channel["id"] for channel in channels])
    titles = {channel["id"]: channel["title"] for channel in channels}

    by_league: dict[str, list[dict[str, Any]]] = {}
    seen = skipped_no_transcript = skipped_unclassified = 0

    for channel_id, playlist in playlists.items():
        for video in recent_videos(
            token, playlist, channel_title=titles.get(channel_id, ""), since=since
        ):
            seen += 1
            league = classify_league(f"{video['title']} {video['channel']}")
            if not league:
                skipped_unclassified += 1
                continue

            transcript = fetch_transcript(video["videoId"])
            if not transcript:
                skipped_no_transcript += 1
                continue

            teams = extract_team_news(transcript, league)
            if not teams:
                continue

            by_league.setdefault(league, []).append(
                {
                    "videoId": video["videoId"],
                    "title": video["title"],
                    "channel": video["channel"],
                    # The leakage guard depends entirely on this field.
                    "publishedAt": video["publishedAt"],
                    "teams": teams,
                }
            )

    payload = {
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "lookbackHours": lookback_hours,
        "model": NVIDIA_MODEL if llm_enabled() else None,
        "videosSeen": seen,
        "skippedNoTranscript": skipped_no_transcript,
        "skippedUnclassified": skipped_unclassified,
        "leagues": by_league,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / INTEL_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    data_dir = Path(__file__).resolve().parent / "docs" / "data"
    try:
        payload = build_intel(data_dir)
    except YouTubeAuthError as error:
        print(f"Auth: {error}")
        return 1

    print(f"Videos seen           : {payload['videosSeen']}")
    print(f"  no transcript       : {payload['skippedNoTranscript']}")
    print(f"  league unrecognised : {payload['skippedUnclassified']}")
    for league, records in sorted(payload["leagues"].items()):
        teams = {team for record in records for team in record["teams"]}
        print(f"  {league:9s} {len(records):3d} videos, {len(teams):3d} teams")

    if payload["videosSeen"] and payload["skippedNoTranscript"] == payload["videosSeen"]:
        print(
            "\nEvery transcript came back empty. That is what a datacenter IP "
            "looks like -- run this from your own machine, not CI."
        )
    print(f"\nWrote {data_dir / INTEL_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
