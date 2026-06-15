#!/usr/bin/env python3
"""CLI for inspecting MLB odds and matchups from SportsBookReview."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from typing import Any

from mlb_data import (
    collect_odds_lines,
    collect_view_types,
    game_summary,
    load_page_props_from_file,
)
from sbr_client import (
    SBRClientError,
    build_matchup_url,
    build_matchups_url,
    build_odds_url,
    fetch_next_data,
    get_game_rows,
    get_matchup,
    get_page_props,
)


def _today_iso() -> str:
    return date.today().isoformat()


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _load_page_props(args: argparse.Namespace) -> dict[str, Any]:
    if args.fixture:
        return load_page_props_from_file(args.fixture)

    return get_page_props(
        args.url,
        retries=args.retries,
        retry_delay=args.delay,
        verify_ssl=not args.insecure,
    )


def cmd_list_odds(args: argparse.Namespace) -> int:
    page_props = _load_page_props(args)
    rows = get_game_rows(page_props)

    if args.format == "json":
        payload = []
        for row in rows:
            item = game_summary(row)
            item["oddsViews"] = [
                {
                    "sportsbook": ov.get("sportsbook"),
                    "viewType": ov.get("viewType"),
                }
                for ov in (row.get("oddsViews") or [])[: args.odds_limit]
                if ov
            ]
            payload.append(item)
        _print_json({"url": args.url, "gameCount": len(rows), "games": payload})
        return 0

    print("url", args.url)
    print("gameRows", len(rows))
    for row in rows:
        summary = game_summary(row)
        print(
            summary["startDate"],
            summary["awayTeam"],
            "@",
            summary["homeTeam"],
            summary["gameStatusText"],
            summary["venueName"],
        )
        for odds_view in (row.get("oddsViews") or [])[: args.odds_limit]:
            if odds_view:
                print(" ", odds_view.get("sportsbook"), odds_view.get("viewType"))
    return 0


def cmd_inspect_odds(args: argparse.Namespace) -> int:
    page_props = _load_page_props(args)
    rows = get_game_rows(page_props)
    sample = rows[: args.limit]

    if args.format == "json":
        payload = []
        for row in sample:
            item = game_summary(row)
            item["lines"] = collect_odds_lines(row, view_filter=args.view_filter)
            payload.append(item)
        _print_json({"url": args.url, "games": payload})
        return 0

    for row in sample:
        summary = game_summary(row)
        print(summary["awayTeam"], "@", summary["homeTeam"])
        for line in collect_odds_lines(row, view_filter=args.view_filter):
            print(line["viewType"], line["sportsbook"])
            print("opening", line["openingLine"])
            print("current", line["currentLine"])
        print("---")
    return 0


def cmd_viewtypes(args: argparse.Namespace) -> int:
    page_props = _load_page_props(args)
    rows = get_game_rows(page_props)

    if args.format == "json":
        payload = []
        for index, row in enumerate(rows):
            item = game_summary(row)
            item["index"] = index
            item["viewTypes"] = collect_view_types(row)
            payload.append(item)
        _print_json({"url": args.url, "games": payload})
        return 0

    for index, row in enumerate(rows):
        summary = game_summary(row)
        print("\nGAME", index, summary["awayTeam"], "@", summary["homeTeam"])
        for view_type in collect_view_types(row):
            print(view_type)
    return 0


def cmd_matchups(args: argparse.Namespace) -> int:
    page_props = _load_page_props(args)

    if args.format == "json":
        _print_json({"url": args.url, "pagePropsKeys": sorted(page_props.keys()), "pageProps": page_props})
        return 0

    print("pageProps keys:", sorted(page_props.keys()))
    print(json.dumps(page_props, indent=2)[: args.preview_chars])
    return 0


def cmd_matchup(args: argparse.Namespace) -> int:
    page_props = _load_page_props(args)

    if args.format == "json":
        _print_json({"url": args.url, "pagePropsKeys": sorted(page_props.keys()), "pageProps": page_props})
        return 0

    print("pageProps keys:", sorted(page_props.keys()))
    print(json.dumps(page_props, indent=2)[: args.preview_chars])
    return 0


def cmd_matchup_keys(args: argparse.Namespace) -> int:
    page_props = _load_page_props(args)
    matchup = get_matchup(page_props)
    odds_views = matchup.get("oddsViews")
    if not isinstance(odds_views, dict):
        raise SBRClientError("matchup.oddsViews missing or not an object")

    if args.format == "json":
        payload = {
            key: {
                "type": type(value).__name__,
                "length": len(value) if hasattr(value, "__len__") else None,
            }
            for key, value in odds_views.items()
        }
        _print_json({"url": args.url, "oddsViews": payload})
        return 0

    print(sorted(odds_views.keys()))
    for key, value in odds_views.items():
        length = len(value) if hasattr(value, "__len__") else ""
        print(key, type(value).__name__, length)
    return 0


def cmd_spreads(args: argparse.Namespace) -> int:
    page_props = _load_page_props(args)
    matchup = get_matchup(page_props)
    odds_views = matchup.get("oddsViews") or {}
    spreads = odds_views.get("spreadOddsViews")
    if not isinstance(spreads, list):
        raise SBRClientError("spreadOddsViews missing or not a list")

    if args.format == "json":
        payload = [
            {
                "sportsbook": item.get("sportsbook"),
                "openingLine": item.get("openingLine"),
                "currentLine": item.get("currentLine"),
            }
            for item in spreads
        ]
        _print_json({"url": args.url, "spreads": payload})
        return 0

    for item in spreads:
        print(item.get("sportsbook"), item.get("openingLine"), item.get("currentLine"))
    return 0


def _dashboard_host_port(args: argparse.Namespace) -> tuple[str, int, bool]:
    port_env = os.environ.get("PORT")
    if port_env:
        return "0.0.0.0", int(port_env), False
    return args.host, args.port, not args.no_browser


def cmd_dashboard(args: argparse.Namespace) -> int:
    from dashboard_server import DashboardConfig, run_dashboard

    host, port, open_browser = _dashboard_host_port(args)
    config = DashboardConfig(
        league=args.league,
        source=args.source,
        fixture=args.fixture,
        retries=args.retries,
        retry_delay=args.delay,
        verify_ssl=not args.insecure,
        view_filter=args.view_filter,
        include_odds=not args.no_odds,
        include_enrichment=not args.no_enrichment,
    )
    run_dashboard(
        host=host,
        port=port,
        open_browser=open_browser,
        config=config,
        strict_port=bool(os.environ.get("PORT")),
    )
    return 0


def cmd_dump(args: argparse.Namespace) -> int:
    if args.fixture:
        with open(args.fixture, encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = fetch_next_data(
            args.url,
            retries=args.retries,
            retry_delay=args.delay,
            verify_ssl=not args.insecure,
        )

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
        handle.write("\n")

    print(f"Wrote {args.output}")
    return 0


def _add_network_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fixture",
        help="Load saved __NEXT_DATA__ or pageProps JSON instead of fetching live",
    )
    parser.add_argument("--retries", type=int, default=3, help="HTTP retry attempts")
    parser.add_argument("--delay", type=float, default=1.0, help="Base delay between retries (seconds)")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS certificate verification (use only if your environment blocks SBR)",
    )


def _add_format_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_odds = subparsers.add_parser("list-odds", help="List games from the odds page")
    list_odds.add_argument("--url", default=build_odds_url(_today_iso()))
    list_odds.add_argument("--date", help="Odds date (YYYY-MM-DD); overrides default url date")
    list_odds.add_argument("--odds-limit", type=int, default=5, help="Max oddsViews to show per game")
    _add_network_options(list_odds)
    _add_format_option(list_odds)
    list_odds.set_defaults(func=cmd_list_odds)

    inspect_odds = subparsers.add_parser("inspect-odds", help="Show sample spread/moneyline lines")
    inspect_odds.add_argument("--url", default=build_odds_url(_today_iso()))
    inspect_odds.add_argument("--date", help="Odds date (YYYY-MM-DD)")
    inspect_odds.add_argument("--limit", type=int, default=2, help="Number of games to inspect")
    inspect_odds.add_argument(
        "--view-filter",
        default="Spread|MoneyLine",
        help="Regex-style substring filter for viewType (default: Spread or MoneyLine)",
    )
    _add_network_options(inspect_odds)
    _add_format_option(inspect_odds)
    inspect_odds.set_defaults(func=cmd_inspect_odds)

    viewtypes = subparsers.add_parser("viewtypes", help="List unique odds view types per game")
    viewtypes.add_argument("--url", default=build_odds_url(_today_iso()))
    viewtypes.add_argument("--date")
    _add_network_options(viewtypes)
    _add_format_option(viewtypes)
    viewtypes.set_defaults(func=cmd_viewtypes)

    matchups = subparsers.add_parser("matchups", help="Inspect MLB matchups listing pageProps")
    matchups.add_argument("--url", default=build_matchups_url())
    matchups.add_argument("--preview-chars", type=int, default=5000)
    _add_network_options(matchups)
    _add_format_option(matchups)
    matchups.set_defaults(func=cmd_matchups)

    matchup = subparsers.add_parser("matchup", help="Inspect a single matchup pageProps")
    matchup.add_argument("--id", dest="matchup_id", type=int, help="Matchup ID")
    matchup.add_argument("--url", help="Full matchup URL (overrides --id)")
    matchup.add_argument("--preview-chars", type=int, default=4000)
    _add_network_options(matchup)
    _add_format_option(matchup)
    matchup.set_defaults(func=cmd_matchup)

    matchup_keys = subparsers.add_parser("matchup-keys", help="List keys under matchup.oddsViews")
    matchup_keys.add_argument("--id", dest="matchup_id", type=int, help="Matchup ID")
    matchup_keys.add_argument("--url", help="Full matchup URL (overrides --id)")
    _add_network_options(matchup_keys)
    _add_format_option(matchup_keys)
    matchup_keys.set_defaults(func=cmd_matchup_keys)

    spreads = subparsers.add_parser("spreads", help="Show spread lines for a matchup")
    spreads.add_argument("--id", dest="matchup_id", type=int, help="Matchup ID")
    spreads.add_argument("--url", help="Full matchup URL (overrides --id)")
    _add_network_options(spreads)
    _add_format_option(spreads)
    spreads.set_defaults(func=cmd_spreads)

    dump = subparsers.add_parser("dump", help="Save __NEXT_DATA__ JSON to a fixture file")
    dump.add_argument("--url", required=True, help="URL to fetch")
    dump.add_argument("--output", required=True, help="Output JSON file path")
    dump.add_argument("--fixture", help="Copy an existing fixture instead of fetching")
    dump.add_argument("--retries", type=int, default=3)
    dump.add_argument("--delay", type=float, default=1.0)
    dump.add_argument("--insecure", action="store_true")
    dump.set_defaults(func=cmd_dump)

    dashboard = subparsers.add_parser("dashboard", help="Launch the local web dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument(
        "--league",
        choices=("mlb", "worldcup", "afl"),
        default="mlb",
        help="League to show (default: mlb)",
    )
    dashboard.add_argument(
        "--source",
        choices=("espn", "sbr"),
        default="espn",
        help="Schedule source (default: espn)",
    )
    dashboard.add_argument("--fixture", help="Serve games from a saved JSON fixture")
    dashboard.add_argument("--retries", type=int, default=3)
    dashboard.add_argument("--delay", type=float, default=1.0)
    dashboard.add_argument("--insecure", action="store_true")
    dashboard.add_argument(
        "--view-filter",
        default="Spread|MoneyLine|Total",
        help="Default market filter for merged sportsbook odds",
    )
    dashboard.add_argument(
        "--no-odds",
        action="store_true",
        help="Skip SportsBookReview odds merge; show schedule only",
    )
    dashboard.add_argument(
        "--no-enrichment",
        action="store_true",
        help="Skip ESPN summary fetches (faster, less detailed reasoning)",
    )
    dashboard.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    dashboard.set_defaults(func=cmd_dashboard)

    return parser


def _resolve_urls(args: argparse.Namespace) -> None:
    if getattr(args, "date", None):
        args.url = build_odds_url(args.date)

    if getattr(args, "matchup_id", None) and not getattr(args, "url", None):
        args.url = build_matchup_url(args.matchup_id)

    if getattr(args, "fixture", None) and getattr(args, "url", None) is None:
        args.url = f"fixture:{args.fixture}"
        return

    if hasattr(args, "url") and args.url is None and args.command in {"matchup", "matchup-keys", "spreads"}:
        raise SBRClientError("Provide --id or --url for matchup commands")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _resolve_urls(args)
        return args.func(args)
    except SBRClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
