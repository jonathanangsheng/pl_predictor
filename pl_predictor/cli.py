"""Command-line entry point: `pl-predictor <command>`."""

from __future__ import annotations

import argparse
import logging
import sys
import time

import requests

from .config import LOG_FORMAT, ConfigError, get_settings, require_token


def _cmd_fixtures(args, settings) -> None:
    from .pipeline import collectable_fixtures, near, season_fixtures

    fixtures = near(season_fixtures(settings))
    active = {f.fixture_id for f in collectable_fixtures(fixtures, settings)}
    for f in fixtures:
        flag = "*" if f.fixture_id in active else " "
        kickoff = f"{f.kickoff:%a %d %b %H:%M}Z"
        print(f"{flag} {f.fixture_id:<16} MW{f.matchweek:<3} {kickoff}  {f.label}")
    print(
        "\nFixtures within a week of today. * = inside the recent-search window (collectable now)"
    )


def _cmd_query(args, settings) -> None:
    from .pipeline import check_fixture_ids, near, season_fixtures

    fixtures = season_fixtures(settings)
    if args.fixture:
        check_fixture_ids(fixtures, [args.fixture])
        fixtures = [f for f in fixtures if f.fixture_id == args.fixture]
    else:
        fixtures = near(fixtures)
    for f in fixtures:
        print(f"{f.fixture_id}: {f.build_query(settings.max_query_length)}")


FIXTURES_REFRESH_S = 15 * 60


def _cmd_run(args, settings) -> None:
    from .collector import TweetCollector
    from .pipeline import run_once, season_fixtures
    from .sentiment import SentimentScorer
    from .storage import Store

    collector = TweetCollector.from_bearer_token(require_token(settings))
    store = Store(settings.db_path)
    scorer = SentimentScorer(settings.model_name)
    fixtures, refreshed = None, 0.0
    while True:
        # The season list changes a few times a week: refresh it now and then, not every loop.
        if time.monotonic() - refreshed >= FIXTURES_REFRESH_S:
            try:
                fixtures, refreshed = season_fixtures(settings), time.monotonic()
            except requests.RequestException as exc:
                if not args.watch:
                    raise
                kept = "keeping the last list" if fixtures else "retrying next loop"
                logging.warning("FPL fixture refresh failed (%s); %s", exc, kept)
        if fixtures is not None:
            collected, scored = run_once(settings, store, collector, scorer, args.fixture, fixtures)
            logging.info("Run complete: %d collected, %d scored", collected, scored)
        if not args.watch:
            break
        time.sleep(args.watch)


def _cmd_score(args, settings) -> None:
    from .pipeline import score_pending
    from .sentiment import SentimentScorer
    from .storage import Store

    scored = score_pending(Store(settings.db_path), SentimentScorer(settings.model_name))
    logging.info("Scored %d tweets", scored)


def _cmd_demo(args, settings) -> None:
    from .demo import seed_demo
    from .pipeline import near, season_fixtures
    from .storage import Store

    added = seed_demo(Store(args.db), near(season_fixtures(settings)), args.tweets)
    logging.info("Seeded %d synthetic tweets into %s", added, args.db)
    logging.info("View it with: pl-predictor dashboard --db %s", args.db)


def _cmd_dashboard(args, settings) -> None:
    from .dashboard import create_app

    create_app(args.db or settings.db_path, settings.teams_path).run(
        host=args.host, port=args.port, debug=args.debug
    )


def _cmd_model(args, settings) -> None:
    from . import model_commands as commands
    from .model import ModelMissing
    from .predict import NoUpcomingFixtures

    try:
        if args.action == "predict":
            commands.predict(settings, gameweek=args.gameweek)
        else:
            run = {
                "evaluate": commands.evaluate,
                "train": commands.train,
                "score": commands.score_history,
                "study": commands.study,
            }
            run[args.action](settings)
    except ModelMissing as exc:
        print(exc)
        sys.exit(2)
    except NoUpcomingFixtures as exc:
        print(exc)


def _cmd_watch(args, settings) -> None:
    from . import watch

    watch.file_logging()
    watch.run(settings, once=args.once, dry_run=args.dry_run)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pl-predictor", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fixtures", help="list fixtures within a week of today (from FPL)").set_defaults(
        func=_cmd_fixtures
    )

    p = sub.add_parser("query", help="print the X search query for each fixture")
    p.add_argument("--fixture", help="only this fixture id")
    p.set_defaults(func=_cmd_query)

    p = sub.add_parser("run", help="collect tweets for active fixtures and score them")
    p.add_argument("--fixture", action="append", help="fixture id (repeatable)")
    p.add_argument("--watch", type=int, metavar="SECONDS", help="repeat every N seconds")
    p.set_defaults(func=_cmd_run)

    sub.add_parser("score", help="score any unscored tweets").set_defaults(func=_cmd_score)

    p = sub.add_parser(
        "demo", help="seed synthetic tweets for this week's fixtures (no X or model)"
    )
    p.add_argument("--tweets", type=int, default=600, help="tweets per fixture")
    # Kept out of the real database: demo tweet IDs would break since_id for real runs.
    p.add_argument("--db", default="data/demo.db", help="database to seed (default: %(default)s)")
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("dashboard", help="start the Plotly Dash dashboard")
    p.add_argument("--host", default="127.0.0.1", help="default: %(default)s")
    p.add_argument("--port", type=int, default=8050, help="default: %(default)s")
    p.add_argument("--db", help="database to show (default: PLS_DB_PATH)")
    p.add_argument("--debug", action="store_true", help="Dash debug mode with hot reload")
    p.set_defaults(func=_cmd_dashboard)

    p = sub.add_parser("model", help="match predictions: evaluate, train, predict, score, study")
    p.add_argument(
        "action",
        choices=["evaluate", "train", "predict", "score", "study"],
        help="evaluate: walk-forward test; train: save models/model.joblib; predict: next "
        "gameweek; score: logged predictions vs results; study: does sentiment help?",
    )
    p.add_argument("--gameweek", type=int, help="predict this gameweek instead of the next one")
    p.set_defaults(func=_cmd_model)

    p = sub.add_parser("watch", help="collect, score and predict each fixture at kickoff - 55 min")
    p.add_argument("--once", action="store_true", help="one pass, then exit")
    p.add_argument("--dry-run", action="store_true", help="list the next slots and exit")
    p.set_defaults(func=_cmd_watch)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    if args.verbose:  # this package only: urllib3 at DEBUG drowns out our logs
        logging.getLogger("pl_predictor").setLevel(logging.DEBUG)
    try:
        args.func(args, get_settings())
    except ConfigError as exc:
        sys.exit(str(exc))
    except requests.RequestException as exc:
        sys.exit(f"Could not reach a data source ({exc}); check the connection and retry.")
