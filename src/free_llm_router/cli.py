import argparse

import uvicorn

from free_llm_router.config import load_settings
from free_llm_router.server import create_app
from free_llm_router.wizard import run_wizard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="free-llm-router")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run the OpenAI-like API server.")
    serve_parser.add_argument("--config", default="config/config.yaml")

    wizard_parser = subparsers.add_parser("wizard", help="Run the interactive config wizard.")
    wizard_parser.add_argument("--output", default="config/config.yaml")

    check_parser = subparsers.add_parser("check", help="Validate the config file and print active inventory.")
    check_parser.add_argument("--config", default="config/config.yaml")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "wizard":
        run_wizard(args.output)
        return

    if args.command == "check":
        settings = load_settings(args.config)
        app = create_app(args.config)
        catalog = app.state.catalog
        print("Config: {0}".format(args.config))
        print("Host: {0}:{1}".format(settings.app.host, settings.app.port))
        print("Providers:")
        for provider in catalog.active_providers():
            print("  - {0} ({1})".format(provider.name, provider.id))
        print("Models:")
        for model in catalog.active_models():
            print("  - {0} [{1}, {2}, {3}]".format(
                model.id,
                model.performance_tier,
                ",".join(model.scenarios),
                model.context_length,
            ))
        return

    if args.command == "serve":
        settings = load_settings(args.config)
        uvicorn.run(
            create_app(args.config),
            host=settings.app.host,
            port=settings.app.port,
        )
