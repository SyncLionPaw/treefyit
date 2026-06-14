"""treefyit — one call to parse, summarize, and visualize."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Build hierarchical trees from Markdown files, with LLM summaries and visualization."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser("build", help="Build tree from a Markdown file")
    p_build.add_argument("file", help="Path to a Markdown (.md) file")
    p_build.add_argument(
        "-m",
        "--model",
        default="deepseek/deepseek-chat",
        help="LLM model (default: deepseek/deepseek-chat)",
    )
    p_build.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "md", "semantic"],
        help="Hierarchy mode: auto (headers + numbering), md (headers only), semantic (LLM-driven)",
    )
    p_build.add_argument(
        "--no-summarize", action="store_true", help="Skip LLM summary generation"
    )
    p_build.add_argument("-o", "--html", default=None, help="Export tree to HTML file")

    # serve
    p_serve = sub.add_parser("serve", help="Start the web UI")
    p_serve.add_argument(
        "-p", "--port", type=int, default=8765, help="Port to listen on (default: 8765)"
    )
    p_serve.add_argument(
        "-H", "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    p_serve.add_argument(
        "--clear",
        action="store_true",
        help="Clear old database, build payloads, caches, and uploads before starting",
    )

    args = parser.parse_args()

    if args.command == "build":
        from src.tree import build_tree
        from src.vis import save_html

        tree = build_tree(
            args.file,
            model=args.model,
            mode=args.mode,
            summarize=not args.no_summarize,
        )
        if args.html:
            save_html(tree, args.html)
            print(f"HTML saved to {args.html}")

    elif args.command == "serve":
        import uvicorn

        if args.clear:
            from src import store

            cleared = store.clear_all()
            print(f"Cleared treefyit state under {cleared}")

        from src.server.server import app

        print(f"🌲 treefyit UI → http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
