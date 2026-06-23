"""treefyit — one call to parse, summarize, and visualize."""

import argparse
from pathlib import Path


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

    # preview — parse only, no LLM
    p_preview = sub.add_parser(
        "preview", help="Parse a file and print the tree (no LLM)"
    )
    p_preview.add_argument("file", help="Path to Markdown / text file")
    p_preview.add_argument(
        "-o", "--html", default=None, help="Export tree to HTML file"
    )
    p_preview.add_argument(
        "-s",
        "--snippet",
        type=int,
        nargs="?",
        const=48,
        default=0,
        metavar="N",
        help="Append a short body preview (N chars, default 48 when flag is set)",
    )

    # url — fetch remote page(s), parse tree, no LLM
    p_url = sub.add_parser("url", help="Fetch a URL and print the tree (no LLM)")
    p_url.add_argument("url", help="http(s):// link")
    p_url.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Follow child links (BFS crawl)",
    )
    p_url.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Max crawl depth when --recursive (default: 2)",
    )
    p_url.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save fetched Markdown/text to a file",
    )
    p_url.add_argument(
        "-s",
        "--snippet",
        type=int,
        nargs="?",
        const=48,
        default=0,
        metavar="N",
        help="Append a short body preview (N chars, default 48 when flag is set)",
    )
    p_url.add_argument(
        "-o", "--html", default=None, help="Export tree visualization to HTML"
    )

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

    elif args.command == "preview":
        from src.parser.html import parse_html
        from src.parser.md import parse_md
        from src.tree.builder import build_nodes
        from src.vis import save_html, show

        path = args.file.lower()
        if path.endswith((".html", ".htm")):
            nodes = parse_html(args.file)
        else:
            nodes = parse_md(args.file)
        tree = build_nodes(nodes)
        show(tree, max_text=args.snippet)
        if args.html:
            save_html(tree, args.html)
            print(f"HTML saved to {args.html}")

    elif args.command == "url":
        from src.parser.md import parse_md_text
        from src.parser.url import is_url, parse_url
        from src.tree.builder import build_nodes
        from src.vis import save_html, show

        if not is_url(args.url):
            raise SystemExit(f"Not a URL: {args.url!r}")

        print(f"Fetching {args.url} ...", flush=True)
        text = parse_url(
            args.url,
            recursive=args.recursive,
            max_depth=args.depth,
        )
        print(f"Fetched {len(text)} chars", flush=True)

        if args.save:
            Path(args.save).write_text(text, encoding="utf-8")
            print(f"Saved to {args.save}")

        nodes = parse_md_text(text)
        print(f"Sections: {len(nodes)}", flush=True)
        tree = build_nodes(nodes)
        show(tree, max_text=args.snippet)
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
