"""MD pipeline + serialize/visualize/persist — python tests/test_md.py"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import parse_md
from src.tree import build_tree
from src.util import save, load
from src.vis import show, save_html

MD = Path(__file__).resolve().parent.parent.parent / "examples/tutorials/doc-search/description.md"
OUT = Path(__file__).resolve().parent / "output"


def main():
    # 1. Parse + Build
    print("=" * 50)
    print("1. Terminal tree (box-drawing)")
    print("=" * 50)
    nodes = parse_md(str(MD))
    tree = build_tree(nodes)
    show(tree)

    # 2. JSON
    print("\n" + "=" * 50)
    print("2. Save / Load JSON")
    print("=" * 50)
    save(tree, OUT.with_suffix(".json"))
    loaded = load(OUT.with_suffix(".json"))
    print(f"  {OUT.with_suffix('.json')} ({OUT.with_suffix('.json').stat().st_size} bytes)")
    print(f"  round-trip OK: {loaded[0]['title']}")

    # 3, HTML
    print("\n" + "=" * 50)
    print("3. HTML export")
    print("=" * 50)
    save_html(tree, str(OUT.with_suffix(".html")), title="Description Document")
    print(f"  {OUT.with_suffix('.html')} ({OUT.with_suffix('.html').stat().st_size} bytes)")
    print(f"  open {OUT.with_suffix('.html')}")

    # Cleanup
    OUT.with_suffix(".json").unlink()
    OUT.with_suffix(".html").unlink()


if __name__ == "__main__":
    main()
