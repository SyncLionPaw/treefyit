"""Tree navigation tools for agents.

Agents reference trees by ID — they never pass raw tree data.
Trees are auto-registered by ``build_tree()`` or manually via ``register()``.

Usage::

    from src.tools import overview, inspect, get_children, list_trees

    list_trees()                        # discover available trees
    overview("paper")                   # shape of the whole tree
    inspect("paper", "0.1")            # full details of node 0.1
    get_children("paper", "0.1")       # list children of node 0.1
"""

from src.tools.query import get_children, inspect, list_trees, overview, register, unregister

__all__ = ["register", "unregister", "list_trees", "overview", "inspect", "get_children"]
