from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
TREEFYIT_HANDLER_NAME = "treefyit.stderr"


def configure_treefyit_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger("treefyit")
    logger.setLevel(level)
    logger.propagate = False

    for handler in logger.handlers:
        if handler.get_name() == TREEFYIT_HANDLER_NAME:
            handler.setLevel(level)
            return

    handler = logging.StreamHandler()
    handler.set_name(TREEFYIT_HANDLER_NAME)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
