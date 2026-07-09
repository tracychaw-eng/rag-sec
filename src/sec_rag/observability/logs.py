"""Structured JSON logging for the serving path.

One JSON object per line on stdout: timestamp, level, logger, message,
plus any `extra={...}` fields the caller attaches. Log collectors
(CloudWatch, Loki, Datadog) ingest this without parsing rules.

setup_logging() is called from the API lifespan; CLI tools (ingestion,
eval) keep human-readable output.
"""

import json
import logging
import sys
import time

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S",
                                time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Route uvicorn's own loggers through the same handler/format
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
