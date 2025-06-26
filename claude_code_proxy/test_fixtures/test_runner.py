#!/usr/bin/env python3
"""Test harness for API conversion fixtures."""
import argparse
import asyncio
import datetime
import json
import logging
import traceback
from contextlib import contextmanager
from pathlib import Path

import jinja2
from fastapi.responses import JSONResponse, StreamingResponse

from claude_code_proxy.server import handle_anthropic
from claude_code_proxy.test_fixtures.fixtures import FIXTURES


class RecordCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

@contextmanager
def capture(logger):
    s = RecordCapture()
    logger.addHandler(s)
    try:
        yield s
    finally:
        logger.removeHandler(s)
formatter = logging.Formatter(
  '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - '
  '%(message)s - %(pathname)s'
)

def log_line(record):
    formatted = formatter.format(record)
    standard_attrs = set(vars(logging.LogRecord('', 0, '', 0, '', (), None)))
    extras = {k: v for k, v in vars(record).items() if k not in standard_attrs}
    if extras:
        formatted += f' - Extras: {extras}'
    return formatted



async def main():
    parser = argparse.ArgumentParser(description="Run API conversion fixtures and generate HTML report")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("test_fixtures_output"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for fixture in FIXTURES:
        print()
        print()
        entry = {
            "fixture": fixture,
            "anthropic_request": fixture.anthropic_request,
        }

        with capture(logging.getLogger("claude_code_proxy")) as c:
            try:
                response = await handle_anthropic(fixture.anthropic_request)
            except Exception as e:
                entry["proxy_response"] = f"ERROR: {e}\n\n{traceback.format_exc()}"
                continue

        if isinstance(response, JSONResponse):
            entry["proxy_response"] = json.dumps(json.loads(response.body.decode("utf-8")), indent=2)
        elif isinstance(response, StreamingResponse):
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            entry["proxy_response"] = "\n\n".join(repr(c) for c in chunks)
        entry["logs"] = "\n".join(log_line(r) for r in c.records)

        #entry["status_code"] = response.status_code,
        #entry["headers"] = dict(response.headers),
        #entry["body"] = response.json()
        #response.raise_for_status()

        results.append(entry)

    loader = jinja2.FileSystemLoader(Path(__file__).parent)
    env = jinja2.Environment(loader=loader)
    template = env.get_template("report_template.html")
    rendered = template.render(output_dir=str(output_dir), results=results)
    out_file = output_dir / "report.html"
    out_file.write_text(rendered, encoding="utf-8")
    print(f"Report written: file://{out_file.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())
