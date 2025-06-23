#!/usr/bin/env python3
"""Test harness for API conversion fixtures."""
import argparse
import asyncio
import datetime
import json
import sys
import traceback
from pathlib import Path

import httpx
import jinja2
from httpx import HTTPStatusError

from claude_code_proxy.config import load_config
from claude_code_proxy.converter_v2 import (anthropic_to_openai_request,
                                            openai_to_anthropic_response)
from claude_code_proxy.server import call_openai_responses_api
from claude_code_proxy.test_fixtures.fixtures import FIXTURES


async def main():
    parser = argparse.ArgumentParser(description="Run API conversion fixtures and generate HTML report")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("test_fixtures_output"))
    parser.add_argument(
        "--template", "-t", type=Path,
        default=Path(__file__).parent / "report_template.html"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    api_key = config.anthropic_api_key
    if not api_key:
        sys.exit("Missing ANTHROPIC_API_KEY")

    client = httpx.Client(timeout=60.0)
    anthropic_headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    anthropic_url = "https://api.anthropic.com/v1/messages"

    results = []
    for fixture in FIXTURES:
        entry = {
            "fixture": fixture,
            "errors": [],
            "anthropic_native": {},
            "openai_native": {},
            "anthropic_converted": {},
        }
        entry["anthropic_native"]["request"] = fixture.anthropic_request
        try:
            r = client.post(anthropic_url, headers=anthropic_headers, json=fixture.anthropic_request)
            entry["anthropic_native"]["raw_response"] = {
                "status_code": r.status_code,
                "headers": dict(r.headers),
                "body": r.text,
            }
            r.raise_for_status()
            entry["anthropic_native"]["response"] = r.json()
        except HTTPStatusError as e:
            # Ignore rate-limit (429) errors for Anthropic (out of credits)
            if e.response.status_code != 429:
                entry["anthropic_native"]["error_type"] = e.__class__.__name__
                entry["anthropic_native"]["error"] = str(e)
                entry["anthropic_native"]["traceback"] = traceback.format_exc()
        except Exception as e:
            entry["anthropic_native"]["error_type"] = e.__class__.__name__
            entry["anthropic_native"]["error"] = str(e)
            entry["anthropic_native"]["traceback"] = traceback.format_exc()
        if entry["anthropic_native"].get("error"):
            et = entry["anthropic_native"].get("error_type")
            msg = entry["anthropic_native"]["error"]
            entry["errors"].append(f"{et}: {msg}" if et else msg)

        entry["openai_native"]["request"] = fixture.openai_request
        try:
            r = await call_openai_responses_api(fixture.openai_request)
            entry["openai_native"]["raw_response"] = {
                "status_code": r.status_code,
                "headers": dict(r.headers),
                "body": r.text,
            }
            r.raise_for_status()
            entry["openai_native"]["response"] = r.json()
        except Exception as e:
            entry["openai_native"]["error_type"] = e.__class__.__name__
            entry["openai_native"]["error"] = str(e)
            entry["openai_native"]["traceback"] = traceback.format_exc()
        if entry["openai_native"].get("error"):
            et = entry["openai_native"].get("error_type")
            msg = entry["openai_native"]["error"]
            entry["errors"].append(f"{et}: {msg}" if et else msg)

        try:
            conv_req = anthropic_to_openai_request(fixture.anthropic_request)
            entry["anthropic_converted"]["converted_request"] = conv_req
            r = await call_openai_responses_api(conv_req)
            entry["anthropic_converted"]["raw_response"] = {
                "status_code": r.status_code,
                "headers": dict(r.headers),
                "body": r.text,
            }
            r.raise_for_status()
            entry["anthropic_converted"]["openai_response"] = r.json()
            entry["anthropic_converted"]["converted_response"] = (
                openai_to_anthropic_response(entry["anthropic_converted"]["openai_response"])
            )
        except Exception as e:
            entry["anthropic_converted"]["error_type"] = e.__class__.__name__
            entry["anthropic_converted"]["error"] = str(e)
            entry["anthropic_converted"]["traceback"] = traceback.format_exc()
        if entry["anthropic_converted"].get("error"):
            et = entry["anthropic_converted"].get("error_type")
            msg = entry["anthropic_converted"]["error"]
            entry["errors"].append(f"{et}: {msg}" if et else msg)

        results.append(entry)

    loader = jinja2.FileSystemLoader(args.template.parent)
    env = jinja2.Environment(loader=loader)
    template = env.get_template(args.template.name)
    rendered = template.render(
        test_time=datetime.datetime.now().isoformat(),
        total_tests=len(results),
        successful_tests=sum(1 for e in results if not e["errors"]),
        failed_tests=sum(1 for e in results if e["errors"]),
        big_model=config.big_model,
        small_model=config.small_model,
        output_dir=str(output_dir),
        results=results,
    )
    out_file = output_dir / "report.html"
    out_file.write_text(rendered, encoding="utf-8")
    print(f"Report written: file://{out_file.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())
