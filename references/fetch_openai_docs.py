#!/usr/bin/env python3
"""
Fetch OpenAI API documentation for analysis.
Downloads the complete OpenAPI spec and Python SDK documentation.
"""

import json
from datetime import datetime
from pathlib import Path

import requests
import yaml


def fetch_file(url: str, output_path: Path, description: str) -> bool:
    """Fetch a file from URL and save it."""
    print(f"Fetching {description}...")
    print(f"  URL: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"  ✓ Saved to: {output_path}")
        print(f"  Size: {len(response.text):,} bytes")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Main function to fetch all documentation."""
    # Setup output directory
    output_dir = Path("openai-api-docs")
    output_dir.mkdir(exist_ok=True)

    # Create metadata file
    metadata = {"fetched_at": datetime.now().isoformat(), "files": []}

    print("OpenAI API Documentation Fetcher")
    print("=" * 50)
    print()

    # Define files to fetch
    files_to_fetch = [
        {
            "url": "https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml",
            "output": output_dir / "openapi-spec.yml",
            "description": "OpenAPI Specification (YAML)",
        },
        {
            "url": "https://raw.githubusercontent.com/openai/openai-python/main/api.md",
            "output": output_dir / "python-sdk-api.md",
            "description": "Python SDK API Reference",
        },
        {
            "url": "https://raw.githubusercontent.com/openai/openai-python/main/README.md",
            "output": output_dir / "python-sdk-readme.md",
            "description": "Python SDK README",
        },
    ]

    # Fetch each file
    success_count = 0
    for file_info in files_to_fetch:
        if fetch_file(file_info["url"], file_info["output"], file_info["description"]):
            success_count += 1
            metadata["files"].append(
                {"path": str(file_info["output"]), "url": file_info["url"], "description": file_info["description"]}
            )

    print()
    print(f"Successfully fetched {success_count}/{len(files_to_fetch)} files")

    # Save metadata
    metadata_path = output_dir / "fetch-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")
    # Try to parse and get basic stats from OpenAPI spec
    try:
        print("\nAnalyzing OpenAPI spec...")
        with open(output_dir / "openapi-spec.yml") as f:
            spec = yaml.safe_load(f)
        if spec:
            paths = spec.get("paths", {})
            responses_endpoints = [p for p in paths if "responses" in p]
            print(f"  Total endpoints: {len(paths)}")
            print(f"  Responses API endpoints: {len(responses_endpoints)}")
            # List Responses API endpoints
            if responses_endpoints:
                print("\n  Responses API endpoints found:")
                for endpoint in responses_endpoints:
                    methods = list(paths[endpoint].keys())
                    print(f"    {endpoint}: {', '.join(methods).upper()}")
    except Exception as e:
        print(f"  Could not analyze OpenAPI spec: {e}")


if __name__ == "__main__":
    main()
