#!/usr/bin/env python3
"""
Generate switcher.json for PyData Sphinx Theme version selector.

This script reads git tags and the current branch to create a JSON file
that the version switcher dropdown can use.
"""

import json
import subprocess
import sys
from pathlib import Path


def get_git_tags():
    """Get all git tags matching version pattern v*.*.* """
    try:
        result = subprocess.run(
            ["git", "tag", "-l", "v*.*.*"],
            capture_output=True,
            text=True,
            check=True
        )
        tags = [tag.strip() for tag in result.stdout.split('\n') if tag.strip()]
        # Sort tags by version (latest first)
        tags.sort(reverse=True, key=lambda x: [int(n) if n.isdigit() else n for n in x.replace('v', '').split('.')])
        return tags
    except subprocess.CalledProcessError as e:
        print(f"Error getting git tags: {e}", file=sys.stderr)
        return []


def get_git_branches():
    """Get main/develop branches if they exist"""
    try:
        result = subprocess.run(
            ["git", "branch", "-r"],
            capture_output=True,
            text=True,
            check=True
        )
        branches = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if 'origin/main' in line:
                branches.append('main')
            elif 'origin/develop' in line:
                branches.append('develop')
        return branches
    except subprocess.CalledProcessError as e:
        print(f"Error getting git branches: {e}", file=sys.stderr)
        return []


def generate_switcher_json(base_url="https://josephdong1000.github.io/neurodent"):
    """Generate switcher.json with all versions"""

    tags = get_git_tags()
    branches = get_git_branches()

    if not tags and not branches:
        print("Warning: No tags or branches found", file=sys.stderr)
        return []

    switcher_data = []

    # Add branches first
    if 'main' in branches:
        switcher_data.append({
            "name": "main (dev)",
            "version": "main",
            "url": f"{base_url}/main/"
        })

    if 'develop' in branches:
        switcher_data.append({
            "name": "develop (unstable)",
            "version": "develop",
            "url": f"{base_url}/develop/"
        })

    # Add tags (latest first gets "preferred: true")
    for i, tag in enumerate(tags):
        version_entry = {
            "name": tag if i != 0 else f"{tag} (stable)",
            "version": tag,
            "url": f"{base_url}/{tag}/"
        }

        # Mark the latest version as preferred
        if i == 0:
            version_entry["preferred"] = True

        switcher_data.append(version_entry)

    return switcher_data


def main():
    """Generate and save switcher.json"""

    # Generate the switcher data
    switcher_data = generate_switcher_json()

    if not switcher_data:
        print("Error: No version data generated", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    # This script should be run after sphinx-multiversion build
    # Save to the root of the build output
    output_dir = Path(__file__).parent / "_build" / "html"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "switcher.json"

    # Write the JSON file
    with open(output_file, 'w') as f:
        json.dump(switcher_data, f, indent=2)

    print(f"Generated {output_file}")
    print(f"Versions included: {len(switcher_data)}")
    for entry in switcher_data:
        preferred = " (preferred)" if entry.get("preferred") else ""
        print(f"  - {entry['name']}{preferred}")


if __name__ == "__main__":
    main()
