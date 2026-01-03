#!/usr/bin/env python3
"""
Anchor Sigil signatures to Web2 platforms for timestamp proof

This creates a "chain of custody" by posting signatures to publicly-verifiable
platforms like Twitter and GitHub. These platforms serve as timestamp oracles
that courts understand and trust.

Usage:
    python -m cli.anchor SIGNATURE_FILE --twitter <tweet_url>
    python -m cli.anchor SIGNATURE_FILE --github <issue_url>
    python -m cli.anchor SIGNATURE_FILE --list

Example:
    # 1. Create signature
    python -m cli.extract video.mp4 --sign

    # 2. Post signature to Twitter (manually or via API)
    # Tweet: "Claiming ownership of video hash abc123... Signature: <paste>"

    # 3. Anchor the tweet URL to the signature
    python -m cli.anchor video.mp4.signature.json --twitter https://twitter.com/user/status/123

    # 4. Optionally: Manually archive at https://web.archive.org for backup
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def add_anchor(signature_path: Path, anchor_type: str, url: str, metadata: Optional[Dict] = None) -> Dict:
    """
    Add an anchor to a signature file.

    Args:
        signature_path: Path to signature.json
        anchor_type: Type of anchor (twitter, github, archive, etc.)
        url: URL of the anchor
        metadata: Optional additional metadata

    Returns:
        Updated signature document
    """
    # Load existing signature
    with signature_path.open('r') as f:
        sig_doc = json.load(f)

    # Ensure anchors array exists
    if 'anchors' not in sig_doc:
        sig_doc['anchors'] = []

    # Create new anchor
    new_anchor = {
        'type': anchor_type,
        'url': url,
        'anchored_at': datetime.utcnow().isoformat() + 'Z'
    }

    if metadata:
        new_anchor['metadata'] = metadata

    # Check for duplicates
    for existing_anchor in sig_doc['anchors']:
        if existing_anchor['url'] == url:
            print("⚠️  Warning: This URL is already anchored in the signature file")
            return sig_doc

    # Add anchor
    sig_doc['anchors'].append(new_anchor)

    # Write back to file
    with signature_path.open('w') as f:
        json.dump(sig_doc, f, indent=2)

    return sig_doc


# Archive.org functionality removed - not needed for legal timestamp proof
# Twitter/GitHub timestamps are sufficient and legally stronger


def cmd_twitter(args):
    """Anchor to Twitter tweet"""
    sig_path = Path(args.signature_file)

    if not sig_path.exists():
        print(f"❌ Signature file not found: {sig_path}", file=sys.stderr)
        sys.exit(1)

    tweet_url = args.twitter
    if not tweet_url.startswith('http'):
        print(f"❌ Invalid Twitter URL: {tweet_url}", file=sys.stderr)
        sys.exit(1)

    try:
        print("🔗 Anchoring signature to Twitter...")
        print(f"   Signature: {sig_path.name}")
        print(f"   Tweet: {tweet_url}")

        # Add anchor
        sig_doc = add_anchor(sig_path, 'twitter', tweet_url)

        print("\n✅ Signature anchored successfully")
        print(f"   Total anchors: {len(sig_doc.get('anchors', []))}")

        if args.verbose:
            print("\n💡 What this proves:")
            print("   1. Your Ed25519 signature proves you possessed the hash")
            print("   2. Twitter's timestamp proves when you made the claim")
            print("   3. Together: You owned this hash at this specific time")
            print("   4. In court: Burden of proof shifts to the defendant")
            print("\n💡 Optional: Manually archive your tweet at web.archive.org for backup")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_github(args):
    """Anchor to GitHub issue/gist/commit"""
    sig_path = Path(args.signature_file)

    if not sig_path.exists():
        print(f"❌ Signature file not found: {sig_path}", file=sys.stderr)
        sys.exit(1)

    github_url = args.github
    if not ('github.com' in github_url):
        print(f"❌ Invalid GitHub URL: {github_url}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"🔗 Anchoring signature to GitHub...")
        print(f"   Signature: {sig_path.name}")
        print(f"   GitHub: {github_url}")

        # Add anchor
        sig_doc = add_anchor(sig_path, 'github', github_url)

        print(f"\n✅ Signature anchored successfully")
        print(f"   Total anchors: {len(sig_doc.get('anchors', []))}")

        if args.verbose:
            print(f"\n💡 What this proves:")
            print(f"   1. Your Ed25519 signature proves you possessed the hash")
            print(f"   2. GitHub's commit/issue timestamp proves when you made the claim")
            print(f"   3. Git history provides permanent, tamper-proof record")
            print(f"\n💡 Optional: Manually archive at web.archive.org for backup")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# cmd_archive removed - Archive.org integration is no longer automatic
# Users can manually archive their Twitter/GitHub posts if desired


def cmd_list(args):
    """List all anchors in a signature file"""
    sig_path = Path(args.signature_file)

    if not sig_path.exists():
        print(f"❌ Signature file not found: {sig_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with sig_path.open('r') as f:
            sig_doc = json.load(f)

        anchors = sig_doc.get('anchors', [])

        if not anchors:
            print(f"📋 No anchors found in signature file")
            print(f"\n💡 Add anchors with:")
            print(f"   python -m cli.anchor {sig_path} --twitter <tweet_url>")
            print(f"   python -m cli.anchor {sig_path} --github <issue_url>")
            sys.exit(0)

        print(f"📋 Anchors for: {sig_path.name}")
        print(f"   Signature: {sig_doc['claim']['hash_hex'][:16]}...")
        print(f"   Key ID: {sig_doc['proof']['key_id']}")
        print(f"   Signed: {sig_doc['proof']['signed_at']}")
        print(f"\n🔗 Timestamp Anchors ({len(anchors)}):")

        for i, anchor in enumerate(anchors, 1):
            anchor_type = anchor.get('type', 'unknown').capitalize()
            anchor_url = anchor.get('url', 'No URL')
            anchored_at = anchor.get('anchored_at', 'Unknown')
            print(f"   {i}. {anchor_type}")
            print(f"      URL: {anchor_url}")
            print(f"      Anchored: {anchored_at}")

        if args.verbose:
            print(f"\n💡 Legal Value:")
            print(f"   These anchors prove your signature existed at specific times")
            print(f"   Web2 platforms (Twitter, GitHub) are legally recognized timestamp oracles")
            print(f"   Courts understand these better than blockchain timestamps")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Anchor Sigil signatures to Web2 platforms for timestamp proof",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Why Anchoring?
  Cryptographic signatures prove WHO signed a hash, but not WHEN.
  By posting signatures to Twitter/GitHub and archiving them,
  you create legally-recognized timestamp proof that courts understand.

Examples:
  # Anchor to Twitter
  python -m cli.anchor video.mp4.signature.json --twitter https://twitter.com/user/status/123

  # Anchor to GitHub gist
  python -m cli.anchor video.mp4.signature.json --github https://gist.github.com/user/abc123

  # List anchors
  python -m cli.anchor video.mp4.signature.json --list

Note:
  Twitter/GitHub timestamps are legally sufficient.
  For backup, manually archive your post at https://web.archive.org
        """
    )

    parser.add_argument(
        'signature_file',
        type=str,
        help='Path to signature.json file'
    )
    parser.add_argument(
        '--twitter',
        type=str,
        help='Twitter/X tweet URL containing signature'
    )
    parser.add_argument(
        '--github',
        type=str,
        help='GitHub issue/gist/commit URL containing signature'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all anchors in signature file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Route to appropriate command
    if args.list:
        cmd_list(args)
    elif args.twitter:
        cmd_twitter(args)
    elif args.github:
        cmd_github(args)
    else:
        print(f"❌ You must specify an anchoring method:", file=sys.stderr)
        print(f"   --twitter <url>  Anchor to Twitter tweet", file=sys.stderr)
        print(f"   --github <url>   Anchor to GitHub issue/gist", file=sys.stderr)
        print(f"   --list           List existing anchors", file=sys.stderr)
        print(f"\n   Use --help for examples", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
