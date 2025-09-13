#!/usr/bin/env python
"""
Extract text/markdown from a PDF using Docling (already in requirements).

Usage:
  python scripts/extract_pdf.py "Docs/Team_Leo.pdf" --out "Docs/Team_Leo.md"

It will:
- Convert PDF to a Docling document
- Print plain text to stdout
- Optionally write Markdown to the given --out path
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter, ConversionConfig


def extract_pdf_to_markdown(pdf_path: Path) -> str:
    config = ConversionConfig()
    converter = DocumentConverter(config=config)
    result = converter.convert(pdf_path)
    # Export as markdown (keeps tables and simple layout)
    md = result.document.export_to_markdown()
    return md


def markdown_to_plain_text(md: str) -> str:
    """Very simple markdown-to-text for console display."""
    try:
        # Prefer a minimal dependency-free approach
        import re
        text = md
        # Remove code fences
        text = re.sub(r"```[\s\S]*?```", "\n", text)
        # Strip markdown headers/formatting
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"`(.*?)`", r"\1", text)
        text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)  # links
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception:
        return md


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract text/markdown from a PDF using Docling")
    p.add_argument("pdf", type=str, help="Path to the input PDF")
    p.add_argument("--out", type=str, default=None, help="Optional output Markdown file path")
    args = p.parse_args(argv)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        return 1

    try:
        md = extract_pdf_to_markdown(pdf_path)
    except Exception as e:
        print("Failed to convert PDF via Docling.", file=sys.stderr)
        print(f"Reason: {e}", file=sys.stderr)
        return 2

    # Print a plain-text view to stdout for quick reading
    text = markdown_to_plain_text(md)
    print(text)

    # Optionally write full markdown
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"\n[Saved Markdown to] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
