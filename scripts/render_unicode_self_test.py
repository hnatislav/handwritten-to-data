from __future__ import annotations

import argparse

from src.analysis.ocr_experiment import render_unicode_self_test, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render known Unicode strings to verify OCR visualization font handling.")
    parser.add_argument("--output", default="outputs/debug/unicode_render_self_test.png")
    parser.add_argument("--font", default=None)
    parser.add_argument("--report", default="outputs/debug/unicode_render_self_test.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = render_unicode_self_test(args.output, font_path=args.font)
    write_json(args.report, report)
    print(f"output={report['output']}")
    print(f"font={report['font']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
