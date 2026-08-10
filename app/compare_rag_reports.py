"""比较两份已有 RAG 评测报告；不会调用 Embedding 或 Chat API。"""

import argparse
import json
from pathlib import Path

if __package__:
    from app.evaluation_comparison import (
        compare_evaluation_report_files,
        write_comparison_report,
    )
else:
    from evaluation_comparison import (
        compare_evaluation_report_files,
        write_comparison_report,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="比较两份 RAG JSON 评测报告。")
    parser.add_argument("baseline", type=Path, help="基线评测报告路径。")
    parser.add_argument("current", type=Path, help="当前评测报告路径。")
    parser.add_argument("--output", type=Path, help="可选的对比 JSON 保存路径。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        comparison = compare_evaluation_report_files(args.baseline, args.current)
        if args.output is not None:
            write_comparison_report(comparison, args.output)
    except Exception as error:
        print(f"评测报告对比失败：{error}")
        return 2

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if args.output is not None:
        print(f"\n对比报告：{args.output}")
    return 1 if comparison["newly_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
