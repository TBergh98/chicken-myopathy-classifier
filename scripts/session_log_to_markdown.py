"""
Convert a session log JSON into a clean Markdown response.

The script keeps only user-facing content, preserves the original order,
and skips system/tool entries and empty assistant messages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox


def _collect_user_facing_blocks(messages: Iterable[dict[str, Any]]) -> List[str]:
    blocks: List[str] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role != "assistant":
            continue
        if not isinstance(content, str) or not content.strip():
            continue

        blocks.append(f"### Assistant\n\n{content.strip()}")

    return blocks


def session_log_to_markdown(payload: dict[str, Any]) -> str:
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("Expected 'messages' to be a list")

    blocks = _collect_user_facing_blocks(messages)
    session_id = payload.get("session_id", "unknown-session")
    model_name = payload.get("model_name", "unknown-model")

    header = [
        f"# Session {session_id}",
        "",
        f"- Model: {model_name}",
    ]

    if not blocks:
        return "\n".join(header + ["", "No user-facing content found."])

    return "\n".join(header + ["", "---", ""] + blocks)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    session_logs_dir = workspace_root / "session_logs"
    docs_dir = workspace_root / "docs"

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    input_json_path = filedialog.askopenfilename(
        title="Select session log JSON",
        initialdir=session_logs_dir,
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    if not input_json_path:
        messagebox.showinfo("Session log to Markdown", "No JSON selected. Operation cancelled.")
        root.destroy()
        return

    output_name = simpledialog.askstring(
        "Markdown name",
        "How do you want to name the final Markdown file?",
        parent=root,
    )
    if not output_name or not output_name.strip():
        messagebox.showinfo("Session log to Markdown", "No output name provided. Operation cancelled.")
        root.destroy()
        return

    output_name = output_name.strip()
    if not output_name.lower().endswith(".md"):
        output_name += ".md"

    docs_dir.mkdir(parents=True, exist_ok=True)
    output_path = docs_dir / output_name

    payload = json.loads(Path(input_json_path).read_text(encoding="utf-8"))
    markdown = session_log_to_markdown(payload)

    output_path.write_text(markdown, encoding="utf-8")

    messagebox.showinfo(
        "Session log to Markdown",
        f"Markdown saved successfully to:\n{output_path}",
    )
    root.destroy()


if __name__ == "__main__":
    main()