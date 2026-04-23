"""Tri-way paper-revision comparison driven by xtragpt-paper-revision-skill.

Uses the skill's actual YAML (scaffolded by `npx xtragpt-paper-revision-skill
init`) as the single source of truth for the prompt template. Fires three
backends on the same task in parallel:

  - XtraGPT-7B  : local vLLM at  http://127.0.0.1:8088/v1
  - Qwen2.5-7B  : local vLLM at  http://127.0.0.1:8089/v1  (same base model, no revision fine-tuning)
  - GLM-5.1     : z.ai open.bigmodel.cn  (much larger general-purpose model)

Writes a side-by-side Markdown report to outputs/before_after.md.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from openai import OpenAI

HERE = pathlib.Path(__file__).parent
SKILL_YAML = HERE / "openclaw" / "skills" / "skill.xtragpt-paper-revision-skill.yaml"
PAPER_JSON = HERE / "paper_sections.json"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

ZAI_KEY = os.environ.get("ZAI_API_KEY") or os.environ.get("OPENCLAW_ZAI_KEY", "")
if not ZAI_KEY:
    raise SystemExit(
        "Set ZAI_API_KEY before running. On a machine that already has OpenClaw "
        "configured, the key is at ~/.openclaw/agents/main/agent/auth-profiles.json "
        "under profiles['zai:default'].key."
    )

BACKENDS = {
    "XtraGPT-7B (local, fine-tuned on paper revision)": {
        "base_url": "http://127.0.0.1:8088/v1",
        "api_key": "EMPTY",
        "model": "Xtra-Computing/XtraGPT-7B",
        "tag": "xtragpt",
    },
    "Qwen2.5-7B-Instruct (local, XtraGPT's base model, no revision fine-tune)": {
        "base_url": "http://127.0.0.1:8089/v1",
        "api_key": "EMPTY",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "tag": "qwen",
    },
    "GLM-5.1 (z.ai, general-purpose ~hundreds-of-B class)": {
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "api_key": ZAI_KEY,
        "model": "GLM-5.1",
        "tag": "glm5",
        # GLM-5.1 is a reasoning model; without this it burns all tokens on CoT
        # before producing any assistant content.
        "extra_body": {"thinking": {"type": "disabled"}},
    },
}


def load_skill():
    doc = yaml.safe_load(SKILL_YAML.read_text())
    return doc["skill"], doc.get("prompt_template") or doc["prompt_template"]


def render_prompt(template: str, *, paper_content: str, selected_content: str,
                  question: str, section_name: str = "", venue_style: str = "",
                  reviewer_feedback: str = "(none)") -> str:
    return template.format(
        paper_content=paper_content,
        selected_content=selected_content,
        question=question,
        section_name=section_name,
        venue_style=venue_style,
        reviewer_feedback=reviewer_feedback,
    )


def call_backend(name: str, cfg: dict, prompt: str, temperature: float = 0.1,
                 max_tokens: int = 1024) -> dict:
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=180.0)
    t0 = time.time()
    err = None
    kwargs = dict(
        model=cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if cfg.get("extra_body"):
        kwargs["extra_body"] = cfg["extra_body"]
    try:
        rsp = client.chat.completions.create(**kwargs)
        text = rsp.choices[0].message.content or ""
        usage = rsp.usage.model_dump() if rsp.usage else {}
    except Exception as exc:  # pragma: no cover - demo-only
        text = ""
        usage = {}
        err = f"{type(exc).__name__}: {exc}"
    dt = time.time() - t0
    return {
        "name": name,
        "tag": cfg["tag"],
        "model": cfg["model"],
        "latency_s": round(dt, 2),
        "usage": usage,
        "output": text.strip(),
        "error": err,
    }


def fan_out(prompt: str) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(BACKENDS)) as pool:
        futs = [pool.submit(call_backend, n, c, prompt) for n, c in BACKENDS.items()]
        return [f.result() for f in as_completed(futs)]


def render_markdown(paper: dict, tasks: list[dict]) -> str:
    md = ["# XtraGPT paper-revision skill — three-way comparison", ""]
    md.append(f"**Paper:** {paper['title']}  ")
    md.append(f"**Source:** {paper['source_pdf']}  ")
    md.append(f"**Skill:** `xtragpt-paper-revision-skill` (prompt template loaded verbatim from its YAML)")
    md.append("")
    md.append("All three backends receive the identical skill-rendered prompt. "
              "Temperature = 0.1, max_tokens = 1024.")
    md.append("")
    for task in tasks:
        md.append(f"## {task['label']}")
        md.append("")
        md.append(f"**Instruction:** _{task['instruction']}_")
        md.append("")
        md.append("### Original")
        md.append("")
        md.append("> " + task["selected"].replace("\n", "\n> "))
        md.append("")
        md.append("### Revisions")
        md.append("")
        for r in task["results"]:
            md.append(f"#### {r['name']}")
            md.append(f"_latency: {r['latency_s']}s · output tokens: "
                      f"{r['usage'].get('completion_tokens', '?')} "
                      f"· prompt tokens: {r['usage'].get('prompt_tokens', '?')}_")
            md.append("")
            if r["error"]:
                md.append(f"> error: `{r['error']}`")
            else:
                md.append("> " + r["output"].replace("\n", "\n> "))
            md.append("")
        md.append("---")
        md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "before_after.md"))
    ap.add_argument("--also-json", default=str(OUT_DIR / "before_after.json"))
    args = ap.parse_args()

    skill, template = load_skill()
    paper = json.loads(PAPER_JSON.read_text())

    paper_content = (paper["full_paper_content"])[:12000]

    tasks_spec = [
        {
            "label": "Task 1 — Abstract: make more concise",
            "selected": paper["abstract"],
            "instruction": "Make this abstract more concise without losing the key contributions "
                           "or changing any numerical claims.",
            "section_name": "abstract",
            "venue_style": "EMNLP",
        },
        {
            "label": "Task 2 — Introduction paragraph: reduce overclaim, strengthen motivation",
            "selected": paper["intro_first_block"],
            "instruction": "Rewrite this introduction paragraph to reduce overclaim, strengthen "
                           "motivation, and keep the citations intact. Do not invent new citations.",
            "section_name": "introduction",
            "venue_style": "EMNLP",
        },
    ]

    tasks_out = []
    for spec in tasks_spec:
        prompt = render_prompt(
            template,
            paper_content=paper_content,
            selected_content=spec["selected"],
            question=spec["instruction"],
            section_name=spec["section_name"],
            venue_style=spec["venue_style"],
        )
        print(f"\n=== {spec['label']} ===")
        print(f"(prompt chars: {len(prompt)})")
        results = fan_out(prompt)
        results.sort(key=lambda r: ["xtragpt", "qwen", "glm5"].index(r["tag"]))
        for r in results:
            status = "OK " if not r["error"] else "ERR"
            print(f"  {status} {r['name'][:60]:60s} {r['latency_s']:6.2f}s "
                  f"{r['usage'].get('completion_tokens', '?')}t")
        tasks_out.append({**spec, "results": results})

    md = render_markdown(paper, tasks_out)
    pathlib.Path(args.out).write_text(md)
    pathlib.Path(args.also_json).write_text(
        json.dumps({"paper": {k: paper.get(k) for k in ("title", "source_pdf")},
                    "tasks": tasks_out}, indent=2)
    )
    print(f"\nwrote {args.out}")
    print(f"wrote {args.also_json}")


if __name__ == "__main__":
    main()
