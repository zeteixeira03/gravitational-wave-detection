# Session rules for this repository

Read `ROADMAP.md` first. It holds the plan, the decisions and the current status.

## HARD RULE: orchestrator / writer split (user decision, 2026-10-17)

This rule applies in every session. It is the user's explicit, standing request to
use subagents, so it overrides any default that says "do not spawn subagents unless
the user asks".

1. The main session is the **orchestrator**. It reads, plans, writes specs, reviews
   and commits. It does not write implementation code itself.
2. All implementation code (modules, tests, scripts, kernels) is written by
   **writer subagents** (Agent tool, `model: "sonnet"`, or `"haiku"` for small
   mechanical edits).
3. Each writer gets a **self-contained spec**: the goal, the files to create or edit
   by path, the conventions it must follow (units and conventions from the module
   docstrings it touches), the tests it must add, and the command that must pass
   (`python -m pytest gwsciml/tests -q`). Do not assume the writer has seen this
   conversation.
4. The orchestrator **reviews every writer diff before each commit**: read the
   diff, run the tests, check the physics and units, and send corrections back to
   the writer (SendMessage) rather than fixing the code itself.
5. Running experiments and jobs (scripts, Kaggle kernels, analysis) is orchestration
   and may be done by the main session.

If a session cannot follow this rule (for example, the Agent tool is unavailable),
it stops and tells the user. It does not fall back to writing the code itself.
