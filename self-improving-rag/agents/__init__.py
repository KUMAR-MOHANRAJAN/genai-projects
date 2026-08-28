"""agents/ — The self-improving loop (LangGraph nodes + optimizer).

Modules:
  evaluator.py   — LLM judges + unified score + gate decision
  diagnoser.py   — Rule-cascade failure classification (F-01..F-05)
  improver.py    — Variant playbook + config deltas (propose-only)
  graph.py       — LangGraph StateGraph wiring + conditional routing
  optimizer.py   — Outer loop: baseline → trials → stop conditions → report
"""
