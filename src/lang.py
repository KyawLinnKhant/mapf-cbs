"""Language interface: natural language command → zone assignments via Ollama."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from .zones import available_zones, ZONE_ALIASES


_SYSTEM_PROMPT = """\
You are a warehouse robot fleet coordinator.
Parse the user's command and assign each robot to a named zone.

Respond ONLY with a valid JSON object — no explanation, no markdown, no extra text.
Format:
{{"assignments": [{{"agent": <int>, "zone": "<zone_name>"}}, ...]}}

Rules:
- Assign ALL robots (IDs 0 to {n_agents_minus_1})
- "the rest" / "others" / "remaining" = all robots not yet assigned
- Only use zone names from the available list
- If a command is ambiguous, distribute evenly across mentioned zones
"""


def _build_prompt(command: str, n_agents: int, level: str) -> str:
    zones = available_zones(level)
    aliases = ", ".join(f'"{k}" = {v}' for k, v in ZONE_ALIASES.items() if v in zones)
    return (
        f"Available zones: {', '.join(zones)}\n"
        f"Aliases: {aliases}\n"
        f"Number of robots: {n_agents} (IDs 0–{n_agents-1})\n\n"
        f"Command: {command}"
    )


def _parse_json_response(text: str, n_agents: int, level: str) -> Dict[int, str]:
    """Extract JSON from LLM response and validate."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    try:
        data = json.loads(text)
        assignments = {int(a["agent"]): a["zone"] for a in data["assignments"]}
        zones = set(available_zones(level))
        # Fill in any missing agents with first zone
        for i in range(n_agents):
            if i not in assignments:
                assignments[i] = available_zones(level)[0]
        return assignments
    except Exception:
        # Fallback: distribute all agents across first zone
        return {i: available_zones(level)[0] for i in range(n_agents)}


def parse_command(
    command: str,
    n_agents: int,
    level: str,
    model: str = "kimi-k2.6:cloud",
) -> Dict[int, str]:
    """
    Parse a natural language command into {agent_id: zone_name}.

    Uses Ollama with the specified model. Falls back to rule-based
    parsing if Ollama is unavailable.
    """
    try:
        import ollama
        system = _SYSTEM_PROMPT.format(n_agents_minus_1=n_agents - 1)
        user   = _build_prompt(command, n_agents, level)

        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={"temperature": 0.0},
        )
        raw = resp["message"]["content"]
        return _parse_json_response(raw, n_agents, level)

    except Exception as e:
        print(f"  [lang] Ollama unavailable ({e}), using rule-based fallback.")
        return _rule_based_parse(command, n_agents, level)


def _rule_based_parse(command: str, n_agents: int, level: str) -> Dict[int, str]:
    """
    Simple regex fallback when Ollama is unavailable.
    Handles patterns like 'send agent 0 to loading_bay' or 'all to charging'.
    """
    zones = available_zones(level)
    cmd   = command.lower()

    # Map any zone alias
    for alias, zone in ZONE_ALIASES.items():
        cmd = cmd.replace(alias, zone)

    assignments: Dict[int, str] = {}

    # Find explicit agent mentions: "agent 0", "robot 2", "#3"
    for zone in zones:
        if zone in cmd:
            # Find agent ids near this zone mention
            idx = cmd.find(zone)
            snippet = cmd[max(0, idx-40):idx+40]
            found = re.findall(r"\b(\d+)\b", snippet)
            for a in found:
                aid = int(a)
                if 0 <= aid < n_agents:
                    assignments[aid] = zone

    # "all" / "everyone" → first zone found or last mentioned
    if not assignments:
        for zone in zones:
            if zone in cmd:
                assignments = {i: zone for i in range(n_agents)}
                break

    # Fill remaining agents
    default_zone = zones[0]
    for i in range(n_agents):
        if i not in assignments:
            assignments[i] = default_zone

    return assignments
