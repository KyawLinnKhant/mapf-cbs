"""Language interface: natural language command → zone assignments via Ollama."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from .zones import available_zones, ZONE_ALIASES


_SYSTEM_PROMPT = """\
You are a warehouse robot fleet coordinator.
Parse the command and output a JSON with two fields:
- "explicit": list the robots with their specific zones
- "rest_zone": the zone for ALL remaining robots not listed in explicit

Respond ONLY with valid JSON — no explanation, no markdown.
Format: {{"explicit": [{{"agent": <int>, "zone": "<zone>"}},...], "rest_zone": "<zone>"}}

Rules:
- "rest_zone" = zone for every robot NOT in "explicit"
- If ALL robots go to the same zone: explicit=[], rest_zone=that zone
- Only use zone names from the available list

Examples:
Command: "Agents 0,1,2 to loading_bay, rest to charging"
Response: {{"explicit": [{{"agent":0,"zone":"loading_bay"}},{{"agent":1,"zone":"loading_bay"}},{{"agent":2,"zone":"loading_bay"}}], "rest_zone": "charging"}}

Command: "All agents to inspection"
Response: {{"explicit": [], "rest_zone": "inspection"}}

Command: "Half to storage_a, half to dispatch" (robots 0-5 explicit, 6-11 dispatch)
Response: {{"explicit": [{{"agent":0,"zone":"storage_a"}},{{"agent":1,"zone":"storage_a"}},{{"agent":2,"zone":"storage_a"}},{{"agent":3,"zone":"storage_a"}},{{"agent":4,"zone":"storage_a"}},{{"agent":5,"zone":"storage_a"}}], "rest_zone": "dispatch"}}
"""


def _build_prompt(command: str, n_agents: int, level: str) -> str:
    zones = available_zones(level)
    aliases = ", ".join(f'"{k}" = {v}' for k, v in ZONE_ALIASES.items() if v in zones)
    return (
        f"Available zones: {', '.join(zones)}\n"
        f"Aliases: {aliases}\n"
        f"Total robots: {n_agents} (IDs 0 to {n_agents-1})\n\n"
        f"Command: {command}"
    )


def _parse_json_response(text: str, n_agents: int, level: str) -> Dict[int, str]:
    """Extract JSON from LLM response and validate.

    Handles two formats:
    1. New compact: {"default": "zone", "overrides": {"0": "zone", ...}}
    2. Legacy full: {"assignments": [{"agent": 0, "zone": "..."}, ...]}
    """
    # Strip markdown code fences and tidy whitespace
    text = re.sub(r"```(?:json)?", "", text).strip()

    valid_zones = set(available_zones(level))
    first_zone  = available_zones(level)[0]

    def _sanitise(z: str) -> str:
        z = str(z).lower().strip()
        return z if z in valid_zones else first_zone

    try:
        data = json.loads(text)

        # Format 1: explicit + rest_zone
        if "rest_zone" in data:
            rest = _sanitise(data.get("rest_zone", first_zone))
            assignments = {i: rest for i in range(n_agents)}
            for entry in (data.get("explicit") or []):
                try:
                    assignments[int(entry["agent"])] = _sanitise(entry["zone"])
                except (KeyError, TypeError, ValueError):
                    pass
            return assignments

        # Format 2: default + overrides
        if "default" in data:
            default_zone = _sanitise(data.get("default", first_zone))
            assignments  = {i: default_zone for i in range(n_agents)}
            for k, v in (data.get("overrides") or {}).items():
                try:
                    assignments[int(k)] = _sanitise(v)
                except (ValueError, TypeError):
                    pass
            return assignments

        # Format 3: legacy assignments list
        if "assignments" in data:
            assignments = {int(a["agent"]): _sanitise(a["zone"])
                           for a in data["assignments"]}
            for i in range(n_agents):
                if i not in assignments:
                    assignments[i] = first_zone
            return assignments

    except Exception:
        pass

    return _rule_based_parse(text, n_agents, level)


def parse_command(
    command: str,
    n_agents: int,
    level: str,
    model: str = "qwen2.5:3b",
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
