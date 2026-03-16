"""MCTS tree visualization component — Mermaid-based."""

from typing import Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

_STOP_WORDS = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "in",
    "for",
    "and",
    "with",
    "by",
    "after",
    "before",
    "from",
    "that",
    "this",
    "into",
    "on",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "ensure",
    "improve",
    "accuracy",
    "correctness",
    "step",
    "approach",
    "method",
    "process",
}


def _shorten_label(text: str, max_words: int = 3) -> str:
    """Shorten a modification description to a Terraform-style label.

    Prefixes: + (added), - (removed), ~ (changed).
    Keeps only the key noun phrases for a compact 2-4 word label.
    """
    clean = text.strip()
    prefix = "~"  # default: changed
    lower = clean.lower()

    # Detect action and strip verbose prefix
    for tag in ("(add)", "add "):
        if lower.startswith(tag):
            prefix = "+"
            clean = clean[len(tag) :].strip()
            break
    else:
        for tag in ("(delete)", "(remove)", "delete ", "remove "):
            if lower.startswith(tag):
                prefix = "-"
                clean = clean[len(tag) :].strip()
                break
        else:
            for tag in (
                "(modify)",
                "modify:",
                "modify ",
                "replace ",
                "change ",
                "introduce ",
                "introduced ",
            ):
                if lower.startswith(tag):
                    prefix = "~"
                    clean = clean[len(tag) :].strip()
                    break

    # Drop stop words and keep key terms
    words = clean.split()
    key_words = [w for w in words if w.lower().rstrip(".,;:") not in _STOP_WORDS]
    if not key_words:
        key_words = words[:max_words]

    # Title-case and limit
    label = " ".join(w.capitalize() for w in key_words[:max_words])
    return f"{prefix} {label}"


def _parse_tree(
    tree_data: Dict,
) -> Tuple[Dict[int, float], int, List[Tuple[int, int, bool, str]]]:
    """Parse processed_experience.json into nodes and edges.

    Returns (scores, best_round, edges) where edges are (parent, child, success,
    modification).  In a single optimization run the parent round number is always
    lower than the child's.  Mixed / restarted runs can produce backward edges
    (parent > child); we normalise them here so the tree always flows chronologically
    from earlier rounds to later rounds.
    """
    scores: Dict[int, float] = {}
    edges: List[Tuple[int, int, str]] = []

    for parent_str, parent_data in tree_data.items():
        parent_id = int(parent_str)
        scores[parent_id] = parent_data.get("score", 0)

        for child_str, child_info in parent_data.get("success", {}).items():
            child_id = int(child_str)
            scores[child_id] = child_info["score"]
            # Skip backward edges (parent > child) — artifacts of mixed/restarted runs
            if parent_id >= child_id:
                continue
            label = child_info.get("short_label") or _shorten_label(
                child_info.get("modification", "")
            )
            edges.append((parent_id, child_id, label))

        for child_str, child_info in parent_data.get("failure", {}).items():
            child_id = int(child_str)
            scores[child_id] = child_info["score"]
            if parent_id >= child_id:
                continue
            label = child_info.get("short_label") or _shorten_label(
                child_info.get("modification", "")
            )
            edges.append((parent_id, child_id, label))

    # Derive success from scores: green if child improved over parent
    edges_final: List[Tuple[int, int, bool, str]] = []
    for parent, child, label in edges:
        success = scores.get(child, 0) > scores.get(parent, 0)
        edges_final.append((parent, child, success, label))

    best_round = max(scores, key=scores.get) if scores else -1
    return scores, best_round, edges_final


def _generate_mcts_mermaid(
    scores: Dict[int, float],
    best_round: int,
    edges: List[Tuple[int, int, bool, str]],
) -> str:
    """Generate a Mermaid flowchart for the MCTS tree."""
    lines = ["graph TD"]

    # Style definitions
    lines.append(
        "    classDef best fill:#FFF3E0,stroke:#FF6F00,color:#E65100,"
        "stroke-width:3px,font-weight:bold"
    )
    lines.append("    classDef normal fill:#E3F2FD,stroke:#0091EA,color:#0D47A1")
    lines.append(
        "    classDef root fill:#E8EAF6,stroke:#283593,color:#1A237E,font-weight:bold"
    )

    # Collect nodes that appear as children — any node NOT a child is a root
    child_set = {child for _, child, _, _ in edges}

    # Include round 1 (template) as the root if not already present
    if 1 not in scores and scores:
        scores[1] = 0.0  # placeholder; template has no eval score in tree data

    # Determine the root node (R0 if present, else R1)
    root_id = 0 if 0 in scores else 1

    # Define all nodes (sorted so earlier rounds appear first → Mermaid puts them higher)
    for round_id, score in sorted(scores.items()):
        node_id = f"R{round_id}"
        if round_id == 0:
            label = f"R0 ({score:.1%}) Eve App"
        elif score > 0:
            label = f"R{round_id} ({score:.1%})"
        else:
            label = f"R{round_id} (template)" if round_id == 1 else f"R{round_id} (0%)"
        lines.append(f'    {node_id}["{label}"]')

        if round_id == best_round:
            cls = "best"
        elif round_id not in child_set:
            cls = "root"
        else:
            cls = "normal"
        lines.append(f"    class {node_id} {cls}")

    # Connect root to any node that has no parent in the edge list
    orphans = sorted(rid for rid in scores if rid != root_id and rid not in child_set)
    for orphan in orphans:
        lines.append(f'    R{root_id} -->|"initial"| R{orphan}')

    # Define edges with colored links (labels already shortened in _parse_tree)
    for parent, child, _success, label in edges:
        p_id = f"R{parent}"
        c_id = f"R{child}"
        # Escape quotes for mermaid
        label = label.replace('"', "'")
        lines.append(f'    {p_id} -->|"{label}"| {c_id}')

    # Color edges: green for success, red for failure
    # Account for the orphan→root edges added above (they have no color override)
    offset = len(orphans)
    for i, (_, _, success, _) in enumerate(edges):
        color = "#00C853" if success else "#D50000"
        lines.append(f"    linkStyle {i + offset} stroke:{color},stroke-width:2px")

    return "\n".join(lines)


def _render_mermaid(mermaid_code: str, height: int = 500) -> None:
    """Render a Mermaid diagram via HTML component."""
    html = f"""
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
    </head>
    <body style="background: transparent; margin: 0; padding: 0;">
        <pre class="mermaid" style="display: flex; justify-content: center;">
{mermaid_code}
        </pre>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'neutral',
                flowchart: {{ curve: 'basis', padding: 15 }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html, height=height)


def display_mcts_tree(
    tree_data: Dict,
    experiences: Optional[Dict[int, Dict]] = None,
    dev_scores: Optional[Dict[int, float]] = None,
    score_label: str = "Train",
    baseline_score: Optional[float] = None,
) -> None:
    """Render the MCTS search tree using Mermaid.

    If dev_scores is provided, the node labels show dev scores and the best
    node is determined by dev performance.  Otherwise, train scores from the
    experience data are used.
    """
    if not tree_data:
        st.info("No MCTS experience data available.")
        return

    scores, best_round, edges = _parse_tree(tree_data)
    if not scores:
        st.info("No tree nodes to display.")
        return

    # Inject Round 0 (Eve app baseline) as root
    if baseline_score is not None:
        scores[0] = baseline_score
        if 1 in scores:
            edges.insert(0, (0, 1, scores.get(1, 0) > baseline_score, "template"))

    # Override scores with dev partition if provided
    if dev_scores:
        for rnd, s in dev_scores.items():
            scores[rnd] = s
        best_round = max(dev_scores, key=dev_scores.get) if dev_scores else best_round

    mermaid_code = _generate_mcts_mermaid(scores, best_round, edges)
    n_nodes = len(scores)
    diagram_height = max(300, n_nodes * 80 + 100)
    _render_mermaid(mermaid_code, height=diagram_height)

    # Legend
    cols = st.columns(4)
    cols[0].markdown(":green_heart: **Green edge** = Improved over parent")
    cols[1].markdown(":red_circle: **Red edge** = Regressed from parent")
    cols[2].markdown(":orange_heart: **Orange node** = Best round")
    cols[3].markdown(f"Scores from **{score_label}** partition")

    # Modification details
    if experiences:
        with st.expander("Modification Details"):
            for round_num in sorted(experiences.keys()):
                exp = experiences[round_num]
                status = "Improved" if exp.get("succeed") else "Regressed"
                st.markdown(
                    f"**Round {round_num}** "
                    f"(from Round {exp.get('father node', exp.get('father_node', '?'))}): "
                    f"{status} ({exp.get('before', 0):.1%} → {exp.get('after', 0):.1%})"
                )
                st.caption(exp.get("modification", ""))
