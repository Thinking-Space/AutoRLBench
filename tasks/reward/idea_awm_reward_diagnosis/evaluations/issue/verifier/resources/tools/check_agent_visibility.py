"""Check visible inputs for reviewer annotations, without changing raw evidence.

This helper stays in the hidden verifier resources. Historical trajectory IDs
are legitimate inside raw evidence; labelled indexes and reviewer conclusions
are not. This is a targeted regression guard, not a general semantic classifier.
"""

from pathlib import Path
import re


REVIEWER_MARKERS = re.compile(
    r"curated_reward_examples|reward_label_examples|selected_trajectories|"
    r"packaging_audit\.json|"
    r"\b(?:counterexample|reward_error)[_-](?:ids?|list|count|index)\b|"
    r"\b(?:audited|checked|confirmed)\s+(?:\d+\s+)?"
    r"(?:full[- ]reward\s+)?(?:counterexamples?|reward[- ]errors?)\b|"
    r"\b\d+\s+(?:(?:audited|checked|confirmed)\s+)?counterexamples?\b|"
    r"Inaccurate rule-based task rewards|Rule-based reward judgments are inaccurate",
    re.IGNORECASE,
)


def is_raw_evidence(relative):
    """Allow identifiers in original logs, not in arbitrary added index files."""
    path = relative.as_posix()
    return bool(
        re.fullmatch(r"logs/rollouts/\d+\.jsonl", path)
        or re.fullmatch(r"logs/idea_analysis/step_\d+\.jsonl", path)
        or re.fullmatch(r"logs/awm_runtime/bridge_[\d_]+\.log", path)
        or path.startswith("logs/wandb/")
    )


def scan_agent_visible(workspace, environment, instruction, *, case_ids=(), answer_fragments=()):
    """Return file-scoped findings for the actual workspace/environment mounts.

    Only hidden callers supply audited IDs and oracle excerpts. Never put those
    values, or this helper, into the visible manifests or workspace.
    """
    findings = []
    case_ids = tuple(case_ids)
    fragments = tuple(" ".join(text.casefold().split()) for text in answer_fragments if text.strip())
    roots = (("workspace", Path(workspace)), ("environment", Path(environment)))
    files = []
    for label, root in roots:
        if root.is_symlink():
            findings.append(f"{label}: symlink mount root")
            continue
        if not root.is_dir():
            findings.append(f"{label}: missing mount root")
            continue
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            name = f"{label}/{relative.as_posix()}"
            if path.is_symlink():
                findings.append(f"{name}: symlink in visible inputs")
            elif path.is_file():
                files.append((name, path, label == "workspace" and is_raw_evidence(relative)))
    instruction = Path(instruction)
    if instruction.is_symlink() or not instruction.is_file():
        findings.append("instruction: missing file or symlink")
    else:
        files.append(("instruction", instruction, False))

    for name, path, raw_evidence in files:
        if REVIEWER_MARKERS.search(name) or any(uid in name for uid in case_ids):
            findings.append(f"{name}: reviewer-labelled filename")
        if path.suffix in {'.xz', '.gz', '.bz2', '.zip', '.tar', '.tgz', '.zst'}:
            findings.append(f"{name}: unexpected archive in directly readable inputs")
            continue
        with path.open('rb') as stream:
            # Overlap also catches a marker or oracle excerpt split at a chunk boundary.
            tail = ""
            overlap = max((len(value) for value in (*case_ids, *fragments)), default=512) + 512
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                text = tail + chunk.decode("utf-8", errors="ignore")
                if REVIEWER_MARKERS.search(text):
                    findings.append(f"{name}: reviewer annotation")
                    break
                if not raw_evidence and any(uid in text for uid in case_ids):
                    findings.append(f"{name}: audited trajectory identifier outside raw evidence")
                    break
                normalized = " ".join(text.casefold().split())
                if any(fragment in normalized for fragment in fragments):
                    findings.append(f"{name}: canonical answer excerpt")
                    break
                tail = text[-overlap:]
    return findings
