"""
Utilities for ECM topology simplification and vectorization.

.. currentmodule:: autoeis.ecm

.. autosummary::
   :toctree: generated/

    parse_circuit_tree
    normalize_tree
    simplify_ecm
    compare_ecm_topology
    deduplicate_ecms
    ecm_to_vector
    vector_to_ecm
    parse_eis_value_tree
    eis_to_vector
    ecm_to_eis_string

"""

from __future__ import annotations

import json
import math
import re
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COMPONENT_RE = re.compile(r"[RCLPW]\d+", re.IGNORECASE)
VALUE_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

ECM_START_CODES = {"series": 0, "parallel": 1}
ECM_LEAF_CODES = {"R": 2, "C": 3, "L": 4, "P": 5, "W": 6}
ECM_END_CODES = {"series": -2, "parallel": -3}

ECM_CODE_TO_NODE = {
    **{v: k for k, v in ECM_START_CODES.items()},
    **{v: k for k, v in ECM_LEAF_CODES.items()},
}
ECM_END_CODE_TO_NODE = {v: k for k, v in ECM_END_CODES.items()}

EIS_START_CODES = {"series": 0, "parallel": 1}
EIS_VALUE_CODE = 2
EIS_END_CODES = {"series": -2, "parallel": -3}

CircuitTree = str | tuple[str, list["CircuitTree"]]
ValueTree = float | tuple[str, list["ValueTree"]]

__all__ = [
    "CircuitTree",
    "ValueTree",
    "split_top_level",
    "parse_circuit_tree",
    "tree_to_circuit",
    "normalize_tree",
    "simplify_tree",
    "simplify_ecm",
    "compare_ecm_topology",
    "deduplicate_ecms",
    "topology_to_weighted_vector",
    "cosine_similarity",
    "structure_similarity",
    "analyze_circuit_structure",
    "compare_circuit_structures",
    "ecm_to_vector",
    "vector_to_ecm",
    "parse_eis_value_tree",
    "eis_to_vector",
    "extract_eis_values",
    "replace_eis_values",
    "simplify_ecm_eis_pair",
    "ecm_to_eis_string",
    "geometric_mean_frequency",
    "compress_parameter_values",
    "order_parameters_by_topology",
    "select_best_circuit_row",
    "extract_vectorizable_params",
    "add_ecm_vectors",
    "ECMVectorTracker",
    "EISVectorTracker",
]


def split_top_level(text: str, sep: str) -> list[str]:
    """Split a circuit string on a separator only at bracket depth zero."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced bracket in {text!r}.")

        if char == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)

    if depth != 0:
        raise ValueError(f"Unbalanced bracket in {text!r}.")
    parts.append("".join(buf))
    return parts


def _component_token(token: str, preserve_labels: bool) -> str:
    token = token.strip()
    match = COMPONENT_RE.fullmatch(token)
    if match:
        value = match.group(0)
        return value.upper() if preserve_labels else value[0].upper()

    type_match = re.fullmatch(r"[RCLPW]", token, flags=re.IGNORECASE)
    if type_match:
        return token.upper()

    raise ValueError(f"Invalid ECM component token: {token!r}.")


def parse_circuit_tree(circuit: str, *, preserve_labels: bool = True) -> CircuitTree:
    """Parse an AutoEIS circuit string into a nested series/parallel tree."""
    text = circuit.strip().replace(" ", "")
    if not text:
        raise ValueError("Circuit string is empty.")

    series_parts = split_top_level(text, "-")
    if len(series_parts) > 1:
        return (
            "series",
            [
                parse_circuit_tree(part, preserve_labels=preserve_labels)
                for part in series_parts
            ],
        )

    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1]
        parallel_parts = split_top_level(inner, ",")
        if len(parallel_parts) < 2:
            raise ValueError(f"Parallel group must contain at least two children: {text!r}.")
        return (
            "parallel",
            [
                parse_circuit_tree(part, preserve_labels=preserve_labels)
                for part in parallel_parts
            ],
        )

    return _component_token(text, preserve_labels)


def _component_type(component: str) -> str:
    return component[0].upper()


def _tree_sort_key(node: CircuitTree | ValueTree) -> str:
    return tree_to_circuit(node) if isinstance(node, tuple) else str(node)


def normalize_tree(
    tree: CircuitTree,
    *,
    sort: bool = True,
    flatten: bool = True,
) -> CircuitTree:
    """Normalize a circuit tree by flattening equal groups and optionally sorting."""
    if isinstance(tree, str):
        return tree

    label, children = tree
    normalized_children = [
        normalize_tree(child, sort=sort, flatten=flatten) for child in children
    ]

    flat: list[CircuitTree] = []
    for child in normalized_children:
        if flatten and isinstance(child, tuple) and child[0] == label:
            flat.extend(child[1])
        else:
            flat.append(child)

    if len(flat) == 1:
        return flat[0]
    if sort:
        flat = sorted(flat, key=_tree_sort_key)
    return (label, flat)


def simplify_tree(tree: CircuitTree, *, preserve_order: bool = True) -> CircuitTree | None:
    """Simplify an ECM tree by merging same-type R/C/L leaves within each group."""
    if isinstance(tree, str):
        return tree

    label, children = tree
    simplified = [simplify_tree(child, preserve_order=preserve_order) for child in children]
    simplified = [child for child in simplified if child is not None]
    if not simplified:
        return None
    if len(simplified) == 1:
        return simplified[0]

    flattened: list[CircuitTree] = []
    for child in simplified:
        if isinstance(child, tuple) and child[0] == label:
            flattened.extend(child[1])
        else:
            flattened.append(child)

    representatives: dict[str, str] = {}
    output: list[CircuitTree] = []
    for child in flattened:
        if isinstance(child, str) and _component_type(child) in {"R", "C", "L"}:
            ctype = _component_type(child)
            if ctype not in representatives:
                representatives[ctype] = child
                output.append(child)
        else:
            output.append(child)

    if not preserve_order:
        output = sorted(output, key=_tree_sort_key)
    if len(output) == 1:
        return output[0]
    return (label, output)


def tree_to_circuit(tree: CircuitTree | ValueTree) -> str:
    """Convert a circuit or numeric value tree back to a string."""
    if isinstance(tree, tuple):
        label, children = tree
        sep = "-" if label == "series" else ","
        body = sep.join(tree_to_circuit(child) for child in children)
        return body if label == "series" else f"[{body}]"
    if isinstance(tree, float):
        return f"{tree:.12g}"
    return str(tree)


def simplify_ecm(circuit: str, *, preserve_labels: bool = True) -> str:
    """Simplify an AutoEIS ECM string while preserving the first label per merged type."""
    tree = parse_circuit_tree(circuit, preserve_labels=preserve_labels)
    simplified = simplify_tree(tree)
    if simplified is None:
        return ""
    return tree_to_circuit(simplified)


def _topology_tree(circuit: str) -> CircuitTree:
    return normalize_tree(parse_circuit_tree(circuit, preserve_labels=False))


def compare_ecm_topology(circuit1: str, circuit2: str, *, simplify: bool = True) -> bool:
    """Return True when two ECM strings have equivalent normalized topology."""
    tree1 = parse_circuit_tree(circuit1, preserve_labels=False)
    tree2 = parse_circuit_tree(circuit2, preserve_labels=False)
    if simplify:
        tree1 = simplify_tree(tree1) or tree1
        tree2 = simplify_tree(tree2) or tree2
    return normalize_tree(tree1) == normalize_tree(tree2)


def deduplicate_ecms(circuits: Iterable[str], *, simplify: bool = True) -> list[str]:
    """Remove duplicate ECM topologies while preserving first occurrences."""
    unique: list[str] = []
    for circuit in circuits:
        is_duplicate = any(
            compare_ecm_topology(circuit, other, simplify=simplify) for other in unique
        )
        if not is_duplicate:
            unique.append(circuit)
    return unique


def topology_to_weighted_vector(tree: CircuitTree | str) -> dict[str, float]:
    """Convert an ECM topology tree to a sparse weighted path vector."""
    if isinstance(tree, str):
        tree = parse_circuit_tree(tree, preserve_labels=False)

    weights: dict[str, float] = defaultdict(float)

    def visit(node: CircuitTree, path: str = "root", depth: int = 0) -> None:
        if isinstance(node, str):
            weights[f"{path}:{_component_type(node)}"] += 3 / (1 + depth)
            return

        label, children = node
        weights[f"{path}:{label}"] += 6 / (1 + depth)
        for idx, child in enumerate(children):
            visit(child, f"{path}-{label}[{idx}]", depth + 1)

    visit(tree)
    return dict(weights)


def cosine_similarity(vector1: Mapping[str, float], vector2: Mapping[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    keys = set(vector1) | set(vector2)
    dot = sum(vector1.get(key, 0.0) * vector2.get(key, 0.0) for key in keys)
    norm1 = math.sqrt(sum(value * value for value in vector1.values()))
    norm2 = math.sqrt(sum(value * value for value in vector2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def structure_similarity(circuit1: str | CircuitTree, circuit2: str | CircuitTree) -> float:
    """Compute weighted topology similarity between two ECMs."""
    return cosine_similarity(
        topology_to_weighted_vector(circuit1),
        topology_to_weighted_vector(circuit2),
    )


def analyze_circuit_structure(
    tree: CircuitTree | str,
    circuit_name: str = "Circuit",
) -> dict[str, Any]:
    """Return depth, element, and connection summaries for an ECM topology."""
    if isinstance(tree, str):
        tree = parse_circuit_tree(tree, preserve_labels=True)

    info: dict[str, Any] = {
        "circuit_name": circuit_name,
        "depth_info": {},
        "max_depth": 0,
        "total_elements": 0,
        "element_types": {},
        "connection_patterns": [],
    }

    def visit(node: CircuitTree, depth: int = 0) -> None:
        info["max_depth"] = max(info["max_depth"], depth)
        info["depth_info"].setdefault(
            depth,
            {"connections": [], "elements": [], "element_count": 0},
        )

        if isinstance(node, str):
            info["depth_info"][depth]["elements"].append(node)
            info["depth_info"][depth]["element_count"] += 1
            info["total_elements"] += 1
            ctype = _component_type(node)
            info["element_types"][ctype] = info["element_types"].get(ctype, 0) + 1
            return

        label, children = node
        info["depth_info"][depth]["connections"].append(label)
        info["connection_patterns"].append(f"Depth{depth}:{label}({len(children)})")
        for child in children:
            visit(child, depth + 1)

    visit(tree)
    return info


def compare_circuit_structures(
    structure1: Mapping[str, Any],
    structure2: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two structure summaries returned by :func:`analyze_circuit_structure`."""
    comparison: dict[str, Any] = {
        "depth_comparison": {},
        "element_comparison": {},
        "summary": {
            "max_depth_diff": structure1["max_depth"] - structure2["max_depth"],
            "total_elements_diff": structure1["total_elements"] - structure2["total_elements"],
        },
    }

    element_types = set(structure1["element_types"]) | set(structure2["element_types"])
    for ctype in element_types:
        count1 = structure1["element_types"].get(ctype, 0)
        count2 = structure2["element_types"].get(ctype, 0)
        comparison["element_comparison"][ctype] = {
            "structure1": count1,
            "structure2": count2,
            "diff": count1 - count2,
        }

    max_depth = max(structure1["max_depth"], structure2["max_depth"])
    empty = {"connections": [], "elements": [], "element_count": 0}
    for depth in range(max_depth + 1):
        depth1 = structure1["depth_info"].get(depth, empty)
        depth2 = structure2["depth_info"].get(depth, empty)
        comparison["depth_comparison"][depth] = {
            "connections_1": depth1["connections"],
            "connections_2": depth2["connections"],
            "elements_1": depth1["elements"],
            "elements_2": depth2["elements"],
            "element_count_diff": depth1["element_count"] - depth2["element_count"],
            "connection_match": set(depth1["connections"]) == set(depth2["connections"]),
            "element_match": set(depth1["elements"]) == set(depth2["elements"]),
        }
    return comparison


def ecm_to_vector(circuit: str | CircuitTree, *, simplify: bool = True) -> list[int]:
    """Convert an ECM topology to a bracketed integer vector."""
    tree = (
        parse_circuit_tree(circuit, preserve_labels=False)
        if isinstance(circuit, str)
        else circuit
    )
    if simplify:
        tree = simplify_tree(tree, preserve_order=True) or tree

    vector: list[int] = []

    def visit(node: CircuitTree) -> None:
        if isinstance(node, tuple):
            label, children = node
            vector.append(ECM_START_CODES[label])
            for child in children:
                visit(child)
            vector.append(ECM_END_CODES[label])
        else:
            vector.append(ECM_LEAF_CODES[_component_type(node)])

    visit(tree)
    return vector


def vector_to_ecm(vector: Sequence[int], *, start_index: int = 1) -> str:
    """Reconstruct a labeled ECM string from an integer topology vector."""
    stack: list[tuple[str, list[CircuitTree]]] = []
    roots: list[CircuitTree] = []

    for code in vector:
        code = int(code)
        if code in ECM_START_CODES.values():
            stack.append((ECM_CODE_TO_NODE[code], []))
        elif code in ECM_END_CODE_TO_NODE:
            expected = ECM_END_CODE_TO_NODE[code]
            if not stack or stack[-1][0] != expected:
                raise ValueError(f"Unbalanced ECM vector near end code {code}.")
            label, children = stack.pop()
            node: CircuitTree = (label, children)
            if stack:
                stack[-1][1].append(node)
            else:
                roots.append(node)
        elif code in ECM_LEAF_CODES.values():
            leaf = ECM_CODE_TO_NODE[code]
            if stack:
                stack[-1][1].append(leaf)
            else:
                roots.append(leaf)
        else:
            raise ValueError(f"Unknown ECM vector code: {code}.")

    if stack:
        raise ValueError("Unbalanced ECM vector: some groups were not closed.")
    if len(roots) != 1:
        raise ValueError("ECM vector must decode to exactly one root node.")

    counters = {key: start_index for key in ECM_LEAF_CODES}

    def label_tree(node: CircuitTree) -> CircuitTree:
        if isinstance(node, str):
            ctype = _component_type(node)
            label = f"{ctype}{counters[ctype]}"
            counters[ctype] += 1
            return label
        label, children = node
        return (label, [label_tree(child) for child in children])

    return tree_to_circuit(label_tree(roots[0]))


def parse_eis_value_tree(expr: str) -> ValueTree:
    """Parse a numeric EIS expression such as ``10-[20,30]`` into a value tree."""
    text = expr.replace(" ", "")
    idx = 0

    def parse_number() -> float:
        nonlocal idx
        match = VALUE_RE.match(text[idx:])
        if not match:
            raise ValueError(f"Expected numeric value at position {idx} in {text!r}.")
        idx += len(match.group(0))
        return float(match.group(0))

    def parse_node() -> ValueTree:
        nonlocal idx
        if idx < len(text) and text[idx] == "[":
            return parse_parallel()
        return parse_number()

    def parse_series() -> ValueTree:
        nonlocal idx
        children = [parse_node()]
        while idx < len(text) and text[idx] == "-":
            idx += 1
            children.append(parse_node())
        return ("series", children) if len(children) > 1 else children[0]

    def parse_parallel() -> ValueTree:
        nonlocal idx
        idx += 1
        children = [parse_series()]
        while idx < len(text) and text[idx] == ",":
            idx += 1
            children.append(parse_series())
        if idx >= len(text) or text[idx] != "]":
            raise ValueError(f"Unclosed parallel group in {text!r}.")
        idx += 1
        return ("parallel", children)

    tree = parse_series()
    if idx != len(text):
        raise ValueError(f"Unexpected trailing text in EIS value expression: {text[idx:]!r}.")
    return tree


def eis_to_vector(eis: str | ValueTree) -> list[int | float]:
    """Convert a numeric EIS value tree to a vector with value markers."""
    tree = parse_eis_value_tree(eis) if isinstance(eis, str) else eis
    vector: list[int | float] = []

    def visit(node: ValueTree) -> None:
        if isinstance(node, tuple):
            label, children = node
            vector.append(EIS_START_CODES[label])
            for child in children:
                visit(child)
            vector.append(EIS_END_CODES[label])
        else:
            vector.extend([EIS_VALUE_CODE, float(node)])

    visit(tree)
    return vector


def extract_eis_values(eis_vector: Sequence[int | float]) -> list[float]:
    """Extract numeric leaves from an EIS vector."""
    values: list[float] = []
    idx = 0
    while idx < len(eis_vector):
        if int(eis_vector[idx]) == EIS_VALUE_CODE and idx + 1 < len(eis_vector):
            values.append(float(eis_vector[idx + 1]))
            idx += 2
        else:
            idx += 1
    return values


def replace_eis_values(
    eis_vector: Sequence[int | float],
    values: Sequence[float],
) -> list[int | float]:
    """Replace numeric leaves in an EIS vector, preserving its topology codes."""
    output = list(eis_vector)
    value_idx = 0
    idx = 0
    while idx < len(output) - 1:
        if int(output[idx]) == EIS_VALUE_CODE:
            if value_idx < len(values):
                output[idx + 1] = float(values[value_idx])
                value_idx += 1
            idx += 2
        else:
            idx += 1
    return output


def _is_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def simplify_ecm_eis_pair(
    ecm: str | CircuitTree,
    eis: str | ValueTree,
) -> tuple[CircuitTree, ValueTree]:
    """Simplify matching ECM and numeric EIS trees together for R/C/L runs."""
    ecm_tree = parse_circuit_tree(ecm, preserve_labels=True) if isinstance(ecm, str) else ecm
    eis_tree = parse_eis_value_tree(eis) if isinstance(eis, str) else eis

    def combine(label: str, ctype: str, values: list[float]) -> float | None:
        if label == "series":
            if ctype in {"R", "L"}:
                return float(sum(values))
            if ctype == "C":
                denom = sum(1 / val for val in values if val != 0)
                return float("inf") if denom == 0 else float(1 / denom)
        if label == "parallel":
            if ctype in {"R", "L"}:
                denom = sum(1 / val for val in values if val != 0)
                return float("inf") if denom == 0 else float(1 / denom)
            if ctype == "C":
                return float(sum(values))
        return None

    def simplify_pair(
        ecm_node: CircuitTree,
        eis_node: ValueTree,
    ) -> tuple[CircuitTree, ValueTree, str | None]:
        if isinstance(ecm_node, str):
            simple_eis = float(eis_node) if _is_number(eis_node) else eis_node
            return ecm_node, simple_eis, _component_type(ecm_node)

        label, ecm_children = ecm_node
        if isinstance(eis_node, tuple) and eis_node[0] == label:
            eis_children = list(eis_node[1])
        else:
            eis_children = [eis_node] * len(ecm_children)

        items = []
        for child_ecm, child_eis in zip(ecm_children, eis_children):
            simple_ecm, simple_eis, ctype = simplify_pair(child_ecm, child_eis)
            items.append(
                {
                    "ecm": simple_ecm,
                    "eis": simple_eis,
                    "type": ctype,
                    "number": float(simple_eis)
                    if ctype in {"R", "C", "L"} and _is_number(simple_eis)
                    else None,
                }
            )

        merged = []
        idx = 0
        while idx < len(items):
            item = items[idx]
            ctype = item["type"]
            if ctype in {"R", "C", "L"} and item["number"] is not None:
                run = [item]
                idx += 1
                while (
                    idx < len(items)
                    and items[idx]["type"] == ctype
                    and items[idx]["number"] is not None
                ):
                    run.append(items[idx])
                    idx += 1
                if len(run) > 1:
                    value = combine(label, ctype, [entry["number"] for entry in run])
                    if value is not None and np.isfinite(value):
                        merged.append({"ecm": run[0]["ecm"], "eis": value, "type": ctype})
                        continue
                merged.extend(run)
            else:
                merged.append(item)
                idx += 1

        ecm_out = [entry["ecm"] for entry in merged]
        eis_out = [entry["eis"] for entry in merged]
        if len(ecm_out) == 1:
            ctype = _component_type(ecm_out[0]) if isinstance(ecm_out[0], str) else None
            return ecm_out[0], eis_out[0], ctype
        return (label, ecm_out), (label, eis_out), None

    simple_ecm, simple_eis, _ = simplify_pair(ecm_tree, eis_tree)
    return simple_ecm, simple_eis


def ecm_to_eis_string(
    ecm: str,
    parameter_values: Mapping[str, float],
    *,
    value_fmt: str = ".6g",
) -> str:
    """Replace ECM component labels with numeric values from a parameter mapping."""

    def replace(match: re.Match[str]) -> str:
        component = match.group(0)
        if component not in parameter_values:
            return component
        return format(float(parameter_values[component]), value_fmt)

    return COMPONENT_RE.sub(replace, ecm)


def geometric_mean_frequency(freq: Sequence[float]) -> float:
    """Return the geometric mean of positive finite frequencies."""
    values = np.asarray(freq, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("No positive finite frequencies provided.")
    return float(np.exp(np.mean(np.log(values))))


def compress_parameter_values(
    parameter_values: Mapping[str, float],
    f_ref_hz: float,
) -> dict[str, float]:
    """Compress parameter labels to one scalar per component for vector workflows."""
    output: dict[str, float] = {}
    omega = 2 * math.pi * float(f_ref_hz)

    grouped: dict[str, dict[str, float]] = {}
    for name, value in parameter_values.items():
        if re.fullmatch(r"[PW]\d+[wn]", name, flags=re.IGNORECASE):
            grouped.setdefault(name[:-1], {})[name[-1].lower()] = float(value)
        elif re.fullmatch(r"[RCL]\d+", name, flags=re.IGNORECASE):
            output[name.upper()] = float(value)

    for base, values in grouped.items():
        coeff = values.get("w", np.nan)
        exponent = values.get("n", np.nan)
        base = base.upper()
        if not np.isfinite(coeff) or not np.isfinite(exponent):
            output[base] = np.nan
        elif base.startswith("P"):
            output[base] = float(coeff * omega ** (exponent - 1))
        elif base.startswith("W"):
            output[base] = float(coeff / omega**exponent)

    return output


def order_parameters_by_topology(
    topology: str,
    scalar_map: Mapping[str, float],
) -> tuple[OrderedDict[str, float], list[float]]:
    """Order scalar component values according to left-to-right ECM topology."""
    seen = set()
    ordered_items = []
    for component in COMPONENT_RE.findall(topology):
        component = component.upper()
        if component in scalar_map and component not in seen:
            ordered_items.append((component, scalar_map[component]))
            seen.add(component)

    for key, value in scalar_map.items():
        key_upper = key.upper()
        if key_upper not in seen:
            ordered_items.append((key, value))

    ordered = OrderedDict(ordered_items)
    return ordered, [value for _, value in ordered_items]


def _bad_parameters_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    text = str(value).strip()
    if text == "" or text.lower() == "none" or text == "[]":
        return 0
    if "," in text:
        return len([part for part in text.split(",") if part.strip()])
    return 1


def _is_finite_scalar(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def select_best_circuit_row(circuits: pd.DataFrame) -> pd.Series:
    """Select the best ranked circuit row using posterior quality and fit metrics."""
    df = circuits.copy()
    if "WAIC (real)" in df:
        df["WAIC (sum)"] = df["WAIC (real)"] + df["WAIC (imag)"]
    elif "WAIC (mag)" in df:
        df["WAIC (sum)"] = df["WAIC (mag)"] + df["WAIC (phase)"]
    else:
        df["WAIC (sum)"] = 0.0

    bad_parameters = (
        df["Bad Parameters"]
        if "Bad Parameters" in df
        else pd.Series(["None"] * len(df), index=df.index)
    )
    df["_bad_params_count"] = bad_parameters.apply(_bad_parameters_count)
    if (df["_bad_params_count"] == 0).any():
        df = df[df["_bad_params_count"] == 0].copy()

    converged = (
        df["converged"] if "converged" in df else pd.Series([True] * len(df), index=df.index)
    )
    divergences = (
        df["divergences"] if "divergences" in df else pd.Series([0] * len(df), index=df.index)
    )
    r2_real = (
        df["R^2 (ravg)"] if "R^2 (ravg)" in df else pd.Series([0] * len(df), index=df.index)
    )
    r2_imag = (
        df["R^2 (iavg)"] if "R^2 (iavg)" in df else pd.Series([0] * len(df), index=df.index)
    )
    mape_real = (
        df["MAPE (ravg)"] if "MAPE (ravg)" in df else pd.Series([0] * len(df), index=df.index)
    )
    mape_imag = (
        df["MAPE (iavg)"] if "MAPE (iavg)" in df else pd.Series([0] * len(df), index=df.index)
    )
    n_params = (
        df["n_params"] if "n_params" in df else pd.Series([0] * len(df), index=df.index)
    )

    df["_not_converged"] = ~converged.astype(bool)
    df["_divergences"] = divergences.fillna(0).astype(float)
    df["_r2_neg"] = -(r2_real.fillna(0) + r2_imag.fillna(0))
    df["_mape"] = mape_real.fillna(0) + mape_imag.fillna(0)
    df["_n_params"] = n_params.fillna(0).astype(float)

    sort_keys = [
        "Posterior Score Rank" if "Posterior Score Rank" in df else "WAIC (sum)",
        "WAIC (sum)",
        "_not_converged",
        "_divergences",
        "_bad_params_count",
        "_r2_neg",
        "_mape",
        "_n_params",
    ]
    df = df.sort_values(sort_keys, ascending=[True] * len(sort_keys), ignore_index=True)
    return df.iloc[0]


def extract_vectorizable_params(
    row: pd.Series,
    *,
    use_sanitized_for_gaussian: bool = False,
    raw_fallback: bool = True,
) -> dict[str, float]:
    """Extract one scalar value per posterior parameter from an annotated row."""
    details = row.get("Posterior Details", {}) or {}
    means = row.get("Posterior Means", {}) or {}
    sanitized = row.get("Sanitized Params", {}) or {}

    output: dict[str, float] = {}
    for parameter in sorted(set(details) | set(means) | set(sanitized)):
        info = details.get(parameter, {})
        use_sanitized = use_sanitized_for_gaussian or info.get("is_gaussian") is False
        value = (
            sanitized.get(parameter, np.nan)
            if use_sanitized
            else means.get(parameter, np.nan)
        )
        if use_sanitized and raw_fallback and not _is_finite_scalar(value):
            value = means.get(parameter, np.nan)
        output[parameter] = float(value) if value is not None else np.nan
    return output


def add_ecm_vectors(
    circuits: pd.DataFrame,
    *,
    circuit_column: str = "circuitstring",
    vector_column: str = "ECM Vector",
    simplify: bool = True,
) -> pd.DataFrame:
    """Return a copy of a circuits dataframe with an ECM vector column added."""
    df = circuits.copy()
    df[vector_column] = df[circuit_column].apply(
        lambda circuit: ecm_to_vector(circuit, simplify=simplify)
    )
    return df


class ECMVectorTracker:
    """Small JSON-backed tracker for ECM topology vectors."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.vectors: list[list[int]] = self._load()

    def _load(self) -> list[list[int]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text())

    def save(self) -> None:
        self.path.write_text(json.dumps(self.vectors, indent=2))

    def add(self, circuit: str, *, simplify: bool = True) -> list[int]:
        vector = ecm_to_vector(circuit, simplify=simplify)
        self.vectors.append(vector)
        self.save()
        return vector

    def latest_similarity(self) -> float | None:
        if len(self.vectors) < 2:
            return None
        previous = vector_to_ecm(self.vectors[-2])
        current = vector_to_ecm(self.vectors[-1])
        return structure_similarity(previous, current)


class EISVectorTracker:
    """Small JSON-backed tracker for numeric EIS vectors."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.vectors: list[list[int | float]] = self._load()

    def _load(self) -> list[list[int | float]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text())

    def save(self) -> None:
        self.path.write_text(json.dumps(self.vectors, indent=2))

    def add(self, eis: str | ValueTree) -> list[int | float]:
        vector = eis_to_vector(eis)
        self.vectors.append(vector)
        self.save()
        return vector
