"""
Benchmark FEASTA/PySTA query-engine tasks on a CSV dump.

This script measures end-to-end design load, topology construction, and a
small suite of query tasks that exercise filtering, point lookup, traversal,
and critical-path extraction.
"""

import argparse
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pysta import Design


def _time_call(fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    return dt, result


def _full_name_to_node_name(full_name):
    if pd.isna(full_name):
        return None

    name = str(full_name)
    if name.startswith("top/"):
        name = name[4:]

    if "/" not in name:
        return name

    hier, pin = name.rsplit("/", 1)
    return f"{hier}:{pin}"


def _build_query_nodes(design):
    nodes = design.nodes.copy()

    if "SlackWorst_ns" in nodes.columns and nodes["SlackWorst_ns"].notna().any():
        return nodes

    pin_df = getattr(design, "_pin_properties_df", None)
    if pin_df is None or pin_df.empty or "FullName" not in pin_df.columns:
        return nodes

    merge_cols = [
        col
        for col in ["FullName", "SlackWorst_ns", "IsClock", "ClockNames", "Capacitance_pf"]
        if col in pin_df.columns
    ]
    if len(merge_cols) <= 1:
        return nodes

    timing = pin_df[merge_cols].copy()
    timing["Name"] = timing["FullName"].map(_full_name_to_node_name)
    timing = timing.drop(columns=["FullName"])
    timing = timing.dropna(subset=["Name"])
    timing = timing.drop_duplicates(subset=["Name"], keep="first")

    nodes = nodes.merge(timing, on="Name", how="left", suffixes=("", "_pinprop"))
    return nodes


def _timing_seed_df(design, nodes_df):
    if "SlackWorst_ns" in nodes_df.columns:
        seed = nodes_df[["Name", "SlackWorst_ns"]].copy()
        seed["SlackWorst_ns"] = pd.to_numeric(seed["SlackWorst_ns"], errors="coerce")
        seed = seed.dropna(subset=["Name", "SlackWorst_ns"])
        if not seed.empty:
            return seed

    pin_df = getattr(design, "_pin_properties_df", None)
    if pin_df is None or pin_df.empty or "FullName" not in pin_df.columns:
        return pd.DataFrame(columns=["Name", "SlackWorst_ns"])

    seed = pin_df[["FullName", "SlackWorst_ns"]].copy()
    seed["Name"] = seed["FullName"].map(_full_name_to_node_name)
    seed["SlackWorst_ns"] = pd.to_numeric(seed["SlackWorst_ns"], errors="coerce")
    valid_names = set(nodes_df["Name"].dropna().astype(str))
    seed = seed[seed["Name"].isin(valid_names)]
    seed = seed.dropna(subset=["Name", "SlackWorst_ns"])
    seed = seed[["Name", "SlackWorst_ns"]].drop_duplicates(subset=["Name"], keep="first")
    return seed


def _annotated_pin_df(design, nodes_df):
    pin_df = getattr(design, "_pin_properties_df", None)
    if pin_df is None or pin_df.empty or "FullName" not in pin_df.columns:
        return pd.DataFrame()

    cols = [
        col
        for col in ["FullName", "SlackWorst_ns", "IsClock", "ClockNames", "Capacitance_pf", "Direction"]
        if col in pin_df.columns
    ]
    pins = pin_df[cols].copy()
    pins["Name"] = pins["FullName"].map(_full_name_to_node_name)
    pins["SlackWorst_ns"] = pd.to_numeric(pins.get("SlackWorst_ns"), errors="coerce")
    if "Capacitance_pf" in pins.columns:
        pins["Capacitance_pf"] = pd.to_numeric(pins["Capacitance_pf"], errors="coerce")
    if "IsClock" in pins.columns:
        pins["IsClock"] = pins["IsClock"].map(
            {True: True, False: False, "true": True, "false": False, "True": True, "False": False}
        ).fillna(False)
    valid_names = set(nodes_df["Name"].dropna().astype(str))
    pins = pins[pins["Name"].isin(valid_names)]
    return pins.drop_duplicates(subset=["Name"], keep="first")


def _node_degree(design, node_name):
    node_id = design._name_to_id.get(node_name)
    if node_id is None or design.topology is None:
        return -1, -1
    fanin = len(design.topology.get_fanin(node_id))
    fanout = len(design.topology.get_fanout(node_id))
    return fanin, fanout


def _pick_worst_endpoint(design, nodes_df):
    slack_df = _timing_seed_df(design, nodes_df)
    if slack_df.empty:
        raise ValueError("No nodes with SlackWorst_ns available")

    slack_df = slack_df.sort_values("SlackWorst_ns").copy()
    for _, row in slack_df.iterrows():
        fanin, fanout = _node_degree(design, row["Name"])
        if fanin > 0:
            return row["Name"], float(row["SlackWorst_ns"])

    worst = slack_df.iloc[0]
    return worst["Name"], float(worst["SlackWorst_ns"])


def _pick_register_q(design, nodes_df):
    name_series = nodes_df["Name"].fillna("")
    q_df = nodes_df[name_series.str.endswith("/Q")].copy()
    if q_df.empty:
        q_df = nodes_df[name_series.str.endswith(":Q")].copy()
    if q_df.empty:
        out_df = nodes_df[nodes_df["Direction"] == "output"].copy()
        if out_df.empty:
            pin_df = getattr(design, "_pin_properties_df", None)
            if pin_df is not None and "FullName" in pin_df.columns:
                tmp = pin_df.copy()
                tmp["Name"] = tmp["FullName"].map(_full_name_to_node_name)
                valid_names = set(nodes_df["Name"].dropna().astype(str))
                tmp = tmp[tmp["Name"].isin(valid_names)]
                tmp = tmp[tmp["Name"].astype(str).str.endswith((":Q", "/Q"))]
                if not tmp.empty:
                    return str(tmp.iloc[0]["Name"])
            raise ValueError("Could not find a register Q pin or output pin")
        return str(out_df.iloc[0]["Name"])

    if "SlackWorst_ns" in q_df.columns:
        q_df["SlackWorst_ns"] = pd.to_numeric(q_df["SlackWorst_ns"], errors="coerce")
        q_df = q_df.sort_values("SlackWorst_ns", na_position="last")

    for _, row in q_df.iterrows():
        fanin, fanout = _node_degree(design, row["Name"])
        if fanout > 0:
            return str(row["Name"])

    return str(q_df.iloc[0]["Name"])


def _worst_violations(seed_df, top_k):
    if seed_df is None or seed_df.empty or "SlackWorst_ns" not in seed_df.columns:
        return {"violation_count": 0, "top_k": []}

    slack = pd.to_numeric(seed_df["SlackWorst_ns"], errors="coerce")
    viol = seed_df[slack < 0].copy()
    if viol.empty:
        return {
            "violation_count": 0,
            "top_k": [],
        }

    cols = [c for c in ["Name", "SlackWorst_ns", "Direction"] if c in viol.columns]
    worst = viol[cols].sort_values("SlackWorst_ns").head(top_k)
    return {
        "violation_count": int(len(viol)),
        "top_k": worst.to_dict(orient="records"),
    }


def _critical_paths(design, seed_df, top_k, max_stages=100):
    if design.topology is None or seed_df.empty:
        return []

    qe = design.pins._qe
    compact = []
    seen = set()
    for _, row in seed_df.sort_values("SlackWorst_ns").iterrows():
        endpoint = row["Name"]
        if endpoint in seen:
            continue
        node_id = design._name_to_id.get(endpoint)
        if node_id is None:
            continue
        fanin, _ = _node_degree(design, endpoint)
        if fanin <= 0:
            continue
        path_ids = qe._trace_back(node_id, max_stages)
        if len(path_ids) <= 1:
            continue
        path_names = [design._id_to_name.get(nid, f"node_{nid}") for nid in path_ids]
        compact.append(
            {
                "startpoint": path_names[0],
                "endpoint": endpoint,
                "slack_ns": round(float(row["SlackWorst_ns"]), 6),
                "stages": len(path_ids) - 1,
                "path_head": path_names[:3],
                "path_tail": path_names[-3:],
            }
        )
        seen.add(endpoint)
        if len(compact) >= top_k:
            break
    return compact


def _path_pair_from_slack(design, seed_df, max_stages=100):
    paths = _critical_paths(design, seed_df, top_k=1, max_stages=max_stages)
    if not paths:
        raise ValueError("Could not derive a startpoint-endpoint path pair")

    path = paths[0]
    return path["startpoint"], path["endpoint"], path["slack_ns"]


def _paths_between(design, startpoint, endpoint, top_k, max_stages=100):
    paths = design.pins.get_paths_between(
        startpoint,
        endpoint,
        top_k=top_k,
        max_stages=max_stages,
    )
    compact = []
    for path in paths:
        path_nodes = path["path"]
        compact.append(
            {
                "startpoint": path["startpoint"],
                "endpoint": path["endpoint"],
                "stages": path["stages"],
                "path_head": path_nodes[:3],
                "path_tail": path_nodes[-3:],
            }
        )
    return compact


def _fanout_cone(design, node_name, depth):
    fanout = design.get_fanout(node_name, depth=depth)
    cols = [c for c in ["Name", "Depth", "SlackWorst_ns", "Direction"] if c in fanout.columns]
    return {
        "center": node_name,
        "depth": depth,
        "node_count": int(len(fanout)),
        "sample": fanout[cols].head(10).to_dict(orient="records") if not fanout.empty else [],
    }


def _clock_summary(pin_df):
    if pin_df is None or pin_df.empty or "IsClock" not in pin_df.columns:
        return {
            "clock_pin_count": 0,
            "cap_sum_pf": 0.0,
            "worst_clock_slack_ns": None,
            "sample": [],
        }

    clock_mask = pin_df["IsClock"].fillna(False).astype(bool)
    clocks = pin_df[clock_mask].copy()
    if clocks.empty:
        return {
            "clock_pin_count": 0,
            "cap_sum_pf": 0.0,
            "worst_clock_slack_ns": None,
            "sample": [],
        }

    cap_sum = 0.0
    if "Capacitance_pf" in clocks.columns:
        cap_sum = float(pd.to_numeric(clocks["Capacitance_pf"], errors="coerce").fillna(0.0).sum())

    worst_slack = None
    if "SlackWorst_ns" in clocks.columns:
        slack = pd.to_numeric(clocks["SlackWorst_ns"], errors="coerce").dropna()
        if not slack.empty:
            worst_slack = float(slack.min())

    cols = [c for c in ["Name", "Capacitance_pf", "SlackWorst_ns", "ClockNames"] if c in clocks.columns]
    sample = clocks[cols].head(10).to_dict(orient="records")
    return {
        "clock_pin_count": int(len(clocks)),
        "cap_sum_pf": cap_sum,
        "worst_clock_slack_ns": worst_slack,
        "sample": sample,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PySTA query-engine tasks")
    parser.add_argument("--csv_dir", required=True, help="Directory containing FEASTA CSVs")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K size for worst violations / critical paths")
    parser.add_argument("--fanout-depth", type=int, default=5, help="Depth for register-Q fanout query")
    args = parser.parse_args()

    load_time, design = _time_call(
        lambda: Design(args.csv_dir, lazy_topology=True, verbose=False)
    )
    query_nodes = _build_query_nodes(design)
    design._nodes_df = query_nodes
    topo_time, _ = _time_call(design._ensure_topology)
    timing_seed_df = _timing_seed_df(design, query_nodes)
    annotated_pins = _annotated_pin_df(design, query_nodes)

    worst_name, worst_slack = _pick_worst_endpoint(design, query_nodes)
    q_name = _pick_register_q(design, query_nodes)
    pair_start, pair_end, pair_slack = _path_pair_from_slack(design, timing_seed_df)

    tasks = [
        (
            "worst_violations",
            f"design.filter(SlackWorst_ns__lt=0) -> sort by SlackWorst_ns -> head({args.top_k})",
            lambda: _worst_violations(annotated_pins, args.top_k),
        ),
        (
            "critical_paths",
            f"design.get_critical_paths(top_k={args.top_k})",
            lambda: _critical_paths(design, timing_seed_df, args.top_k),
        ),
        (
            "paths_between_pair",
            f"design.get_paths_between('{pair_start}', '{pair_end}', top_k={args.top_k})",
            lambda: _paths_between(design, pair_start, pair_end, args.top_k),
        ),
        (
            "fanout_register_q",
            f"design.get_fanout('{q_name}', depth={args.fanout_depth})",
            lambda: _fanout_cone(design, q_name, args.fanout_depth),
        ),
        (
            "clock_summary",
            "design.filter(IsClock=True) + aggregate clock-pin stats",
            lambda: _clock_summary(annotated_pins),
        ),
    ]

    print("=" * 72)
    print("FEASTA Query Benchmark")
    print("=" * 72)
    print(f"CSV dir                 : {args.csv_dir}")
    print(f"PySTA Load CSV          : {load_time:.2f}s")
    print(f"Topology Build          : {topo_time:.2f}s")
    print(f"Worst endpoint seed     : {worst_name} ({worst_slack:.6f} ns)")
    print(f"Path pair seed          : {pair_start} -> {pair_end} ({pair_slack:.6f} ns)")
    print(f"Register/Q seed         : {q_name}")
    print("=" * 72)

    for label, command, fn in tasks:
        dt, result = _time_call(fn)
        print(f"[{label}]")
        print(f"Command                 : {command}")
        print(f"Runtime                 : {dt:.4f}s")
        print("Result                  :")
        print(result)
        print("-" * 72)


if __name__ == "__main__":
    main()
