# app.py — REDCap RDS Tree Automata (English UI)
import io
import os
import tempfile
import subprocess

from collections import defaultdict, deque
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
import requests
import streamlit as st

st.set_page_config(page_title="REDCap RDS Tree Automata", layout="wide")


def get_rscript_path():
    """
    Cross-platform Rscript path:
    - Windows: use installed Rscript.exe
    - macOS/Linux: use Rscript from PATH
    """
    if os.name == "nt":
        return r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe"
    return "Rscript"


#  Explanations for cleaning and estimation options
# ==============================
st.markdown("## Explanation of Cleaning and Estimation Options")

with st.expander("Cleaning options (network size adjustments)"):
    st.markdown("""
- **Fix underreported network size**  
  Occasionally participants report a personal network size (degree) smaller than the number of recruits they actually brought into the study.  
  For example, if someone recruited 3 peers but reported a network size of 2, this is underreported
  *Fix underreport* replaces such underreported cases with the observed value from the recruitment tree (out-degree + 1 for non-seeds; out-degree for seeds).

- **Impute median for NA and 0**  
  If a participant reports their network size as `NA` (missing) or `0`, the value is invalid for weighting because RDS estimators depend on network size.  
  This option replaces `NA` or `0` with a user-specified value, **median of the current network size distribution** (recommended).  


- **Set cap**  
  Sometimes individuals report extremely large network sizes (e.g., 500), which can disproportionately affect estimates.  
  *Set cap* limits reported values at a user-specified maximum (often the **75th percentile** of the distribution is recommended).  
  This reduces the influence of outliers while preserving most of the data.
""")

with st.expander("Estimation methods (weights and hidden size)"):
    st.markdown("""
- **Gile’s Successive Sampling (SS) Weights**  
  In Respondent-Driven Sampling (RDS), participants with larger personal networks are more likely to be sampled.  
  Gile’s SS estimator corrects for this by modeling recruitment as successive draws *without replacement* from a finite population.  
  The inclusion probability for each participant depends on:  
  - Their reported personal network size  
  - The total sample size  
  - The assumed prior population size in that region `N` (user-specified)  
  Weights are the inverse of these inclusion probabilities, allowing less biased population estimates.

- **SSPSE (Successive Sampling Population Size Estimation)**  
  SSPSE builds on the same successive sampling model but aims to estimate the *total hidden population size*.  
  It compares the observed distribution of reported network sizes with expected distributions under different candidate population sizes.  
  Using Bayesian inference, SSPSE generates a **posterior distribution** for the population size, summarized by mean, median, mode, and credible intervals.  
  The output includes both a summary table (prior vs posterior) and a posterior density plot.  
  RDS adjustment and SSPSE could only be applied at site-level, ideally with sample sizes >200 for stable estimation (<100 not recommended).
""")


# ==============================
# Helpers
# ==============================
def fmt_id(n) -> str:
    """Format node ID as an integer-like string (strip trailing .0 etc.)."""
    s = str(n).strip()
    if s.endswith(".0"):
        p = s[:-2]
        if p.replace("-", "").isdigit():
            return p
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def to_downloadable_html(fig, filename_label: str):
    """Render a Plotly figure as downloadable HTML."""
    html = to_html(fig, include_plotlyjs="cdn", full_html=True)
    st.download_button(
        f"Download {filename_label} (HTML)",
        data=html.encode("utf-8"),
        file_name=f"{filename_label.replace(' ', '_').lower()}.html",
        mime="text/html",
    )


# ==============================
# A) Upload + schema inference
# ==============================
def infer_schema(df: pd.DataFrame):
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    def pick_one(cands):
        for pat in cands:
            for lc, orig in lower.items():
                if pat in lc:
                    return orig
        return None

    in_field = pick_one(
        [
            "inpon_number",
            "inpon",
            "inc",
            "incoming",
            "coupon_in",
            "in_coupon",
            "recruit_id",
            "participant_id",
        ]
    )
    seed_field = pick_one(["seed_id", "seed"])
    networksize_field = pick_one(
        ["networksize", "network_size", "degree", "net_size", "personal_network"]
    )

    out_fields = []
    for c in cols:
        lc = c.lower()
        if any(
            lc.startswith(p)
            for p in ["outpons_", "outpon", "out_coupon", "coupon_out", "recruit", "out_"]
        ):
            out_fields.append(c)
    if not out_fields:
        out_fields = [
            c
            for c in cols
            if any(x in c.lower() for x in ["out1", "out2", "out3", "out4", "out5", "out6"])
        ]
    return {
        "in_field": in_field,
        "seed_field": seed_field,
        "out_fields": out_fields[:10],
        "networksize_field": networksize_field,
    }


def read_uploaded_file(uploaded_file, sep=None, sheet=None):
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        xls = pd.ExcelFile(uploaded_file)
        use_sheet = xls.sheet_names[0] if sheet is None else sheet
        return pd.read_excel(xls, use_sheet)
    content = uploaded_file.read()
    sample = content[:10000].decode("utf-8", errors="ignore")
    sep = "\t" if "\t" in sample and sample.count("\t") > sample.count(",") else ","
    return pd.read_csv(io.BytesIO(content), sep=sep)


# ==============================
# B) Build graph / attributes
# ==============================
def fetch_redcap(api_url: str, token: str) -> pd.DataFrame:
    payload = {"token": token, "content": "record", "format": "json", "type": "flat"}
    r = requests.post(api_url, data=payload)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def build_graph(df: pd.DataFrame, in_field: str, seed_field: str, out_fields: list):
    """
    Build a directed recruitment graph.

    CRITICAL RULE:
    - A "person/node" must appear in incoming coupon (incoupon) OR as a seed.
    - Coupons that appear only in outpon* but never in incoupon/seed are NOT persons.
    """
    df = df.copy()
    df["inc_code"] = df.apply(
        lambda row: row[in_field]
        if pd.notna(row[in_field]) and str(row[in_field]).strip() != ""
        else row[seed_field],
        axis=1,
    ).astype(str)

    edges = []
    for out in out_fields:
        if out not in df.columns:
            continue
        valid = df[df[out].notna() & (df[out].astype(str).str.strip() != "")]
        edges += [(row["inc_code"], str(row[out])) for _, row in valid.iterrows()]

    valid_nodes = set(df["inc_code"].astype(str))  # ONLY real persons

    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G.subgraph(valid_nodes).copy()  # drop edges to non-person coupons


def compute_network_size(
    G: nx.DiGraph, df: pd.DataFrame, seed_field: str, in_field: str, networksize_field: str
):
    if networksize_field not in df.columns:
        raise ValueError(f"networksize field '{networksize_field}' not in dataframe.")

    reported_map = (
        df.assign(
            inc_code=lambda d: d[in_field].where(
                d[in_field].notna() & (d[in_field].astype(str).str.strip() != ""),
                d[seed_field],
            )
        )
        .astype({"inc_code": str})
        .set_index("inc_code")[networksize_field]
        .apply(pd.to_numeric, errors="coerce")
        .to_dict()
    )

    seeds_in_df = set(df[seed_field].dropna().astype(str)) if seed_field in df.columns else set()

    ns = {}
    for node in G.nodes():
        out_deg = G.out_degree(node)  # only counts recruits who became real persons
        networksize_in_tree = out_deg + 1
        rep = reported_map.get(node)
        reported_networksize = int(rep) if pd.notna(rep) else networksize_in_tree
        ns[node] = {
            "is_seed": node in seeds_in_df,
            "networksize_in_tree": networksize_in_tree,
            "reported_networksize": reported_networksize,
        }
    return ns


def compute_wave(G, seed):
    wave = {seed: 0}
    for child in nx.bfs_tree(G, seed):
        if child == seed:
            continue
        parents = list(G.predecessors(child))
        wave[child] = wave[parents[0]] + 1 if parents else 0
    return wave


def bfs_subgraph_upto_wave(G, seed, max_wave=None):
    if max_wave is None:
        nodes = list(nx.bfs_tree(G, seed).nodes())
    else:
        nodes = [
            n
            for n, d in nx.single_source_shortest_path_length(G, seed).items()
            if d <= max_wave
        ]
    return G.subgraph(nodes).copy()


def seeds_from_prefix(df: pd.DataFrame, seed_field: str, site_prefix: str) -> set:
    if seed_field not in df.columns:
        return set()
    return set(
        df[df[seed_field].astype(str).str.startswith(site_prefix)][seed_field].astype(str)
    )


def max_wave_of_graph(G: nx.DiGraph, seeds) -> int:
    maxw = 0
    for s in seeds:
        if s in G:
            depths = nx.single_source_shortest_path_length(G, s)
            if depths:
                maxw = max(maxw, max(depths.values()))
    return maxw


# ==============================
# C) Layout: Layered (Tidy wave)
# ==============================
def _tree_children(G: nx.DiGraph, seed):
    T = nx.bfs_tree(G, seed)
    ch = {n: [] for n in T.nodes()}
    for u, v in T.edges():
        ch[u].append(v)
    return ch, T.nodes()


def _subtree_size(children, node):
    if not children.get(node):
        return 1
    return 1 + sum(_subtree_size(children, c) for c in children[node])


def _tidy_assign_x(children, node, depth, x_cursor, x_map, y_map, order_by="size"):
    kids = children.get(node, [])
    if order_by == "size":
        kids = sorted(kids, key=lambda k: (-_subtree_size(children, k), str(k)))
    else:
        kids = sorted(kids, key=lambda k: str(k))

    if not kids:
        x_map[node] = x_cursor[0]
        y_map[node] = depth
        x_cursor[0] += 1
    else:
        xs = []
        for k in kids:
            _tidy_assign_x(children, k, depth + 1, x_cursor, x_map, y_map, order_by=order_by)
            xs.append(x_map[k])
        x_map[node] = float(sum(xs)) / len(xs)
        y_map[node] = depth


def _layout_tidy_one_tree(G, seed, node_gap=1.0, layer_gap=1.0):
    children, nodes = _tree_children(G, seed)
    depth = nx.single_source_shortest_path_length(G, seed)

    x_map, y_map, x_cursor = {}, {}, [0]
    _tidy_assign_x(children, seed, 0, x_cursor, x_map, y_map)

    min_x = min(x_map.values()) if x_map else 0
    pos = {}
    for n in nodes:
        x = (x_map[n] - min_x) * node_gap
        y = -float(y_map.get(n, depth.get(n, 0))) * layer_gap  # seed at top (y=0)
        pos[n] = np.array([x, y], dtype=float)
    width = (max(x_map.values()) - min_x + 1) * node_gap if x_map else 0
    return pos, width


def layout_layered_tidy_forest(G, seeds, node_gap=1.2, layer_gap=1.0, tree_gap=4.0):
    if not seeds:
        seeds = [n for n in G.nodes() if G.in_degree(n) == 0] or list(G.nodes())[:1]

    pos_all = {}
    x_offset = 0.0
    for s in seeds:
        pos_t, w = _layout_tidy_one_tree(G, s, node_gap=node_gap, layer_gap=layer_gap)
        for n, (x, y) in pos_t.items():
            pos_all[n] = np.array([x + x_offset, y], dtype=float)
        x_offset += w + tree_gap
    return pos_all


def plot_graph_layered_tidy(
    G,
    ns,
    seed_info=None,
    title="Recruitment Tree (Layered · Tidy)",
    node_gap=1.2,
    layer_gap=1.0,
    tree_gap=4.0,
    edge_alpha=0.6,
    show_edges=True,
    jitter=0.0,
):
    seeds = [n for n in G.nodes() if ns.get(n, {}).get("is_seed")]
    pos = layout_layered_tidy_forest(
        G, seeds, node_gap=node_gap, layer_gap=layer_gap, tree_gap=tree_gap
    )

    if jitter and jitter > 0:
        for n in pos:
            pos[n][0] += np.random.uniform(-jitter, jitter)

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        marker=dict(
            size=9,
            color=["#d62728" if ns[n]["is_seed"] else "#1f77b4" for n in G.nodes()],
            line=dict(width=1, color="white"),
        ),
        text=[fmt_id(n) for n in G.nodes()],
        textposition="top center",
        hoverinfo="text",
        hovertext=[
            "ID: "
            + fmt_id(n)
            + f"<br>is_seed: {ns[n]['is_seed']}"
            + f"<br>networksize_in_tree: {ns[n]['networksize_in_tree']}"
            + f"<br>reported_networksize: {ns[n]['reported_networksize']}"
            + (
                f"<br><b>Seed</b>: Wave={seed_info[n]['wave']} N={seed_info[n]['n']}"
                if seed_info and ns[n]["is_seed"] and n in seed_info
                else ""
            )
            for n in G.nodes()
        ],
    )

    traces = []
    if show_edges and G.number_of_edges() > 0:
        traces.append(
            go.Scatter(
                x=sum([[pos[u][0], pos[v][0], None] for u, v in G.edges()], []),
                y=sum([[pos[u][1], pos[v][1], None] for u, v in G.edges()], []),
                mode="lines",
                line=dict(width=1, color=f"rgba(120,120,120,{edge_alpha})"),
                hoverinfo="none",
            )
        )
    traces.append(node_trace)

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
    )
    return fig


# ==============================
# D) UI (English, Tidy-only)
# ==============================
st.title("REDCap RDS Tree Automata")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Select source", ["REDCap API", "Upload file"], index=1)

    st.header("Layout")
    st.caption("Only Layered (Tidy wave) is available.")
    jitter_on = st.checkbox("Reduce overlap with small jitter", value=True)
    jitter_val = st.slider("Jitter amount (x-axis)", 0.0, 0.5, 0.15, 0.01)
    show_edges_tidy = st.checkbox("Show edges", value=True, key="show_edges_tidy")
    edge_alpha = st.slider("Edge opacity", 0.05, 1.0, 0.60, 0.05)

if "site_trees" not in st.session_state:
    st.session_state.site_trees = {}

default_in_field = "inpon_number"
default_seed_field = "seed_id"
default_out_fields = [f"outpons_{i}" for i in range(1, 6)]
default_networksize_field = "networksize"

# Source: REDCap
if source == "REDCap API":
    api_url = st.text_input(
        "REDCap API URL", value="https://mrprcbcw.hosts.jhmi.edu/redcap/api/"
    )
    api_token = st.text_input("API Token", value="", type="password")

    with st.expander("Field mapping (optional)"):
        in_field = st.text_input("Incoming coupon field", value=default_in_field)
        seed_field = st.text_input("Seed field", value=default_seed_field)
        out_fields_text = st.text_input(
            "Recruitment out fields (comma separated)", value=",".join(default_out_fields)
        )
        networksize_field = st.text_input(
            "Network size field", value=default_networksize_field
        )
        out_fields = [c.strip() for c in out_fields_text.split(",") if c.strip()]

    if st.button("Fetch from REDCap"):
        try:
            df = fetch_redcap(api_url, api_token)
            st.success(f"Fetched {len(df)} rows from REDCap")
            st.session_state.df = df
            st.session_state.mapping = {
                "in_field": in_field,
                "seed_field": seed_field,
                "out_fields": out_fields,
                "networksize_field": networksize_field,
            }
        except Exception as e:
            st.error(f"Failed to fetch from REDCap: {e}")

# Source: Upload
else:
    uploaded = st.file_uploader("Upload CSV/TSV/XLSX", type=["csv", "tsv", "xlsx"])
    if uploaded:
        try:
            df_up = read_uploaded_file(uploaded)
            st.write("Preview", df_up.head())

            schema = infer_schema(df_up)
            in_field = st.selectbox(
                "Incoming coupon field",
                df_up.columns,
                index=(
                    df_up.columns.get_loc(schema["in_field"])
                    if schema["in_field"] in df_up.columns
                    else 0
                ),
            )
            seed_field = st.selectbox(
                "Seed field",
                df_up.columns,
                index=(
                    df_up.columns.get_loc(schema["seed_field"])
                    if schema["seed_field"] in df_up.columns
                    else 0
                ),
            )
            networksize_field = st.selectbox(
                "Network size field",
                df_up.columns,
                index=(
                    df_up.columns.get_loc(schema["networksize_field"])
                    if schema["networksize_field"] in df_up.columns
                    else 0
                ),
            )
            out_fields = st.multiselect(
                "Recruitment out fields",
                list(df_up.columns),
                default=[c for c in schema["out_fields"] if c in df_up.columns],
            )

            if st.button("Use this uploaded file"):
                st.session_state.df = df_up
                st.session_state.mapping = {
                    "in_field": in_field,
                    "seed_field": seed_field,
                    "out_fields": out_fields,
                    "networksize_field": networksize_field,
                }
                st.success("Data and field mapping saved (used by Draw)")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

# Site options
add_site = st.checkbox("Add site-level recruitment")
if add_site:
    prefix_len = st.number_input(
        "Leading digits length for site code", min_value=1, max_value=10, value=3
    )
    site_prefix = st.text_input("Site prefix digits (e.g., 150):")
    site_name = st.text_input("Site name (e.g., Site A):")

# Draw
if st.button("Draw"):
    if "df" not in st.session_state or "mapping" not in st.session_state:
        st.error("Fetch from REDCap or upload a file and finish field mapping first.")
        st.stop()

    df = st.session_state.df
    in_field = st.session_state.mapping["in_field"]
    seed_field = st.session_state.mapping["seed_field"]
    out_fields = st.session_state.mapping["out_fields"]
    networksize_field = st.session_state.mapping["networksize_field"]

    needed = set([in_field, seed_field, networksize_field] + list(out_fields))
    missing = needed - set(df.columns)
    if missing:
        st.error(f"Missing fields: {', '.join(missing)}")
        st.stop()

    try:
        G_all = build_graph(df, in_field, seed_field, out_fields)
        ns_all = compute_network_size(G_all, df, seed_field, in_field, networksize_field)
    except Exception as e:
        st.error(f"Graph construction or network size computation failed: {e}")
        st.stop()

    seeds = [n for n in G_all.nodes() if ns_all[n]["is_seed"]]
    seed_info_all = {}
    for node in seeds:
        wave_map = compute_wave(G_all, node)
        seed_info_all[node] = {
            "wave": (max(wave_map.values()) if len(wave_map) else 0),
            "n": (len(wave_map) if len(wave_map) else 1),
        }
    st.session_state.G_all = G_all
    st.session_state.ns_all = ns_all
    st.session_state.seed_info_all = seed_info_all

    # Build site-level subtrees
    if add_site and site_prefix and site_name:
        try:
            site_seeds = seeds_from_prefix(df, seed_field, site_prefix)
            nodes_site = set()
            for s in site_seeds:
                if s in G_all:
                    nodes_site |= set(nx.bfs_tree(G_all, s).nodes())

            if nodes_site:
                G_site = G_all.subgraph(nodes_site).copy()
                ns_site = compute_network_size(G_site, df, seed_field, in_field, networksize_field)
                for n in G_site.nodes():
                    ns_site[n]["is_seed"] = (n in site_seeds)

                seed_info_site = {}
                for node in site_seeds:
                    if node in G_site:
                        wave_map = compute_wave(G_site, node)
                        seed_info_site[node] = {
                            "wave": (max(wave_map.values()) if wave_map else 0),
                            "n": (len(wave_map) if wave_map else 1),
                        }

                st.session_state.site_trees[site_name] = {
                    "graph": G_site,
                    "networksize": ns_site,
                    "seed_info": seed_info_site,
                }
            else:
                st.warning(
                    f"No seeds starting with prefix '{site_prefix}' were found in the data."
                )
        except Exception as e:
            st.warning(f"Failed to build site subtree: {e}")

# Show FULL tree
if "G_all" in st.session_state:
    G_all = st.session_state.G_all
    ns_all = st.session_state.ns_all
    seed_info_all = st.session_state.seed_info_all

    seeds_all = sorted([n for n in G_all.nodes() if ns_all[n]["is_seed"]], key=lambda x: fmt_id(x))
    focus_seed = st.selectbox(
        "Focus on seed (optional)", ["<All seeds>"] + [fmt_id(s) for s in seeds_all], index=0
    )

    fmt_to_raw = {fmt_id(s): s for s in seeds_all}
    G_view = G_all
    ns_view = ns_all
    if focus_seed != "<All seeds>":
        raw_seed = fmt_to_raw[focus_seed]
        max_allowed = seed_info_all.get(raw_seed, {}).get("wave", 0)
        max_wave = st.slider("Show up to wave (inclusive)", 0, int(max_allowed), int(max_allowed))
        G_view = bfs_subgraph_upto_wave(G_all, raw_seed, max_wave)
        ns_view = {n: ns_all[n] for n in G_view.nodes()}

    fig = plot_graph_layered_tidy(
        G_view,
        ns_view,
        seed_info_all,
        title="Full Recruitment (Layered · Tidy)",
        node_gap=1.2,
        layer_gap=1.0,
        tree_gap=4.0,
        edge_alpha=edge_alpha,
        show_edges=show_edges_tidy,
        jitter=(jitter_val if jitter_on else 0.0),
    )
    st.plotly_chart(fig, use_container_width=True)
    to_downloadable_html(fig, "full_tree_tidy")

    # Cleaning / export (current view)
    under_ids = [
        n
        for n in G_view.nodes()
        if ns_view[n]["networksize_in_tree"] > ns_view[n]["reported_networksize"]
    ]
    st.write(f"Underreported count: **{len(under_ids)}**")
    if under_ids:
        st.write("Underreported coupon IDs:", ", ".join(fmt_id(x) for x in under_ids))
    fix_under = st.checkbox("Fix underreported networksize", key="fix_under")
    if fix_under and not under_ids:
        st.warning("No records to fix underreported.")
        fix_under = False

    reported_s = pd.Series(
        [ns_view[n]["reported_networksize"] for n in G_view.nodes()],
        index=list(G_view.nodes()),
    )
    in_s = pd.Series(
        [ns_view[n]["networksize_in_tree"] for n in G_view.nodes()],
        index=list(G_view.nodes()),
    )
    fixed = in_s.where(in_s > reported_s, reported_s)

    vals = [ns_view[n]["reported_networksize"] for n in ns_view]
    pct = pd.Series(vals).quantile([0, 0.25, 0.5, 0.75, 1.0]).to_frame("reported_networksize")
    pct.index = [f"{int(q * 100)}%" for q in pct.index]
    st.write("Percentiles of reported_networksize")
    st.dataframe(pct)

    show_vals = fixed if fix_under else reported_s
    median_val = float(pd.Series(show_vals).replace(0, np.nan).dropna().median())
    impute_na0 = st.checkbox("Impute NA and 0", key="impute_na0")
    impute_val = st.number_input(
        "Imputation value for NA/0", min_value=1.0, step=1.0, value=median_val, format="%.2f", key="impute_val"
    )
    st.caption(f"Median networksize (current): {median_val:.2f}")

    cap = st.number_input("Set networksize cap:", min_value=1, step=1, format="%d", key="cap")
    apply_cap = st.checkbox("Apply cap to reported_networksize", key="apply_cap")

    capped = reported_s.where(reported_s <= cap, cap)
    fixed_then = fixed.where(fixed <= cap, cap)

    if fix_under and apply_cap and impute_na0:
        clean_tmp = fixed_then.copy().where(lambda s: (s.notna()) & (s != 0), impute_val)
        networksize_clean = clean_tmp
    elif fix_under and impute_na0:
        clean_tmp = fixed.copy().where(lambda s: (s.notna()) & (s != 0), impute_val)
        networksize_clean = clean_tmp
    elif apply_cap and impute_na0:
        clean_tmp = capped.copy().where(lambda s: (s.notna()) & (s != 0), impute_val)
        networksize_clean = clean_tmp
    elif impute_na0:
        clean_tmp = reported_s.copy().where(lambda s: (s.notna()) & (s != 0), impute_val)
        networksize_clean = clean_tmp
    elif fix_under and apply_cap:
        networksize_clean = fixed_then
    elif fix_under:
        networksize_clean = fixed
    elif apply_cap:
        networksize_clean = capped
    else:
        networksize_clean = reported_s

    networksize_clean = networksize_clean.clip(lower=1.0).fillna(1.0)

    df_clean = pd.DataFrame(
        {"id": [fmt_id(x) for x in networksize_clean.index], "networksize_clean": networksize_clean.values}
    )
    st.download_button(
        "Export cleaned networksize (current view)",
        data=df_clean.to_csv(index=False).encode("utf-8"),
        file_name="networksize_clean_current_view.csv",
        mime="text/csv",
    )

# ==============================
# E) Site-level trees (Tidy-only, with weights)
# ==============================
if st.session_state.site_trees:
    st.subheader("Site-Level Trees")
    to_del = []

    for site, info in st.session_state.site_trees.items():
        with st.expander(site):
            G_site = info["graph"]
            ns_site = info["networksize"]
            seed_info_site = info["seed_info"]

            seeds_site_all = sorted(
                [n for n in G_site.nodes() if ns_site[n]["is_seed"]], key=lambda x: fmt_id(x)
            )
            focus_seed_site = st.selectbox(
                f"[{site}] Focus on seed",
                ["<All seeds>"] + [fmt_id(s) for s in seeds_site_all],
                index=0,
                key=f"seed_{site}",
            )

            fmt_to_raw_site = {fmt_id(s): s for s in seeds_site_all}
            Gs, nss = G_site, ns_site
            seeds_for_h = seeds_site_all

            if focus_seed_site != "<All seeds>":
                raw_seed = fmt_to_raw_site[focus_seed_site]
                max_allowed = max_wave_of_graph(G_site, [raw_seed])
                max_wave_site = st.slider(
                    f"[{site}] Show up to wave (inclusive)",
                    0,
                    int(max_allowed),
                    int(max_allowed),
                    key=f"wave_{site}",
                )
                Gs = bfs_subgraph_upto_wave(G_site, raw_seed, max_wave_site)
                nss = {n: ns_site[n] for n in Gs.nodes()}
                seeds_for_h = [raw_seed]

            fig_site = plot_graph_layered_tidy(
                Gs,
                nss,
                seed_info_site,
                title=f"Tree: {site} (Layered · Tidy)",
                node_gap=1.2,
                layer_gap=1.0,
                tree_gap=4.0,
                edge_alpha=edge_alpha,
                show_edges=show_edges_tidy,
                jitter=(jitter_val if jitter_on else 0.0),
            )
            st.plotly_chart(fig_site, use_container_width=True)
            to_downloadable_html(fig_site, f"{site}_tree_tidy")

            st.markdown(f"- Participants: {Gs.number_of_nodes()}")
            st.markdown(f"- Seeds: {sum(nss[n]['is_seed'] for n in Gs.nodes())}")
            st.markdown(f"- Max wave: {max_wave_of_graph(Gs, seeds_for_h)}")

            # Cleaning (site)
            under_ids_site = [
                n
                for n in Gs.nodes()
                if nss[n]["networksize_in_tree"] > nss[n]["reported_networksize"]
            ]
            st.write(f"Underreported count: **{len(under_ids_site)}**")
            if under_ids_site:
                st.write("Underreported coupon IDs:", ", ".join(fmt_id(x) for x in under_ids_site))

            fix_under_site = st.checkbox(f"[{site}] Fix underreported networksize", key=f"fix_under_{site}")
            if fix_under_site and not under_ids_site:
                st.warning("No records to fix underreported.")
                fix_under_site = False

            reported_s_site = pd.Series(
                [nss[n]["reported_networksize"] for n in Gs.nodes()],
                index=list(Gs.nodes()),
            )
            in_s_site = pd.Series(
                [nss[n]["networksize_in_tree"] for n in Gs.nodes()],
                index=list(Gs.nodes()),
            )
            fixed_site = in_s_site.where(in_s_site > reported_s_site, reported_s_site)

            vals_site = [nss[n]["reported_networksize"] for n in nss]
            pct_site = pd.Series(vals_site).quantile([0, 0.25, 0.5, 0.75, 1.0]).to_frame("reported_networksize")
            pct_site.index = [f"{int(q * 100)}%" for q in pct_site.index]
            st.write("Percentiles of reported_networksize")
            st.dataframe(pct_site)

            show_vals_site = fixed_site if fix_under_site else reported_s_site
            median_site_val = float(pd.Series(show_vals_site).replace(0, np.nan).dropna().median())
            impute_na0_site = st.checkbox(f"[{site}] Impute NA and 0", key=f"impute_{site}")
            impute_val_site = st.number_input(
                f"[{site}] Imputation value for NA/0",
                min_value=1.0,
                step=1.0,
                value=median_site_val,
                format="%.2f",
                key=f"impute_val_{site}",
            )
            st.caption(f"[{site}] Median networksize (current): {median_site_val:.2f}")

            cap_site = st.number_input(f"[{site}] Set networksize cap:", min_value=1, step=1, format="%d", key=f"cap_{site}")
            apply_cap_site = st.checkbox(f"[{site}] Apply cap to reported_networksize", key=f"apply_cap_{site}")

            capped_site = reported_s_site.where(reported_s_site <= cap_site, cap_site)
            fixed_then_site = fixed_site.where(fixed_site <= cap_site, cap_site)

            if fix_under_site and apply_cap_site and impute_na0_site:
                clean_tmp = fixed_then_site.copy().where(lambda s: (s.notna()) & (s != 0), impute_val_site)
                networksize_clean_site = clean_tmp
            elif fix_under_site and impute_na0_site:
                clean_tmp = fixed_site.copy().where(lambda s: (s.notna()) & (s != 0), impute_val_site)
                networksize_clean_site = clean_tmp
            elif apply_cap_site and impute_na0_site:
                clean_tmp = capped_site.copy().where(lambda s: (s.notna()) & (s != 0), impute_val_site)
                networksize_clean_site = clean_tmp
            elif impute_na0_site:
                clean_tmp = reported_s_site.copy().where(lambda s: (s.notna()) & (s != 0), impute_val_site)
                networksize_clean_site = clean_tmp
            elif fix_under_site and apply_cap_site:
                networksize_clean_site = fixed_then_site
            elif fix_under_site:
                networksize_clean_site = fixed_site
            elif apply_cap_site:
                networksize_clean_site = capped_site
            else:
                networksize_clean_site = reported_s_site

            networksize_clean_site = networksize_clean_site.clip(lower=1.0).fillna(1.0)

            df_clean_site = pd.DataFrame(
                {"id": [fmt_id(x) for x in networksize_clean_site.index], "networksize_clean": networksize_clean_site.values}
            )
            st.download_button(
                f"[{site}] Export cleaned networksize",
                data=df_clean_site.to_csv(index=False).encode("utf-8"),
                file_name=f"{site}_networksize_clean.csv",
                mime="text/csv",
            )

            # Weights (site-level only)
            N_est_site = st.number_input(f"[{site}] Population estimate:", min_value=1, step=1, key=f"N_{site}")
            if st.button(f"[{site}] Run Gile's Weights", key=f"run_weight_{site}"):
                try:
                    id_list = list(Gs.nodes())
                    recruiter_lst = [
                        list(Gs.predecessors(n))[0] if list(Gs.predecessors(n)) else "0"
                        for n in id_list
                    ]
                    ns_int = [int(round(float(x))) for x in networksize_clean_site.reindex(id_list).tolist()]

                    df_w = pd.DataFrame({"id": id_list, "recruiter": recruiter_lst, "network.size": ns_int})
                    df_w["network.size"] = df_w["network.size"].astype(int)

                    st.write("network.size to R:", df_w["network.size"].tolist())

                    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as in_csv, \
                            tempfile.NamedTemporaryFile(mode="r", suffix=".csv", delete=False) as out_csv:
                        df_w.to_csv(in_csv.name, index=False)
                        cmd = [
                            get_rscript_path(),
                            os.path.join(os.path.dirname(__file__), "compute_rds_weights.R"),
                            in_csv.name,
                            out_csv.name,
                            str(int(N_est_site)),
                        ]
                        res = subprocess.run(cmd, capture_output=True, text=True)

                        if res.returncode != 0:
                            st.error("R script failed")
                            st.text(res.stderr)
                        else:
                            out_df = pd.read_csv(out_csv.name)
                            st.success("Weights computed")
                            st.dataframe(out_df)
                            st.download_button(
                                f"[{site}] Download weights",
                                data=out_df.to_csv(index=False).encode("utf-8"),
                                file_name=f"{site}_weights.csv",
                                mime="text/csv",
                            )
                except FileNotFoundError as fe:
                    st.error(f"compute_rds_weights.R not found or file error: {fe}")
                except Exception as e:
                    st.error(f"Failed to compute weights via R: {e}")

            # ---- SSPSE (site-level only) ----
            if st.button(f"[{site}] Run SSPSE", key=f"run_sspse_{site}"):
                try:
                    id_list = list(Gs.nodes())
                    recruiter_lst = [
                        list(Gs.predecessors(n))[0] if list(Gs.predecessors(n)) else "0"
                        for n in id_list
                    ]
                    ns_int = [int(round(float(x))) for x in networksize_clean_site.reindex(id_list).tolist()]

                    df_w = pd.DataFrame({"id": id_list, "recruiter": recruiter_lst, "network.size": ns_int})
                    df_w["network.size"] = df_w["network.size"].astype(int)

                    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as in_csv, \
                            tempfile.NamedTemporaryFile(mode="r", suffix=".csv", delete=False) as out_csv:
                        df_w.to_csv(in_csv.name, index=False)
                        cmd = [
                            get_rscript_path(),
                            os.path.join(os.path.dirname(__file__), "compute_sspse.R"),
                            in_csv.name,
                            out_csv.name,
                            str(int(N_est_site)),
                        ]
                        res = subprocess.run(cmd, capture_output=True, text=True)

                        if res.returncode != 0:
                            st.error("SSPSE R script failed")
                            st.text(res.stderr)
                        else:
                            out_df = pd.read_csv(out_csv.name, index_col=0)
                            st.success("✅ SSPSE population size estimation completed successfully")
                            st.dataframe(out_df)
                            st.session_state[f"sspse_result_{site}"] = out_df

                            if os.path.exists("sspse_plot.png"):
                                st.image("sspse_plot.png", caption=f"[{site}] Posterior population size distribution (SSPSE)")
                            if os.path.exists("sspse_visibility.png"):
                                st.image("sspse_visibility.png", caption=f"[{site}] Visibility plot (SSPSE)")
                except Exception as e:
                    st.error(f"Failed to compute SSPSE for {site}: {e}")

        # ---- Delete site ----
        if st.button(f"Delete {site}", key=f"del_{site}"):
            to_del.append(site)

    for s in to_del:
        st.session_state.site_trees.pop(s, None)


# ==============================
# G) Generate Research Report (PDF)
# ==============================
import sys
import math

# ---- reportlab (graceful fallback) ----
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    reportlab_available = True
except ImportError:
    reportlab_available = False

st.markdown("## Generate Research Report")

# Small helpers that DO NOT change earlier app logic
def _series_clean_from_options(reported_s, in_s, fix_under=False, apply_cap=False, cap_val=None,
                               impute_na0=False, impute_val=None):
    """
    Reproduce the same cleaning pipeline as UI, so report reflects exactly what user selected.
    """
    # 1) fix underreported
    fixed = in_s.where(in_s > reported_s, reported_s)

    # choose base
    base = fixed if fix_under else reported_s

    # 2) cap
    if apply_cap and cap_val is not None:
        try:
            if not math.isnan(float(cap_val)):
                base = base.where(base <= cap_val, cap_val)
        except Exception:
            pass

    # 3) impute NA/0
    if impute_na0:
        tmp = base.copy()
        tmp = tmp.where((tmp.notna()) & (tmp != 0), impute_val)
        base = tmp

    # final safety
    base = base.clip(lower=1.0).fillna(1.0)
    return base


def _kv_table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths or [220, 300])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _df_table(df, max_rows=50, max_total_width=500, col_width=None):
    """
    Render a DataFrame as a reportlab Table that automatically fits within page width.
    Backward-compatible: accepts old col_width argument but ignores it (auto width).
    """
    if df is None or len(df) == 0:
        data = [["(No data available)"]]
        t = Table(data)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        return t

    show_df = df.copy()
    if len(show_df) > max_rows:
        show_df = show_df.head(max_rows)

    data = [list(show_df.columns)] + show_df.astype(object).values.tolist()

    n_cols = len(show_df.columns)
    if n_cols == 0:
        n_cols = 1

    max_total_width = min(max_total_width, 520)
    col_width_auto = max_total_width / n_cols
    col_widths = [col_width_auto] * n_cols

    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))

    return t


def _weights_for_graph(G, networksize_clean_series, N, rscript_path, script_name="compute_rds_weights.R"):
    """
    Compute Gile's SS weights synchronously (just for the report), does not alter previous app state.
    """
    try:
        id_list = list(G.nodes())
        recruiter_lst = [
            (list(G.predecessors(n))[0] if list(G.predecessors(n)) else "0")
            for n in id_list
        ]
        ns_int = [int(round(float(x))) for x in networksize_clean_series.reindex(id_list).tolist()]

        df_w = pd.DataFrame({"id": id_list, "recruiter": recruiter_lst, "network.size": ns_int})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as in_csv, \
                tempfile.NamedTemporaryFile(mode="r", suffix=".csv", delete=False) as out_csv:
            df_w.to_csv(in_csv.name, index=False)
            cmd = [
                get_rscript_path(),
                os.path.join(rscript_path, script_name),
                in_csv.name,
                out_csv.name,
                str(int(N)),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                return None, res.stderr
            out_df = pd.read_csv(out_csv.name)
            return out_df, None
    except Exception as e:
        return None, str(e)


def _sspse_for_graph(G, networksize_clean_series, N, rscript_path, script_name="compute_sspse.R"):
    """
    Compute SSPSE synchronously (just for the report), does not alter previous app state.
    """
    try:
        id_list = list(G.nodes())
        recruiter_lst = [
            (list(G.predecessors(n))[0] if list(G.predecessors(n)) else "0")
            for n in id_list
        ]
        ns_int = [int(round(float(x))) for x in networksize_clean_series.reindex(id_list).tolist()]

        df_w = pd.DataFrame({"id": id_list, "recruiter": recruiter_lst, "network.size": ns_int})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as in_csv, \
                tempfile.NamedTemporaryFile(mode="r", suffix=".csv", delete=False) as out_csv:
            df_w.to_csv(in_csv.name, index=False)
            cmd = [
                get_rscript_path(),
                os.path.join(rscript_path, script_name),
                in_csv.name,
                out_csv.name,
                str(int(N)),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                return None, res.stderr
            out_df = pd.read_csv(out_csv.name, index_col=0)
            return out_df, None
    except Exception as e:
        return None, str(e)


# ---- UI: controls for report content (non-invasive) ----
include_full = st.checkbox("Include Full-Tree section in report", value=True)
include_sites = st.checkbox("Include Site-level sections in report (if available)", value=True)
compute_inside_report = st.checkbox("Compute Weights & SSPSE during report generation (if not previously saved)", value=True)


# Where R scripts live (by default: same dir as this file)
rscript_dir = os.path.dirname(__file__)
weights_script_exists = os.path.exists(os.path.join(rscript_dir, "compute_rds_weights.R"))
sspse_script_exists = os.path.exists(os.path.join(rscript_dir, "compute_sspse.R"))

if not reportlab_available:
    st.error("⚠️ Reportlab is not installed. Please run `pip install reportlab` in your environment to enable PDF export.")
else:
    if st.button("Generate PDF Report"):
        # ---------- Start building PDF ----------
        buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(buffer.name, pagesize=A4)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12))
        story = []

        # ==== Title ====
        story.append(Paragraph("REDCap RDS Tree Automata — Analysis Report", styles["Title"]))
        story.append(Spacer(1, 14))

        # ==== Methods (high-level, research style) ====
        story.append(Paragraph("Methods", styles["Heading2"]))
        story.append(Paragraph(
            "We constructed directed recruitment trees from REDCap or uploaded data. A person is considered a node only if their coupon appears as an incoming coupon or as a seed. "
            "We computed an observed network size from the tree as out-degree plus one. Reported personal network size values were optionally adjusted using three cleaning rules: "
            "(i) fixing underreported values when observed size exceeds the reported value, "
            "(ii) imputing the median (or user-specified value) for missing/zero entries, and "
            "(iii) truncating extreme values at a user-defined cap.",
            styles["Normal"])
        )
        story.append(Paragraph(
            "For weighting, we applied Gile’s Successive Sampling (SS) estimator, which models recruitment as successive draws without replacement from a finite population. "
            "Inclusion probabilities depend on reported degree, sample size, and an assumed population size N; weights are the inverse of inclusion probabilities. "
            "For hidden population size, we used SSPSE, which compares the empirical degree distribution to expectations under candidate population sizes and produces a Bayesian posterior with summary statistics and a density plot.",
            styles["Normal"])
        )
        story.append(Spacer(1, 10))

        # ==== Data & Mapping ====
        story.append(Paragraph("Data & Field Mapping", styles["Heading2"]))
        src = "REDCap API" if 'source' in locals() and source == "REDCap API" else "Uploaded file"
        n_rows = int(st.session_state.get("df", pd.DataFrame()).shape[0]) if "df" in st.session_state else 0
        mapping = st.session_state.get("mapping", {})
        map_rows = [["Source", src],
                    ["Rows", str(n_rows)],
                    ["Incoming coupon field", mapping.get("in_field", "")],
                    ["Seed field", mapping.get("seed_field", "")],
                    ["Network size field", mapping.get("networksize_field", "")],
                    ["Out fields", ", ".join(mapping.get("out_fields", []))]]
        story.append(_kv_table([["Item", "Value"]] + map_rows))
        story.append(Spacer(1, 10))

        # ==== Full-tree Section ====
        if include_full and ("G_all" in st.session_state) and ("ns_all" in st.session_state):
            G_all = st.session_state["G_all"]
            ns_all = st.session_state["ns_all"]
            seeds_all = [n for n in G_all.nodes() if ns_all[n]["is_seed"]]
            story.append(Paragraph("Full-Tree Summary", styles["Heading2"]))

            # Graph summary
            graph_rows = [
                ["Participants (nodes)", str(G_all.number_of_nodes())],
                ["Number of out-coupons", str(G_all.number_of_edges())],
                ["Seeds (count)", str(len(seeds_all))]
            ]
            story.append(_kv_table([["Metric", "Value"]] + graph_rows))
            story.append(Spacer(1, 6))

            # Build reported_s & in_s
            reported_s = pd.Series(
                [ns_all[n]["reported_networksize"] for n in G_all.nodes()],
                index=list(G_all.nodes())
            )
            in_s = pd.Series(
                [ns_all[n]["networksize_in_tree"] for n in G_all.nodes()],
                index=list(G_all.nodes())
            )

            # read UI options (full-tree)
            fix_under = bool(st.session_state.get("fix_under", False))
            impute_na0 = bool(st.session_state.get("impute_na0", False))
            impute_val = float(st.session_state.get("impute_val", pd.Series(reported_s).median()))
            apply_cap = bool(st.session_state.get("apply_cap", False))
            cap_val = st.session_state.get("cap", None)
            if cap_val is not None:
                try:
                    cap_val = int(cap_val)
                except Exception:
                    cap_val = None

            # Produce cleaned series (exactly like UI pipeline)
            cleaned_s = _series_clean_from_options(
                reported_s, in_s,
                fix_under=fix_under,
                apply_cap=apply_cap, cap_val=cap_val,
                impute_na0=impute_na0, impute_val=impute_val
            )

            # Summaries: RAW vs CLEANED
            raw_pct = pd.Series(reported_s).quantile([0, 0.25, 0.5, 0.75, 1.0]).to_frame("Raw reported")
            raw_pct.index = [f"{int(q * 100)}%" for q in raw_pct.index]
            cln_pct = pd.Series(cleaned_s).quantile([0, 0.25, 0.5, 0.75, 1.0]).to_frame("After cleaning")
            cln_pct.index = [f"{int(q * 100)}%" for q in cln_pct.index]

            both_pct = pd.concat([raw_pct, cln_pct], axis=1).reset_index().rename(columns={"index": "Percentile"})
            story.append(Paragraph("Network Size Distribution (Raw vs Cleaned)", styles["Heading3"]))
            story.append(_df_table(both_pct, col_width=120))
            story.append(Spacer(1, 6))

            # What was changed? (counts)
            n_under = int(((in_s > reported_s) & fix_under).sum())
            n_zero_na = int((((reported_s == 0) | reported_s.isna()) & impute_na0).sum())
            n_capped = int(((apply_cap) and (reported_s > cap_val)).sum()) if (apply_cap and cap_val is not None) else 0

            change_rows = [
                ["Fix underreported", f"{'ON' if fix_under else 'OFF'} (affected: {n_under})"],
                ["Impute NA/0", f"{'ON' if impute_na0 else 'OFF'} (affected: {n_zero_na}; value={impute_val})"],
                ["Cap", f"{'ON' if apply_cap else 'OFF'}" + (f" (cap={cap_val}, affected≈{n_capped})" if apply_cap and cap_val else "")]
            ]
            story.append(_kv_table([["Cleaning step", "Status / Impact"]] + change_rows))
            story.append(Spacer(1, 10))





       

        # ==== Site-level Sections ====
        if include_sites and st.session_state.get("site_trees"):
            story.append(Paragraph("Site-level Analyses", styles["Heading2"]))
            for site, info in st.session_state["site_trees"].items():
                story.append(Paragraph(f"Site: {site}", styles["Heading3"]))
                Gs = info["graph"]
                nss = info["networksize"]

                # Graph summary
                seeds_site = [n for n in Gs.nodes() if nss[n]["is_seed"]]
                site_rows = [
                    ["Participants (nodes)", str(Gs.number_of_nodes())],
                    ["Number of out-coupons", str(Gs.number_of_edges())],
                    ["Seeds (count)", str(len(seeds_site))]
                ]
                story.append(_kv_table([["Metric", "Value"]] + site_rows))
                story.append(Spacer(1, 6))

                # Build series
                reported_s_site = pd.Series([nss[n]["reported_networksize"] for n in Gs.nodes()], index=list(Gs.nodes()))
                in_s_site = pd.Series([nss[n]["networksize_in_tree"] for n in Gs.nodes()], index=list(Gs.nodes()))

                # read site UI options (keys were defined in your app)
                fix_under_site = bool(st.session_state.get(f"fix_under_{site}", False))
                impute_site = bool(st.session_state.get(f"impute_{site}", False))
                impute_val_site = float(st.session_state.get(f"impute_val_{site}", pd.Series(reported_s_site).median()))
                apply_cap_site = bool(st.session_state.get(f"apply_cap_{site}", False))
                cap_site = st.session_state.get(f"cap_{site}", None)
                if cap_site is not None:
                    try:
                        cap_site = int(cap_site)
                    except Exception:
                        cap_site = None

                cleaned_site = _series_clean_from_options(
                    reported_s_site, in_s_site,
                    fix_under=fix_under_site,
                    apply_cap=apply_cap_site, cap_val=cap_site,
                    impute_na0=impute_site, impute_val=impute_val_site
                )

                # RAW vs CLEAN percentiles
                raw_pct_s = pd.Series(reported_s_site).quantile([0, 0.25, 0.5, 0.75, 1.0]).to_frame("Raw reported")
                raw_pct_s.index = [f"{int(q * 100)}%" for q in raw_pct_s.index]
                cln_pct_s = pd.Series(cleaned_site).quantile([0, 0.25, 0.5, 0.75, 1.0]).to_frame("After cleaning")
                cln_pct_s.index = [f"{int(q * 100)}%" for q in cln_pct_s.index]
                both_pct_s = pd.concat([raw_pct_s, cln_pct_s], axis=1).reset_index().rename(columns={"index": "Percentile"})
                story.append(_df_table(both_pct_s, col_width=110))
                story.append(Spacer(1, 6))

                n_under_s = int(((in_s_site > reported_s_site) & fix_under_site).sum())
                n_zero_na_s = int((((reported_s_site == 0) | reported_s_site.isna()) & impute_site).sum())
                n_capped_s = int(((apply_cap_site) and (reported_s_site > cap_site)).sum()) if (apply_cap_site and cap_site is not None) else 0
                change_rows_s = [
                    ["Fix underreported", f"{'ON' if fix_under_site else 'OFF'} (affected: {n_under_s})"],
                    ["Impute NA/0", f"{'ON' if impute_site else 'OFF'} (affected: {n_zero_na_s}; value={impute_val_site})"],
                    ["Cap", f"{'ON' if apply_cap_site else 'OFF'}" + (f" (cap={cap_site}, affected≈{n_capped_s})" if apply_cap_site and cap_site else "")]
                ]
                story.append(_kv_table([["Cleaning step", "Status / Impact"]] + change_rows_s))
                story.append(Spacer(1, 6))

                # Population size for site (from your UI control key=f"N_{site}")
                N_site = st.session_state.get(f"N_{site}", None)
                if N_site is None:
                    story.append(Paragraph("⚠ Population size N was not provided for this site — weights skipped.", styles["Small"]))
                    continue


                # Weights (site)
                weights_df_site = st.session_state.get(f"gile_weights_{site}")
                weights_err_site = None
                if (weights_df_site is None) and compute_inside_report and weights_script_exists:
                    weights_df_site, weights_err_site = _weights_for_graph(
                        Gs, cleaned_site, N_site, rscript_dir, "compute_rds_weights.R"
                    )

                story.append(Paragraph("Gile’s SS Weights (Site)", styles["Heading4"]))
                if weights_df_site is not None:
                    try:
                        wdesc = weights_df_site["weight"].describe()
                        rows = [
                            ["N (weights)", str(len(weights_df_site))],
                            ["Mean", f"{wdesc['mean']:.3f}"],
                            ["Std", f"{wdesc['std']:.3f}"],
                            ["Min", f"{wdesc['min']:.3f}"],
                            ["Median (50%)", f"{weights_df_site['weight'].median():.3f}"],
                            ["Max", f"{wdesc['max']:.3f}"],
                        ]
                        story.append(_kv_table([["Statistic", "Value"]] + rows))
                    except Exception:
                        story.append(Paragraph("Weights computed but failed to summarize.", styles["Small"]))
                else:
                    msg = "No weights available." + (f" Error: {weights_err_site}" if weights_err_site else "")
                    story.append(Paragraph(msg, styles["Small"]))
                story.append(Spacer(1, 4))

                # SSPSE (site)
                story.append(Paragraph("SSPSE (Site)", styles["Heading4"]))
                sspse_df_site = st.session_state.get(f"sspse_result_{site}")
                sspse_err_site = None
                if (sspse_df_site is None) and compute_inside_report and sspse_script_exists:
                    sspse_df_site, sspse_err_site = _sspse_for_graph(
                        Gs, cleaned_site, N_site, rscript_dir, "compute_sspse.R"
                    )

                if sspse_df_site is not None:
                    story.append(_df_table(sspse_df_site.reset_index().rename(columns={"index": "Group"}), col_width=100))
                    story.append(Spacer(1, 2))

                    # ---- Posterior plot ----
                    plot_path = "sspse_plot.png"
                    if os.path.exists(plot_path):
                        try:
                            story.append(Image(plot_path, width=360, height=260))
                            story.append(Paragraph("Figure: Posterior population size distribution (SSPSE).", styles["Small"]))
                        except Exception:
                            story.append(Paragraph("Posterior plot exists but could not be embedded.", styles["Small"]))
                    else:
                        story.append(Paragraph("Posterior plot not found.", styles["Small"]))



        # ==== Discussion / Notes ====
        story.append(Paragraph("Discussion & Notes", styles["Heading2"]))
        story.append(Paragraph(
            "Cleaning decisions (fixing underreport, imputing NA/0 with the median, and capping extreme values) affect both weighting and SSPSE. "
            "Fixing underreport enforces logical consistency with observed recruitments; imputing NA/0 stabilizes estimators when degree is missing or invalid; "
            "capping controls leverage from extreme self-reports. Gile’s SS weights correct for differential inclusion by degree. "
            "SSPSE summarizes uncertainty about population size through a posterior distribution; credible intervals should be interpreted as the range of plausible sizes under the model. "
            "Site-level analyses are preferred when recruitment processes and degree distributions are heterogeneous across sites.",
            styles["Normal"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "Limitations: estimates depend on degree reporting accuracy, recruitment trace completeness, and the chosen prior population size. "
            "Sensitivity analyses over N and cleaning thresholds are recommended.",
            styles["Small"]))

        # ---- Build and offer download ----
        doc.build(story)
        with open(buffer.name, "rb") as f:
            st.download_button(
                "⬇️ Download Research Report (PDF)",
                f.read(),
                file_name="rds_analysis_report.pdf",
                mime="application/pdf"
            )

    # --------------------------------------
    # FOOTNOTE below the report
    # --------------------------------------
    st.markdown("""
    <br><br>
    <div style='font-size:14px; color:#666'>
    ⚠️ <b>Note:</b> SS-PSE and Gile's SS weights are <b>not recommended</b> for full pooled datasets.  
    For valid estimation, these methods should be applied at the <b>site level</b>, not the combined dataset.
    </div>
    """, unsafe_allow_html=True)

# ===========================================
# TAB: Model Fit Guide (Good vs Bad Fit)
# ===========================================

fit_tab = st.tabs(["Model Fit Guide (Good vs Bad)"])[0]

with fit_tab:

    st.header("📘 How to Evaluate SS-PSE Model Fit")

    st.markdown("""
    The figures below explain how to evaluate whether the **posterior** (solid curve) 
    and **prior** (dashed curve) indicate a *good* or *bad* SS-PSE model fit.

    A good model fit should show:

    - ✔ Prior and posterior curves overlap **moderately**
    - ✔ Posterior is **not wildly different** from prior
    - ✔ Posterior median stays within the high-density posterior region
    - ✔ Posterior mode shifts *somewhat*, but not drastically
    - ❌ Posterior should **NOT** completely ignore the prior
    - ❌ Posterior should **NOT** be many orders of magnitude away

    If the curves diverge strongly → **your prior assumption is likely incorrect**,  
    and you should adjust the median prior and rerun SS-PSE.
    """)

    # --------------------------------------
    # GOOD FIT EXAMPLE
    # --------------------------------------
    st.subheader("✅ Example of GOOD fit (GUA_MT)")

    st.image("POSTERIOR_GUA_MT.png",
             caption="Good fit example: Prior and posterior curves overlap well.")

    st.markdown("""
    ### Why this is a GOOD fit?

    - Posterior curve shifts but **still overlaps** with the prior.  
    - Prior information meaningfully contributes to the posterior.  
    - Sample visibility and recruitment patterns are consistent with the prior belief.  
    """)

    st.markdown("---")

    # --------------------------------------
    # BAD FIT EXAMPLE
    # --------------------------------------
    st.subheader("❌ Example of POOR fit (GUA_UDI)")

    st.image("POSTERIOR_GUA_UDI.png",
             caption="Bad fit example: Prior and posterior curves do NOT overlap.")

    st.markdown("""
    ### Why this is a BAD fit?

    - Prior (dashed) and posterior (solid) almost **do not overlap**.  
    - Posterior strongly diverges → **the median prior is likely incorrect**.  
    - The sample’s visibility and recruitment dynamics contradict prior belief.  

    📌 **In this case, adjust your prior and rerun SS-PSE**  
    Aim for a curve that moderately overlaps with the prior  
    (similar to the GOOD example above).
    """)

    # Optional PDF download
    pdf_path = "POSTERIOR_GUA_UDI.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download full PDF (GUA_UDI example)",
                data=pdf_file,
                file_name="POSTERIOR_GUA_UDI.pdf",
                mime="application/pdf"
            )





#streamlit run /Users/miyuanqi/Desktop/python/app7.py