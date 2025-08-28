import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objs as go
import math
import io
import re

# =========================
# TOP-OF-PAGE: Equation Selector
# =========================
st.set_page_config(page_title="Fluorescein Efflux Single-Cell Kinetics", layout="centered")
st.title("Fluorescein Efflux Single-Cell Kinetics")

st.markdown("### Model Selection")
eq_mode = st.radio(
    "Choose the curve-fitting model:",
    [
        "Exponential decay (y = y0 + A * exp(-x / Tau))",
        "Exponential increase (y = Yb + A * (1 - exp(-x / Tau))",
        "Custom equation",
    ],
    index=0,
    help="Use a built-in model or supply your own equation in terms of x and parameters."
)

# --- Defaults / Inputs for Custom Equation ---
custom_eq = None
custom_params = []
custom_p0 = None
custom_bounds = None

def parse_float_list(s):
    """Parse a comma-separated string of floats into a list. Returns None if s is empty/blank."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    out = []
    for p in s.split(","):
        p = p.strip()
        if p == "":
            continue
        out.append(float(p))
    return out

if eq_mode == "Custom equation":
    with st.container():
        st.markdown("**Enter your equation in terms of `x` and your parameters.**")
        st.caption("Example: `y0 + A * np.exp(-x / Tau)` or `B / (1 + (x/x0)**n)`")
        custom_eq = st.text_input(
            "Equation (RHS only):",
            value="y0 + A * np.exp(-x / Tau)",
            help="Write only the right-hand side; `y =` is not needed. You can use NumPy via `np.`"
        )
        params_str = st.text_input(
            "Parameter names (comma-separated, order matters):",
            value="y0, A, Tau",
            help="List the parameter names you use in the equation, in the order they should be fit."
        )
        # --- Mini explanations for new users ---
        with st.expander("What do common parameters usually mean?"):
            st.markdown(
                "- **y0 / Yb**: Baseline intensity.\n"
                "- **y**: Fluorescence intensity at any given time point for a single cell.\n"
                "- **x**: Time.\n"
                "- **A**: Amplitude (magnitude of change).\n"
                "- **Tau**: Time constant controlling the rate.\n"
                "- **TD**: Delay/onset time (not used in the built-in models below).\n"
                "- **n**: Exponent controlling steepness (for Hill-like forms).\n"
                "- **k**: Rate constant."
            )

        p0_str = st.text_input(
            "Initial guesses p0 (comma-separated, optional):",
            value="",
            help="Optional initial guesses for parameters. Leave blank to auto-guess (if possible)."
        )
        lb_str = st.text_input(
            "Lower bounds (comma-separated, optional):",
            value="",
            help="Optional lower bounds. Leave blank for no lower bounds (i.e. -inf)."
        )
        ub_str = st.text_input(
            "Upper bounds (comma-separated, optional):",
            value="",
            help="Optional upper bounds. Leave blank for no upper bounds (i.e. +inf)."
        )

        # Parse param names
        custom_params = [p.strip() for p in params_str.split(",") if p.strip()]

        # Parse p0 and bounds
        custom_p0 = parse_float_list(p0_str)
        lb = parse_float_list(lb_str)
        ub = parse_float_list(ub_str)

        if lb is None and ub is None:
            custom_bounds = None
        else:
            npar = len(custom_params)
            if lb is None:
                lb = [-np.inf] * npar
            if ub is None:
                ub = [np.inf] * npar
            if len(lb) < npar:
                lb = lb + [-np.inf] * (npar - len(lb))
            if len(ub) < npar:
                ub = ub + [np.inf] * (npar - len(ub))
            lb = lb[:npar]
            ub = ub[:npar]
            custom_bounds = (lb, ub)

# =========================
# MODEL HELPERS
# =========================
def exp_decay(x, y0, A, Tau):
    # True decay shape is enforced by A >= 0 in bounds; curve rises toward y0 if A < 0, so we forbid that.
    return y0 + A * np.exp(-x / Tau)

def exp_increase_nodelay(x, Yb, A, Tau):
    # Rise from Yb at t=0 with no delay
    return Yb + A * (1.0 - np.exp(-x / Tau))

def r_squared(y, y_fit):
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

_ALLOWED_NUMPY = {
    'np': np, 'numpy': np,
    'exp': np.exp, 'log': np.log, 'log10': np.log10, 'log2': np.log2,
    'sqrt': np.sqrt, 'abs': np.abs,
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'power': np.power, 'pow': np.power, 'where': np.where,
}

def make_custom_model(eq_str, param_names):
    """Build a callable f(x, *params) from an equation string and ordered parameter names."""
    if not eq_str or not param_names:
        raise ValueError("Custom equation and parameter names must be provided.")
    forbidden = ["__import__", "os.", "sys.", "open(", "exec(", "eval(", "globals(", "locals("]
    low = eq_str.lower().replace(" ", "")
    for fbd in forbidden:
        if fbd in low:
            raise ValueError("Forbidden token in custom equation.")

    def f(x, *theta):
        if len(theta) != len(param_names):
            raise ValueError("Parameter length mismatch in custom model.")
        local_ns = {'x': x}
        for name, val in zip(param_names, theta):
            local_ns[name] = val
        return eval(eq_str, {"__builtins__": {}}, {**_ALLOWED_NUMPY, **local_ns})
    # attach param names so we can find Tau later
    f._param_names = list(param_names)
    return f

# =========================
# LOAD & CACHE
# =========================
@st.cache_data
def load_csv_from_content(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes), header=None)
    df.columns = ["Time"] + [f"Cell {i}" for i in range(1, len(df.columns))]
    df["Time"] = pd.to_numeric(df["Time"], errors='coerce')
    df.iloc[:, 1:] = df.iloc[:, 1:].round(3)
    return df

@st.cache_data
def cached_fit_all_cells(df: pd.DataFrame, model_spec: dict):
    """
    Cache-friendly fitter. Reconstructs the model from a serializable spec.
    model_spec:
      - {'mode': 'exp'}
      - {'mode': 'exp_increase'}
      - {'mode': 'custom', 'eq': str, 'params': [..], 'p0': [..] or None,
         'bounds': (lb_list, ub_list) or None}
    """
    # Global ranges to set sane bounds for built-in models
    xdata_full = df["Time"].values.astype(float)
    x_min, x_max = float(np.nanmin(xdata_full)), float(np.nanmax(xdata_full))
    y_all = df.iloc[:, 1:].values.astype(float)
    if y_all.size:
        y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    else:
        y_min, y_max = 0.0, 1.0

    # Rebuild model + p0 builder + bounds from spec
    if model_spec['mode'] == 'exp':
        model_fn = exp_decay
        # Enforce true decay: A >= 0 ; Tau > 0
        bounds = ([-np.inf, 0.0, 0.1], [np.inf, np.inf, 1000])

        def p0_builder(xdata, ydata):
            y0_init = float(ydata[-1])
            A_init = max(float(ydata[0] - y0_init), 1e-6)  # nonnegative
            Tau_init = 3.0
            return [y0_init, A_init, Tau_init]

        tau_name = "Tau"
        param_names = ["y0", "A", "Tau"]

    elif model_spec['mode'] == 'exp_increase':
        model_fn = exp_increase_nodelay
        # No delay: Yb within observed range; A >= 0; Tau > 0
        bounds = (
            [y_min, 0.0, 0.1],     # [Yb, A, Tau] lower
            [y_max, np.inf, 1000]  # upper
        )

        def p0_builder(xdata, ydata):
            n = max(1, int(0.1 * len(ydata)))
            Yb_init = float(np.nanmean(ydata[:n])) if n > 0 else float(ydata[0])
            m = max(1, int(0.1 * len(ydata)))
            A_init = max(float(np.nanmean(ydata[-m:]) - Yb_init), 1e-6)
            Tau_init = 3.0
            return [Yb_init, A_init, Tau_init]

        tau_name = "Tau"
        param_names = ["Yb", "A", "Tau"]

    else:
        model_fn = make_custom_model(model_spec['eq'], model_spec['params'])
        bounds = model_spec.get('bounds', None)
        user_p0 = model_spec.get('p0', None)
        param_names = list(model_spec['params'])
        tau_name = "Tau" if "Tau" in param_names else None

        def p0_builder(xdata, ydata):
            if user_p0 is not None and len(user_p0) == len(param_names):
                return user_p0
            # Heuristics if p0 not provided
            p0_guess = []
            for name in param_names:
                if name.lower() == "y0":
                    p0_guess.append(float(ydata[-1]))
                elif name.lower() in ("yb",):
                    p0_guess.append(float(np.nanmean(ydata[:max(1, int(0.1*len(ydata)))])))
                elif name.lower() in ("a", "amp", "amplitude"):
                    p0_guess.append(float(ydata[-1] - ydata[0]))
                elif name.lower() in ("tau", "t", "tc", "timeconst"):
                    p0_guess.append(3.0)
                elif name.lower() in ("td", "delay", "onset"):
                    p0_guess.append(float(xdata[0]))
                else:
                    p0_guess.append(1.0)
            return p0_guess

    # Fit loop
    results = {}
    xdata = df["Time"].values.astype(float)
    for cell in df.columns[1:]:
        ydata = df[cell].values.astype(float)

        try:
            p0 = p0_builder(xdata, ydata)
            if bounds is None:
                popt, pcov = curve_fit(model_fn, xdata, ydata, p0=p0, maxfev=5000)
            else:
                popt, pcov = curve_fit(model_fn, xdata, ydata, p0=p0, bounds=bounds, maxfev=5000)

            y_fit = model_fn(xdata, *popt)
            r2 = r_squared(ydata, y_fit)
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else None

            # Map parameter names to values for easy access (Amplitude, etc.)
            param_map = {}
            if popt is not None:
                for i, name in enumerate(param_names):
                    if i < len(popt):
                        param_map[name] = popt[i]

            # Determine Tau-like parameter and its stderr (only meaningful if present)
            Tau_val, stderr_tau = np.nan, np.nan
            if tau_name is not None and tau_name in param_names:
                idx = param_names.index(tau_name)
                Tau_val = popt[idx] if popt is not None and idx < len(popt) else np.nan
                if perr is not None and idx < len(perr):
                    stderr_tau = perr[idx]

            rel_err = (stderr_tau / Tau_val) if (Tau_val not in [0, None] and not np.isnan(Tau_val)) else np.nan

            # Success rule
            status = "Success" if (r2 > 0.9 and (not np.isnan(rel_err) and rel_err < 0.3)) else "Failed"
        except Exception:
            popt, y_fit, r2, status, stderr_tau, Tau_val = None, None, 0, "Failed", np.nan, np.nan
            param_map = {}

        results[cell] = {
            "popt": popt,
            "params": param_map,        # for Amplitude lookup
            "y_fit": y_fit,
            "r2": r2,
            "status": status,
            "ydata": ydata,
            "stderr_tau": stderr_tau,
            "Tau_like": Tau_val
        }
    return results

# =========================
# PLOTTING
# =========================
def create_trace(x, y, y_fit, label, color):
    traces = [go.Scatter(x=x, y=y, mode='markers', name=f'{label} Data', marker=dict(color=color, size=6))]
    if y_fit is not None:
        traces.append(go.Scatter(x=x, y=y_fit, mode='lines', name=f'{label} Fit', line=dict(color=color, dash='dash')))
    return traces

def get_y_range_rounded(cells, fit_results, round_base=5, padding_fraction=0.1):
    all_y = []
    for cell in cells:
        res = fit_results[cell]
        all_y.extend(res["ydata"])
        if res.get("y_fit") is not None:
            all_y.extend(res["y_fit"])
    if not all_y:
        return None, None
    y_min = min(all_y)
    y_max = max(all_y)
    data_range = y_max - y_min
    padding = max(data_range * padding_fraction, 1) if data_range != 0 else 1
    y_min_r = math.floor((y_min - padding) / round_base) * round_base
    y_max_r = math.ceil((y_max + padding) / round_base) * round_base
    return y_min_r, y_max_r

# =========================
# HELPERS
# =========================
def sort_cells_ascending(cells):
    """
    Sorts by numeric suffix if present (e.g., 'Cell 2' < 'Cell 10').
    Non-numeric names are placed after numeric ones, in alphabetical order.
    """
    def key_fn(name):
        m = re.search(r'(\d+)$', name.strip())
        if m:
            return (0, int(m.group(1)), name)
        return (1, float('inf'), name.lower())
    return sorted(cells, key=key_fn)

# ---------- Excel-style column labels for cells ----------
def _excel_col_letter(n: int) -> str:
    """1 -> 'A', 2 -> 'B', ..., 27 -> 'AA'"""
    letters = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters

def cell_to_excel_letter(cell_name: str) -> str:
    """
    Returns the Excel-style column letter for a given 'Cell N'.
    Time is column 'A'. Therefore, Cell 1 maps to 'B', Cell 2 -> 'C', etc.
    """
    m = re.search(r'(\d+)$', cell_name.strip())
    if not m:
        return ""
    n = int(m.group(1))
    return _excel_col_letter(n + 1)  # +1 because 'A' is Time

def cell_with_letter(cell_name: str) -> str:
    """Return 'Cell N (Letter)' when a valid letter exists."""
    letter = cell_to_excel_letter(cell_name)
    return f"{cell_name} ({letter})" if letter else cell_name
# ---------------------------------------------------------

# =========================
# FILE UPLOAD (after model choice)
# =========================
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# =========================
# MAIN APP
# =========================
if uploaded_file is not None:
    file_content = uploaded_file.getvalue()
    df = load_csv_from_content(file_content)

    # Build a SERIALIZABLE model spec for caching
    if eq_mode.startswith("Exponential decay"):
        model_spec = {'mode': 'exp'}
        model_label = "y = y0 + A * exp(-x / Tau) (A ≥ 0)"
        is_exp_mode = True
    elif eq_mode.startswith("Exponential increase"):
        model_spec = {'mode': 'exp_increase'}
        model_label = "y = Yb + A * (1 - exp(-x / Tau)) (A ≥ 0)"
        is_exp_mode = True
    else:
        is_exp_mode = False
        if not custom_eq or not custom_params:
            st.error("Please provide a custom equation and parameter names.")
            st.stop()
        model_spec = {
            'mode': 'custom',
            'eq': custom_eq,
            'params': list(custom_params),
            'p0': list(custom_p0) if custom_p0 is not None else None,
            'bounds': custom_bounds  # tuple of lists or None
        }
        model_label = f"y = {custom_eq} | params: {', '.join(custom_params)}"

    # Fit all cells (cached by df contents + model_spec contents)
    fit_results = cached_fit_all_cells(df, model_spec)

    # Summary header with model info
    st.info(f"**Model in use:** {model_label}")

    # Metrics
    succ = sum(1 for r in fit_results.values() if r["status"] == "Success")
    fail = sum(1 for r in fit_results.values() if r["status"] == "Failed")
    total = len(fit_results)
    c1, c2, c3 = st.columns(3)
    c1.metric("Successes", succ)
    c2.metric("Failures", fail)
    c3.metric("Total Cells", total)

    # Graph controls
    st.write("## Graph Options and Filters")
    rm_fail = st.checkbox("Remove failed fits from graph", True)
    match_y = st.checkbox("Match Y-axis scale for selected cells", True)
    all_overlay = st.checkbox("Overlay all cells", False)

    cells_for_graph = [cell for cell, res in fit_results.items() if not (rm_fail and res['status'] == 'Failed')]

    # Cell dropdown shows Excel letters in built-in modes
    st.write("### Select cells to plot")
    if is_exp_mode:
        label_map = {cell_with_letter(c): c for c in cells_for_graph}
        options = list(label_map.keys())
        default_labels = [cell_with_letter(c) for c in cells_for_graph[:2]]
        sel_labels = st.multiselect("Select cells", options=options, default=default_labels)
        sel = [label_map[l] for l in sel_labels]
    else:
        sel = st.multiselect("Select cells", options=cells_for_graph, default=cells_for_graph[:2])

    mode = st.radio("Comparison mode", ["Overlay", "Side by side"], index=0)

    x = df["Time"].values.astype(float)
    x_min, x_max = x.min(), x.max()
    buf = (x_max - x_min) * 0.02 if x_max > x_min else 0.1
    x_range = [x_min - buf, x_max + buf]

    if all_overlay:
        st.write("### Overlay of All Cells")
        fig = go.Figure()
        for i, (cell, r) in enumerate(fit_results.items()):
            clr = f'rgba({(i*53)%255},{(i*97)%255},{(i*191)%255},0.5)'
            label = cell_with_letter(cell) if is_exp_mode else cell
            fig.add_trace(go.Scatter(x=x, y=r['ydata'], mode='lines+markers', name=label,
                                     line=dict(color=clr), marker=dict(size=4, opacity=0.6)))
        y0, y1 = get_y_range_rounded(fit_results.keys(), fit_results)
        fig.update_layout(title="Overlay of All Cells",
                          xaxis=dict(title="Time", range=x_range, zeroline=False),
                          yaxis=dict(title="Intensity", range=[y0, y1]),
                          height=600, margin=dict(l=40, r=40, t=50, b=50))
        st.plotly_chart(fig, use_container_width=True)
    elif sel:
        y0g, y1g = get_y_range_rounded(sel, fit_results)
        if mode == "Overlay":
            fig = go.Figure()
            colors = ['blue','red','green','orange','purple','brown','pink','gray']
            for cell, col in zip(sel, colors):
                r = fit_results[cell]
                label = cell_with_letter(cell) if is_exp_mode else cell
                for tr in create_trace(x, r['ydata'], r['y_fit'], label, col):
                    fig.add_trace(tr)
            fig.update_layout(xaxis=dict(title="Time", range=x_range, showline=True, linewidth=1,
                                         linecolor='black', mirror=True, ticks='outside', showgrid=False, zeroline=False),
                              yaxis=dict(title="Intensity", showline=True, linewidth=1,
                                         linecolor='black', mirror=True, ticks='outside', showgrid=False,
                                         range=[y0g, y1g] if match_y else None),
                              height=500, margin=dict(l=60, r=20, t=30, b=60))
            st.plotly_chart(fig, use_container_width=True)
        else:
            cols = st.columns(min(len(sel), 4))
            for idx, cell in enumerate(sel):
                r = fit_results[cell]
                fig = go.Figure()
                label = cell_with_letter(cell) if is_exp_mode else cell
                for tr in create_trace(x, r['ydata'], r['y_fit'], label, 'blue'):
                    fig.add_trace(tr)
                y0, y1 = (y0g, y1g) if match_y else (min(r['ydata']), max(r['ydata']))
                fig.update_layout(xaxis=dict(title="Time", range=x_range, showline=True, linewidth=1,
                                             linecolor='black', mirror=True, ticks='outside', showgrid=False, zeroline=False),
                                  yaxis=dict(title="Intensity", showline=True, linewidth=1,
                                             linecolor='black', mirror=True, ticks='outside', showgrid=False,
                                             range=[y0, y1]),
                                  annotations=[dict(text=f"<b>{label}</b>", x=0.5, y=1.12,
                                                    xref="paper", yref="paper", showarrow=False,
                                                    font=dict(size=16), xanchor='center')],
                                  height=400, margin=dict(l=60, r=20, t=60, b=50))
                cols[idx % len(cols)].plotly_chart(fig, use_container_width=True)

    # --- TABLE ---
    st.write("## Table Options and Filters")
    rm_fail_tab = st.checkbox("Remove failed fits from table", True, key="t1")
    sort_desc = st.checkbox("Sort table by STError/Tau descending", True, key="t2")
    rm_high_err = st.checkbox("Remove entries with STError/Tau > 1", True, key="t3")

    rows = []
    for cell, r in fit_results.items():
        if rm_fail_tab and r['status'] == 'Failed':
            continue
        Tau = r.get('Tau_like', np.nan)
        st_err = r['stderr_tau'] if r['stderr_tau'] is not None else np.nan
        ratio = st_err / Tau if Tau and Tau != 0 else np.nan
        if rm_high_err and not np.isnan(ratio) and ratio > 1:
            continue
        k_val = 1 / Tau if Tau and Tau != 0 and not np.isnan(Tau) else np.nan

        # Amplitude (A) when present
        amp_val = np.nan
        if isinstance(r.get("params"), dict) and "A" in r["params"]:
            amp_val = r["params"]["A"]

        # Show cell number with its Excel column letter (Cell 1 = B, Cell 2 = C, ...)
        excel_letter = cell_to_excel_letter(cell)
        cell_display = f"{cell} ({excel_letter})" if excel_letter else cell

        rows.append({
            "CellRaw": cell,          # keep the raw name for internal logic
            "Cell": cell_display,     # display name in the table
            "Tau": Tau,
            "STError": st_err,
            "STError/Tau": ratio,
            "k": k_val,
            "Amplitude": amp_val,
            "Status": r['status']
        })
    df_table = pd.DataFrame(rows)
    if sort_desc and "STError/Tau" in df_table.columns:
        df_table = df_table.sort_values(by="STError/Tau", ascending=False, na_position="last")
    st.write("### Fit Parameter Table")
    st.dataframe(
        df_table.drop(columns=["CellRaw"]).style.format(
            {"Tau": "{:.3f}", "STError": "{:.3f}", "STError/Tau": "{:.3f}", "k": "{:.4f}", "Amplitude": "{:.3f}"}
        )
    )

    # --- HISTOGRAM ---
    st.write("## Histogram of kMDR values")
    k_vals = df_table.loc[df_table['Status'] == 'Success', 'k'].dropna().values

    if len(k_vals) == 0:
        st.write("No successful fits with valid k values to plot histogram.")
    else:
        min_k, max_k = 0.0, float(np.max(k_vals))
        max_k_cap = st.slider(
            "Cap maximum k value for histogram (values above cap are excluded)",
            min_value=min_k, max_value=max_k, value=max_k, step=0.001
        )
        hist_color = st.color_picker("Select histogram bar color", value="#1f77b4")
        show_bars = st.checkbox("Show Histogram Bars", True)
        show_step = st.checkbox("Show Step Plot", True)

        # Use the raw cell names here to preserve sorting and mapping
        k_source = df_table.loc[
            (df_table['Status'] == 'Success') & df_table['k'].notna() & (df_table['k'] <= max_k_cap),
            ['CellRaw', 'k']
        ].rename(columns={'CellRaw': 'Cell'})
        k_filt = k_source['k'].values

        if len(k_filt) == 0:
            st.write("No k values less than or equal to the cap to plot.")
        else:
            # ---- NEW: explicit bin size controls ----
            bin_mode = st.radio(
                "Binning method",
                ["Auto (adaptive)", "By number of bins", "By bin width"],
                index=0,
                horizontal=True,
                help="Choose automatic bins, a fixed number of bins, or specify an exact bin width."
            )

            if bin_mode == "By number of bins":
                n_bins = st.slider("Number of bins", min_value=1, max_value=200, value=30)
                bin_edges = np.linspace(0.0, max_k_cap, n_bins + 1)

            elif bin_mode == "By bin width":
                # Ensure a sensible default width from data spread (30 bins target)
                default_width = max(max_k_cap / 30.0, 1e-6)
                bin_width = st.number_input(
                    "Bin width",
                    min_value=1e-6,
                    max_value=max(1e-6, float(max_k_cap)),
                    value=float(np.round(default_width, 6)),
                    step=0.0001,
                    format="%.6f",
                    help="Set the width of each bin. Bins start at 0 and extend to the cap."
                )
                # Guard against zero/NaN
                bin_width = float(bin_width) if (bin_width is not None and bin_width > 0) else default_width
                # Make sure last edge reaches (or slightly exceeds) max_k_cap
                last_edge = max_k_cap + (bin_width - (max_k_cap % bin_width)) % bin_width
                bin_edges = np.arange(0.0, last_edge + bin_width / 2, bin_width)

            else:  # Auto (adaptive) — original behavior
                non_zero_k_vals = k_filt[k_filt > 0]
                min_nonzero_k = np.min(non_zero_k_vals) if len(non_zero_k_vals) > 0 else max_k_cap
                bin_edges = [0.0, min_nonzero_k]
                bin_edges += list(np.linspace(min_nonzero_k, max_k_cap, num=30)[1:])
                bin_edges = np.round(bin_edges, 6)

            counts, bins = np.histogram(k_filt, bins=bin_edges)
            widths = np.diff(bins)
            centres = (bins[:-1] + bins[1:]) / 2
            gap_fraction = 0.2
            bar_widths = widths * (1 - gap_fraction)

            # Y-axis extension slider
            auto_ymax = int(np.max(counts)) if len(counts) else 10
            ymax_upper_bound = max(auto_ymax * 5, auto_ymax + 10)
            y_max_limit = st.slider(
                "Y-axis upper limit (extend if tall bins get clipped)",
                min_value=int(max(1, auto_ymax)),
                max_value=int(ymax_upper_bound),
                value=int(auto_ymax),
                step=1
            )

            fig_hist = go.Figure()
            if show_bars:
                fig_hist.add_trace(go.Bar(
                    x=centres,
                    y=counts,
                    width=bar_widths,
                    name="Histogram Bars",
                    marker_color=hist_color,
                    opacity=0.75
                ))

            if show_step:
                x_step = np.repeat(bins, 2)[1:-1]
                y_step = np.repeat(counts, 2)
                fig_hist.add_trace(go.Scatter(
                    x=x_step, y=y_step, mode='lines',
                    name="Step Plot", line=dict(color=hist_color, width=3)
                ))

            fig_hist.update_layout(
                title="Histogram of kMDR Values",
                xaxis=dict(title="k<sub>MDR</sub> (min<sup>-1</sup>)", range=[0.0, max_k_cap]),
                yaxis=dict(title="Count", range=[0, y_max_limit]),
                height=450,
                bargap=0.2,
                margin=dict(l=50, r=40, t=40, b=60),
                hovermode="x unified"
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            # --- Bin selector with range + center ---
            bin_labels = []
            for i in range(len(bins)-1):
                center = (bins[i] + bins[i+1]) / 2
                # Close last bin with ']', others with ')'
                if i == len(bins) - 2:
                    label = f"[{bins[i]:.4f}, {bins[i+1]:.4f}] | center: {center:.4f}"
                else:
                    label = f"[{bins[i]:.4f}, {bins[i+1]:.4f}) | center: {center:.4f}"
                bin_labels.append(label)

            chosen_bin = st.selectbox("Select a bin (range + center) to see cells", bin_labels)
            if chosen_bin:
                idx = bin_labels.index(chosen_bin)
                low, high = bins[idx], bins[idx+1]
                if idx == len(bins) - 2:
                    mask = (k_source['k'] >= low) & (k_source['k'] <= high)
                else:
                    mask = (k_source['k'] >= low) & (k_source['k'] < high)
                cells_in_bin = list(k_source.loc[mask, 'Cell'])

                # Vertical, numerically ascending list using raw names
                cells_sorted = sort_cells_ascending(cells_in_bin)
                st.markdown(f"**Cells in bin {chosen_bin}:**")
                if cells_sorted:
                    st.markdown("\n".join(
                        f"{i}. {cell} ({cell_to_excel_letter(cell)})"
                        for i, cell in enumerate(cells_sorted, start=1)
                    ))
                else:
                    st.markdown("None")

# Optional: quick cache reset button (handy after code edits to bounds/models)
with st.expander("Advanced"):
    if st.button("Clear cached data & fits"):
        st.cache_data.clear()
        st.success("Cleared Streamlit cache. Re-run to refit with current code.")
