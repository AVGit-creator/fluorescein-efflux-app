import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objs as go
import math
import io

st.title("Fluorescein Efflux Single-Cell Kinetics")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# --- MODEL ---
def exp_decay(x, y0, A, Tau):
    return y0 + A * np.exp(-x / Tau)

def r_squared(y, y_fit):
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

# --- LOAD & CACHE ---
@st.cache_data
def load_csv_from_content(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes), header=None)
    df.columns = ["Time"] + [f"Cell {i}" for i in range(1, len(df.columns))]
    df["Time"] = pd.to_numeric(df["Time"], errors='coerce')
    df.iloc[:, 1:] = df.iloc[:, 1:].round(3)
    return df

@st.cache_data
def cached_fit_all_cells(df: pd.DataFrame):
    results = {}
    xdata = df["Time"].values.astype(float)
    for cell in df.columns[1:]:
        ydata = df[cell].values.astype(float)
        y0_init = ydata[-1]
        A_init = ydata[0] - y0_init
        Tau_init = 3
        p0 = [y0_init, A_init, Tau_init]

        try:
            popt, pcov = curve_fit(
                exp_decay, xdata, ydata, p0=p0,
                bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 1000]),
                maxfev=3000
            )
            y_fit = exp_decay(xdata, *popt)
            r2 = r_squared(ydata, y_fit)
            perr = np.sqrt(np.diag(pcov))
            stderr_tau = perr[2] if len(perr) > 2 else np.nan
            rel_err = stderr_tau / popt[2] if popt[2] != 0 else np.nan
            status = "Success" if r2 > 0.9 and rel_err < 0.3 else "Failed"
        except Exception:
            popt, y_fit, r2, status, stderr_tau = None, None, 0, "Failed", np.nan

        results[cell] = {
            "popt": popt,
            "y_fit": y_fit,
            "r2": r2,
            "status": status,
            "ydata": ydata,
            "stderr_tau": stderr_tau
        }
    return results

# --- PLOTTING ---
def create_trace(x, y, y_fit, cell, color):
    traces = [go.Scatter(x=x, y=y, mode='markers', name=f'{cell} Data', marker=dict(color=color, size=6))]    
    if y_fit is not None:
        traces.append(go.Scatter(x=x, y=y_fit, mode='lines', name=f'{cell} Fit', line=dict(color=color, dash='dash')))
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

# --- MAIN APP ---
if uploaded_file is not None:
    file_content = uploaded_file.getvalue()
    df = load_csv_from_content(file_content)
    fit_results = cached_fit_all_cells(df)

    succ = sum(1 for r in fit_results.values() if r["status"] == "Success")
    fail = sum(1 for r in fit_results.values() if r["status"] == "Failed")
    total = len(fit_results)
    c1, c2, c3 = st.columns(3)
    c1.metric("Successes", succ)
    c2.metric("Failures", fail)
    c3.metric("Total Cells", total)

    st.write("## Graph Options and Filters")
    rm_fail = st.checkbox("Remove failed fits from graph", True)
    match_y = st.checkbox("Match Y-axis scale for selected cells", True)
    all_overlay = st.checkbox("Overlay all cells", False)

    cells_for_graph = [cell for cell, res in fit_results.items() if not (rm_fail and res['status'] == 'Failed')]

    st.write("### Select cells to plot")
    sel = st.multiselect("Select cells", cells_for_graph, default=cells_for_graph[:2])
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
            fig.add_trace(go.Scatter(x=x, y=r['ydata'], mode='lines+markers', name=cell,
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
                for tr in create_trace(x, r['ydata'], r['y_fit'], cell, col):
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
                for tr in create_trace(x, r['ydata'], r['y_fit'], cell, 'blue'):
                    fig.add_trace(tr)
                y0, y1 = (y0g, y1g) if match_y else (min(r['ydata']), max(r['ydata']))
                fig.update_layout(xaxis=dict(title="Time", range=x_range, showline=True, linewidth=1,
                                             linecolor='black', mirror=True, ticks='outside', showgrid=False, zeroline=False),
                                  yaxis=dict(title="Intensity", showline=True, linewidth=1,
                                             linecolor='black', mirror=True, ticks='outside', showgrid=False,
                                             range=[y0, y1]),
                                  annotations=[dict(text=f"<b>{cell}</b>", x=0.5, y=1.12,
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
        Tau = r['popt'][2] if r['popt'] is not None else np.nan
        st_err = r['stderr_tau'] if r['stderr_tau'] is not None else np.nan
        ratio = st_err / Tau if Tau and Tau != 0 else np.nan
        if rm_high_err and not np.isnan(ratio) and ratio > 1:
            continue
        k_val = 1 / Tau if Tau and Tau != 0 else np.nan
        rows.append({
            "Cell": cell,
            "Tau": Tau,
            "STError": st_err,
            "STError/Tau": ratio,
            "k": k_val,
            "Status": r['status']
        })
    df_table = pd.DataFrame(rows)
    if sort_desc:
        df_table = df_table.sort_values(by="STError/Tau", ascending=False)
    st.write("### Fit Parameter Table")
    st.dataframe(df_table.style.format({"Tau": "{:.3f}", "STError": "{:.3f}", "STError/Tau": "{:.3f}", "k": "{:.4f}"}))

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

        k_filt = k_vals[k_vals <= max_k_cap]
        if len(k_filt) == 0:
            st.write("No k values less than or equal to the cap to plot.")
        else:
            use_manual = st.checkbox("Manually set number of bins", False)
            if use_manual:
                n_bins = st.slider("Number of bins", min_value=1, max_value=200, value=30)
                bin_edges = np.linspace(0.0, max_k_cap, n_bins + 1)
            else:
                non_zero_k_vals = k_filt[k_filt > 0]
                min_nonzero_k = np.min(non_zero_k_vals) if len(non_zero_k_vals) > 0 else max_k_cap
                bin_edges = [0.0, min_nonzero_k]
                bin_edges += list(np.linspace(min_nonzero_k, max_k_cap, num=30)[1:])
                bin_edges = np.round(bin_edges, 6)

            st.write(f"Number of histogram bins: {len(bin_edges) - 1}")
            counts, bins = np.histogram(k_filt, bins=bin_edges)
            widths = np.diff(bins)
            centres = (bins[:-1] + bins[1:]) / 2
            gap_fraction = 0.2
            bar_widths = widths * (1 - gap_fraction)

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

            # Add manual Y-axis override
            max_count = counts.max()
            y_padding = max(1, int(max_count * 0.1))
            default_y_max = max_count + y_padding
            manual_y_max = st.slider("Y-axis max (Count)", min_value=default_y_max, max_value=default_y_max * 5, value=default_y_max, step=1)

            fig_hist.update_layout(
                title="Histogram of kMDR Values",
                xaxis=dict(title="k<sub>MDR</sub> (min<sup>-1</sup>)", range=[0.0, max_k_cap]),
                yaxis=dict(title="Count", range=[0, manual_y_max]),
                height=450,
                bargap=0.2,
                margin=dict(l=50, r=40, t=40, b=60),
                hovermode="x unified"
            )

            st.plotly_chart(fig_hist, use_container_width=True)
