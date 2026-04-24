"""
Resampling accuracy analysis
-----------------------------
Input:  CSV with columns  50_hz  and  32_hz  (20 s, no timestamp)
Output: resampling_report.html  — interactive Plotly report, paste into
        Confluence via the HTML macro (Insert > Other macros > HTML)

Usage:
    pip install pandas numpy scipy plotly
    python resampling_analysis.py --csv your_file.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import welch
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",      default="data.csv",               help="Path to input CSV")
parser.add_argument("--out",      default="resampling_report.html", help="Output HTML path")
parser.add_argument("--duration", type=float, default=20.0,         help="Signal duration in seconds")
parser.add_argument("--fs50",     type=int,   default=50,           help="High-rate sample rate")
parser.add_argument("--fs32",     type=int,   default=32,           help="Low-rate sample rate")
parser.add_argument("--col50",    default="50_hz",                  help="Column name for 50 Hz signal")
parser.add_argument("--col32",    default="32_hz",                  help="Column name for 32 Hz signal")
parser.add_argument("--session",  default="session11",              help="Session identifier shown in report")
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────────
df  = pd.read_csv(args.csv)
y50 = df[args.col50].dropna().values
y32 = df[args.col32].dropna().values

fs50, fs32 = args.fs50, args.fs32
dur        = args.duration

t50 = np.linspace(0, dur, len(y50))
t32 = np.linspace(0, dur, len(y32))

# ── Common grid (5x the higher rate) ─────────────────────────────────────────
fs_grid = fs50 * 5
t_grid  = np.linspace(0, dur, int(dur * fs_grid) + 1)

cs50 = CubicSpline(t50, y50)
cs32 = CubicSpline(t32, y32)
s50  = cs50(t_grid)
s32  = cs32(t_grid)

# ── Cross-correlation → align s32 onto s50 ───────────────────────────────────
def xcorr_peak(a, b):
    a  = (a - a.mean()) / (a.std() + 1e-12)
    b  = (b - b.mean()) / (b.std() + 1e-12)
    cc = np.correlate(a, b, mode="full") / len(a)
    lags_samples = np.arange(-(len(a) - 1), len(a))
    idx = int(np.argmax(cc))
    return float(lags_samples[idx] / fs_grid), float(cc[idx])

peak_lag, peak_corr = xcorr_peak(s50, s32)
print(f"Peak XC : {peak_corr:.4f}  at lag {peak_lag*1000:.2f} ms")

t_shifted   = np.clip(t_grid - peak_lag, t32[0], t32[-1])
s32_aligned = cs32(t_shifted)

# ── Metrics ───────────────────────────────────────────────────────────────────
error     = s50 - s32_aligned
abs_error = np.abs(error)
rmse      = float(np.sqrt(np.mean(error**2)))
mae       = float(np.mean(abs_error))
max_err   = float(np.max(abs_error))
mean50    = np.mean(s50)
ss_res    = np.sum(error**2)
ss_tot    = np.sum((s50 - mean50)**2)
r2        = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

print(f"RMSE    : {rmse:.6f}")
print(f"MAE     : {mae:.6f}")
print(f"Max err : {max_err:.6f}")
print(f"R2      : {r2:.6f}")

# ── PSD (Welch) ───────────────────────────────────────────────────────────────
f50w, p50 = welch(y50, fs=fs50, nperseg=min(256, len(y50) // 2))
f32w, p32 = welch(y32, fs=fs32, nperseg=min(256, len(y32) // 2))

nyq32      = fs32 / 2
psd_db_all = np.concatenate([10*np.log10(p50 + 1e-20), 10*np.log10(p32 + 1e-20)])
psd_ymin   = float(psd_db_all.min()) - 3
psd_ymax   = float(psd_db_all.max()) + 3

# ── Colours ───────────────────────────────────────────────────────────────────
BLUE  = "#2C2C2A"  # dark grey for 32 Hz
PINK  = "#D4537E"  # pink-red for 50 Hz
AMBER = "#BA7517"
GREEN = "#3B6D11"
GRAY  = "#888780"
BG    = "#ffffff"
PAPER = "#f7f6f3"
GRID  = "rgba(0,0,0,0.07)"

# ── Error y-axis scale ────────────────────────────────────────────────────────
# Use 4x the signal range so the error trace is clearly readable but
# visually smaller than the signals (occupies ~25% of the axis height).
sig_range = float(np.max(np.abs(np.concatenate([s50, s32_aligned]))))
err_mag   = sig_range * 3.0   # error axis half-range (3x signal range)

# ── Figure layout ─────────────────────────────────────────────────────────────
# Row 1: full-width dual-axis (signals left y, error right y, shared x)
# Row 2: col1 = KDE density,  col2 = overlaid PSD comparison
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"colspan": 2, "secondary_y": True}, None],
        [{"secondary_y": False},              {"secondary_y": False}],
    ],
    row_heights=[0.58, 0.42],
    vertical_spacing=0.18,
    horizontal_spacing=0.12,
)

# ── Row 1: signals + error ────────────────────────────────────────────────────
fig.add_trace(go.Scatter(x=t_grid, y=s32_aligned, name="32 Hz (aligned)",
                         line=dict(color=BLUE, width=1.5)),
              row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=t_grid, y=s50, name="50 Hz",
                         line=dict(color=PINK, width=1.5, dash="dot")),
              row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=t_grid, y=error, name="Error",
                         line=dict(color=AMBER, width=1.0),
                         fill="tozeroy", fillcolor="rgba(186,117,23,0.10)"),
              row=1, col=1, secondary_y=True)

fig.update_yaxes(
    title_text="Amplitude", secondary_y=False, row=1, col=1,
    showgrid=True, gridcolor=GRID, zeroline=False,
)
fig.update_yaxes(
    title_text="Error", secondary_y=True, row=1, col=1,
    range=[-err_mag, err_mag],
    rangemode="tozero",
    showgrid=False,
    zeroline=True, zerolinecolor=GRAY, zerolinewidth=1,
    tickfont=dict(color=AMBER),
    title_font=dict(color=AMBER),
)
fig.update_xaxes(title_text="Time (s)", row=1, col=1,
                 showgrid=True, gridcolor=GRID, zeroline=False)

# ── Row 2 left: KDE ──────────────────────────────────────────────────────────
kde   = gaussian_kde(abs_error, bw_method="scott")
x_kde = np.linspace(0, abs_error.max() * 1.05, 400)
fig.add_trace(go.Scatter(x=x_kde, y=kde(x_kde), name="Density |error|",
                         line=dict(color=GREEN, width=1.5),
                         fill="tozeroy", fillcolor="rgba(59,109,17,0.12)"),
              row=2, col=1)
fig.update_xaxes(title_text="Abs Error", row=2, col=1,
                 showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(title_text="Density", row=2, col=1,
                 showgrid=True, gridcolor=GRID, zeroline=False)

# ── Row 2 right: overlaid PSD ─────────────────────────────────────────────────
fig.add_trace(go.Scatter(x=f32w, y=10*np.log10(p32 + 1e-20), name="PSD 32 Hz",
                         line=dict(color=BLUE, width=1.5), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=f50w, y=10*np.log10(p50 + 1e-20), name="PSD 50 Hz",
                         line=dict(color=PINK, width=1.5, dash="dot"), showlegend=False), row=2, col=2)
fig.update_xaxes(title_text="Frequency (Hz)", range=[0, nyq32], row=2, col=2,
                 showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(title_text="dB", range=[psd_ymin, psd_ymax], row=2, col=2,
                 showgrid=True, gridcolor=GRID, zeroline=False)

# ── Subplot titles — placed manually to avoid colspan offset bug ──────────────
# Row 1 spans full width: centre at x=0.5
# Row 2: col1 centre ~0.225, col2 centre ~0.775 (with 0.12 h-spacing, each col ~0.44 wide)
title_style = dict(showarrow=False, xref="paper", yref="paper",
                   xanchor="center", yanchor="bottom",
                   font=dict(size=13, color="#2c2c2a"))

fig.add_annotation(x=0.5,   y=1.01,
                   text=f"<b>Signals & Error</b> (32 Hz shifted {peak_lag*1000:.2f} ms)",
                   **title_style)
fig.add_annotation(x=0.225, y=0.375,
                   text="<b>Abs Error Density (KDE)</b>",
                   **title_style)
fig.add_annotation(x=0.775, y=0.375,
                   text="<b>Power Spectral Density Comparison</b>",
                   **title_style)

# ── Metrics banner ────────────────────────────────────────────────────────────
details_text = (
    f"Session: <b>{args.session}</b>    "
    f"Duration: <b>{dur:.0f} s</b>"
)
metrics_text = (
    f"RMSE: <b>{rmse:.4f}</b>    "
    f"MAE: <b>{mae:.4f}</b>    "
    f"Max Error: <b>{max_err:.4f}</b>    "
    f"R\u00b2: <b>{r2:.5f}</b>    "
    f"XC Lag: <b>{peak_lag*1000:.2f} ms</b>"
)
banner_text = details_text + "<br>" + metrics_text
fig.add_annotation(
    xref="paper", yref="paper",
    x=0.5, y=1.10,
    text=banner_text,
    showarrow=False,
    align="center",
    bgcolor="#E6F1FB",
    bordercolor="#185FA5",
    borderwidth=1.2,
    font=dict(size=14, family="monospace", color="#042C53"),
    xanchor="center",
)

fig.update_layout(
    width=860,
    height=1000,
    title=dict(
        text="Resampling Accuracy Report \u2014 50 Hz vs 32 Hz - Half segment mirroring",
        font=dict(size=16),
        y=0.97,
        x=0.5,
        xanchor="center",
    ),
    paper_bgcolor=PAPER, plot_bgcolor=BG,
    showlegend=True,
    legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
    font=dict(family="Arial, sans-serif", size=11),
    margin=dict(t=155, b=90, l=55, r=65),
)

# ── Export + inject JS ────────────────────────────────────────────────────────
html_str = fig.to_html(
    include_plotlyjs="cdn", full_html=True,
    config={"displayModeBar": True, "scrollZoom": True},
)
html_str = html_str.replace(
    "</head>",
    "<style>body{margin:0;padding:0;} .plotly-graph-div{max-width:100%!important;}</style></head>"
)

# Double-click on PSD subplot resets x to [0, nyq32].
# With the new 2-row layout: xaxis=row1, xaxis2=row2col1(KDE), xaxis3=row2col2(PSD)
sync_js = """
<script>
(function() {
  var NYQ32 = NYQ32_VAL;
  var div = document.querySelector('.plotly-graph-div');
  div.on('plotly_doubleclick', function() {
    Plotly.relayout(div, {
      'xaxis3.range':     [0, NYQ32],
      'xaxis3.autorange': false,
    });
  });
})();
</script>
""".replace("NYQ32_VAL", str(float(nyq32)))

html_str = html_str.replace("</body>", sync_js + "\n</body>")

with open(args.out, "w", encoding="utf-8") as f:
    f.write(html_str)

print(f"\nSaved -> {args.out}")
print("Paste into Confluence: Insert > Other macros > HTML, then paste the full file content.")