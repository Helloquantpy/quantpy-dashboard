import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
from scipy import stats
from fpdf import FPDF, XPos, YPos
import datetime

# ============================================================
# CONFIGURATION
# ============================================================
SOCIETE = "Quantpy"
SLOGAN  = "Analyse et simulation forex pour les traders curieux"

MAJOR_PAIRS = {
    "EUR/USD": {"ticker": "EURUSD=X", "pip": 0.0001, "pip_val": 10},
    "USD/JPY": {"ticker": "USDJPY=X", "pip": 0.01,   "pip_val": 9.5},
    "GBP/USD": {"ticker": "GBPUSD=X", "pip": 0.0001, "pip_val": 10},
    "USD/CHF": {"ticker": "USDCHF=X", "pip": 0.0001, "pip_val": 10},
    "AUD/USD": {"ticker": "AUDUSD=X", "pip": 0.0001, "pip_val": 10},
    "USD/CAD": {"ticker": "USDCAD=X", "pip": 0.0001, "pip_val": 10},
    "NZD/USD": {"ticker": "NZDUSD=X", "pip": 0.0001, "pip_val": 10},
}

MINOR_PAIRS = {
    "EUR/GBP": {"ticker": "EURGBP=X", "pip": 0.0001, "pip_val": 10},
    "EUR/JPY": {"ticker": "EURJPY=X", "pip": 0.01,   "pip_val": 9.5},
    "EUR/CHF": {"ticker": "EURCHF=X", "pip": 0.0001, "pip_val": 10},
    "EUR/AUD": {"ticker": "EURAUD=X", "pip": 0.0001, "pip_val": 10},
    "EUR/CAD": {"ticker": "EURCAD=X", "pip": 0.0001, "pip_val": 10},
    "GBP/JPY": {"ticker": "GBPJPY=X", "pip": 0.01,   "pip_val": 9.5},
    "GBP/CHF": {"ticker": "GBPCHF=X", "pip": 0.0001, "pip_val": 10},
    "GBP/AUD": {"ticker": "GBPAUD=X", "pip": 0.0001, "pip_val": 10},
    "AUD/JPY": {"ticker": "AUDJPY=X", "pip": 0.01,   "pip_val": 9.5},
    "CAD/JPY": {"ticker": "CADJPY=X", "pip": 0.01,   "pip_val": 9.5},
    "CHF/JPY": {"ticker": "CHFJPY=X", "pip": 0.01,   "pip_val": 9.5},
}

EXOTIC_PAIRS = {
    "USD/MXN": {"ticker": "USDMXN=X", "pip": 0.0001, "pip_val": 10},
    "USD/TRY": {"ticker": "USDTRY=X", "pip": 0.0001, "pip_val": 10},
    "USD/ZAR": {"ticker": "USDZAR=X", "pip": 0.0001, "pip_val": 10},
    "USD/NOK": {"ticker": "USDNOK=X", "pip": 0.0001, "pip_val": 10},
    "USD/SEK": {"ticker": "USDSEK=X", "pip": 0.0001, "pip_val": 10},
    "USD/SGD": {"ticker": "USDSGD=X", "pip": 0.0001, "pip_val": 10},
    "USD/HKD": {"ticker": "USDHKD=X", "pip": 0.0001, "pip_val": 10},
    "USD/PLN": {"ticker": "USDPLN=X", "pip": 0.0001, "pip_val": 10},
    "EUR/NOK": {"ticker": "EURNOK=X", "pip": 0.0001, "pip_val": 10},
    "USD/BRL": {"ticker": "USDBRL=X", "pip": 0.0001, "pip_val": 10},
}

ALL_PAIRS = {**MAJOR_PAIRS, **MINOR_PAIRS, **EXOTIC_PAIRS}

BRAND  = "#002B5C"
ACCENT = "#0078C8"
GREEN  = "#00A86B"
RED    = "#D7263D"
BG     = "#0A0F1E"
CARD   = "#111827"
TEXT   = "#E8EDF5"
MUTED  = "#6B7A99"

# ============================================================
# MOTEUR D'ANALYSE
# ============================================================
def run_analysis(pair_name, nb_sim, jours, lot, capital):
    config  = ALL_PAIRS[pair_name]
    pip     = config["pip"]
    pip_val = config["pip_val"]

    data = yf.download(config["ticker"], period="1y", interval="1d", progress=False)
    df   = data[["Close"]].copy()
    df.columns = ["Close"]
    df["Rendement"] = df["Close"].pct_change()

    moyenne      = float(df["Rendement"].mean())
    volatilite   = float(df["Rendement"].std())
    dernier_prix = float(df["Close"].iloc[-1])

    df["MM20"] = df["Close"].rolling(20).mean()
    df["MM50"] = df["Close"].rolling(50).mean()
    delta       = df["Close"].diff()
    gain        = delta.where(delta > 0, 0).rolling(14).mean()
    loss        = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"]   = 100 - (100 / (1 + gain / loss))

    mm20 = float(df["MM20"].iloc[-1])
    mm50 = float(df["MM50"].iloc[-1])
    rsi  = float(df["RSI"].iloc[-1])

    sims = np.zeros((jours, nb_sim))
    for s in range(nb_sim):
        prix = dernier_prix
        for j in range(jours):
            prix = prix * (1 + np.random.normal(moyenne, volatilite))
            sims[j, s] = prix

    prix_finaux = sims[-1, :]
    p10 = float(np.percentile(prix_finaux, 10))
    p50 = float(np.percentile(prix_finaux, 50))
    p90 = float(np.percentile(prix_finaux, 90))
    prob_hausse = float(np.mean(prix_finaux > dernier_prix) * 100)
    prob_baisse = 100 - prob_hausse

    sl        = p10
    tp        = p90
    pips_sl   = abs(dernier_prix - sl) / pip
    pips_tp   = abs(tp - dernier_prix) / pip
    ratio_rr  = pips_tp / pips_sl if pips_sl > 0 else 0
    gain_lot  = pips_tp * pip_val * lot
    perte_lot = pips_sl * pip_val * lot
    gain_cap  = (tp - dernier_prix) / dernier_prix * capital
    perte_cap = (dernier_prix - sl) / dernier_prix * capital

    score    = 0
    tendance = "Haussiere" if mm20 > mm50 else "Baissiere"
    if mm20 > mm50:
        score += 25

    if 40 <= rsi <= 60:
        score += 25
        rsi_lbl = "Zone neutre"
    elif rsi < 35:
        score += 20
        rsi_lbl = "Survente"
    elif rsi > 65:
        score += 10
        rsi_lbl = "Surachat"
    else:
        score += 15
        rsi_lbl = "Zone intermediaire"

    if prob_hausse >= 65:     score += 25
    elif prob_hausse >= 55:   score += 15
    elif prob_hausse >= 45:   score += 10
    else:                     score += 5

    if ratio_rr >= 2.5:       score += 25
    elif ratio_rr >= 1.5:     score += 15
    elif ratio_rr >= 1.0:     score += 10
    else:                     score += 5

    confiance = "FORT" if score >= 80 else "MODERE" if score >= 55 else "FAIBLE"

    if prob_hausse >= 55 and mm20 > mm50:     signal = "LONG"
    elif prob_baisse >= 55 and mm20 < mm50:   signal = "SHORT"
    else:                                      signal = "NEUTRE"

    return {
        "df": df, "pair": pair_name,
        "prix": dernier_prix, "mm20": mm20, "mm50": mm50,
        "rsi": rsi, "rsi_lbl": rsi_lbl, "tendance": tendance,
        "prob_hausse": prob_hausse, "prob_baisse": prob_baisse,
        "sims": sims, "prix_finaux": prix_finaux,
        "p10": p10, "p50": p50, "p90": p90,
        "sl": sl, "tp": tp,
        "pips_sl": pips_sl, "pips_tp": pips_tp, "ratio_rr": ratio_rr,
        "gain_lot": gain_lot, "perte_lot": perte_lot,
        "gain_cap": gain_cap, "perte_cap": perte_cap,
        "score": score, "confiance": confiance, "signal": signal,
        "nb_sim": nb_sim, "jours": jours, "lot": lot, "capital": capital,
    }

# ============================================================
# APP DASH
# ============================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Quantpy"
)
server = app.server  # Requis pour Railway / gunicorn

app.layout = html.Div(
    style={"display": "flex", "flexDirection": "column", "height": "100vh",
           "fontFamily": "system-ui, sans-serif", "background": BG, "color": TEXT},
    children=[

        # HEADER
        html.Div(
            style={"background": f"linear-gradient(135deg, {BRAND} 0%, #001845 100%)",
                   "padding": "16px 24px", "borderBottom": f"2px solid {ACCENT}",
                   "display": "flex", "justifyContent": "space-between", "alignItems": "center"},
            children=[
                html.Div([
                    html.H1(SOCIETE, style={"color": "white", "margin": 0,
                        "fontSize": "24px", "fontWeight": "800",
                        "letterSpacing": "3px", "fontFamily": "Georgia, serif"}),
                    html.P(SLOGAN, style={"color": "#7BAFD4", "margin": 0,
                        "fontSize": "10px", "letterSpacing": "1px"})
                ]),
                html.Div([
                    html.Span("● LIVE", style={"color": GREEN, "fontSize": "11px",
                                               "fontWeight": "bold", "letterSpacing": "2px"}),
                    html.Span(f"  {datetime.datetime.now().strftime('%d/%m/%Y')}",
                              style={"color": MUTED, "fontSize": "11px"})
                ])
            ]
        ),

        # CORPS
        html.Div(
            style={"display": "flex", "flex": "1", "overflow": "hidden"},
            children=[

                # PANNEAU GAUCHE
                html.Div(
                    style={"width": "240px", "minWidth": "240px", "background": CARD,
                           "padding": "20px", "borderRight": "1px solid #1E2D45",
                           "overflowY": "auto", "display": "flex",
                           "flexDirection": "column", "gap": "12px"},
                    children=[
                        html.H3("PARAMETRES", style={"color": TEXT, "fontSize": "12px",
                            "fontWeight": "700", "letterSpacing": "2px",
                            "margin": "0 0 8px 0", "paddingBottom": "10px",
                            "borderBottom": "1px solid #1E2D45"}),

                        html.Label("Categorie", style={"color": MUTED, "fontSize": "11px"}),
                        dcc.Dropdown(id="categorie",
                            options=[{"label": "Majors",         "value": "major"},
                                     {"label": "Minors (Cross)", "value": "minor"},
                                     {"label": "Exotiques",      "value": "exotic"}],
                            value="major", clearable=False, style={"fontSize": "12px"}),

                        html.Label("Paire", style={"color": MUTED, "fontSize": "11px"}),
                        dcc.Dropdown(id="paire",
                            options=[{"label": k, "value": k} for k in MAJOR_PAIRS],
                            value="EUR/USD", clearable=False, style={"fontSize": "12px"}),

                        html.Hr(style={"borderColor": "#1E2D45", "margin": "4px 0"}),

                        html.Label("Simulations Monte Carlo", style={"color": MUTED, "fontSize": "11px"}),
                        dcc.Slider(id="nb-sim", min=100, max=1000, step=100, value=500,
                            marks={100: "100", 500: "500", 1000: "1k"},
                            tooltip={"placement": "bottom"}),

                        html.Label("Horizon (jours)", style={"color": MUTED, "fontSize": "11px"}),
                        dcc.Slider(id="jours", min=7, max=90, step=7, value=30,
                            marks={7: "7j", 30: "30j", 60: "60j", 90: "90j"},
                            tooltip={"placement": "bottom"}),

                        html.Hr(style={"borderColor": "#1E2D45", "margin": "4px 0"}),

                        html.Label("Taille du lot", style={"color": MUTED, "fontSize": "11px"}),
                        dcc.Input(id="lot", type="number", value=1.0, step=0.01, min=0.01,
                            style={"width": "100%", "padding": "8px", "background": "#1A2535",
                                   "border": "1px solid #2A3A55", "color": TEXT,
                                   "borderRadius": "6px", "fontSize": "12px"}),

                        html.Label("Capital (USD)", style={"color": MUTED, "fontSize": "11px"}),
                        dcc.Input(id="capital", type="number", value=10000, step=1000, min=100,
                            style={"width": "100%", "padding": "8px", "background": "#1A2535",
                                   "border": "1px solid #2A3A55", "color": TEXT,
                                   "borderRadius": "6px", "fontSize": "12px"}),

                        html.Button("ANALYSER", id="btn-analyser", n_clicks=0,
                            style={"width": "100%", "padding": "12px",
                                   "background": f"linear-gradient(135deg, {ACCENT}, {BRAND})",
                                   "color": "white", "border": "none", "borderRadius": "8px",
                                   "fontSize": "12px", "fontWeight": "700",
                                   "letterSpacing": "2px", "cursor": "pointer",
                                   "marginTop": "8px"}),

                        html.Button("TELECHARGER PDF", id="btn-pdf", n_clicks=0,
                            style={"width": "100%", "padding": "12px",
                                   "background": "transparent", "color": ACCENT,
                                   "border": f"1px solid {ACCENT}", "borderRadius": "8px",
                                   "fontSize": "12px", "fontWeight": "700",
                                   "cursor": "pointer"}),

                        dcc.Download(id="download-pdf"),
                        dcc.Store(id="store-result"),
                    ]
                ),

                # ZONE PRINCIPALE
                html.Div(
                    style={"flex": "1", "padding": "20px", "overflowY": "auto", "background": BG},
                    children=[

                        html.Div(id="kpi-cards", style={"display": "flex", "gap": "10px",
                                                         "marginBottom": "16px", "flexWrap": "wrap"}),

                        html.Div(
                            style={"display": "flex", "gap": "12px", "marginBottom": "12px"},
                            children=[
                                html.Div(
                                    style={"flex": "2", "background": CARD, "padding": "14px",
                                           "borderRadius": "10px", "border": "1px solid #1E2D45"},
                                    children=[
                                        html.P("PRIX & MOYENNES MOBILES", style={"color": MUTED,
                                            "fontSize": "10px", "letterSpacing": "1px", "margin": "0 0 6px 0"}),
                                        dcc.Graph(id="graph-prix", style={"height": "260px"},
                                                  config={"displayModeBar": False})
                                    ]
                                ),
                                html.Div(
                                    style={"flex": "1", "background": CARD, "padding": "14px",
                                           "borderRadius": "10px", "border": "1px solid #1E2D45"},
                                    children=[
                                        html.P("RSI (14 JOURS)", style={"color": MUTED,
                                            "fontSize": "10px", "letterSpacing": "1px", "margin": "0 0 6px 0"}),
                                        dcc.Graph(id="graph-rsi", style={"height": "260px"},
                                                  config={"displayModeBar": False})
                                    ]
                                ),
                            ]
                        ),

                        html.Div(
                            style={"display": "flex", "gap": "12px"},
                            children=[
                                html.Div(
                                    style={"flex": "1", "background": CARD, "padding": "14px",
                                           "borderRadius": "10px", "border": "1px solid #1E2D45"},
                                    children=[
                                        html.P("MONTE CARLO - TRAJECTOIRES", style={"color": MUTED,
                                            "fontSize": "10px", "letterSpacing": "1px", "margin": "0 0 6px 0"}),
                                        dcc.Graph(id="graph-mc", style={"height": "260px"},
                                                  config={"displayModeBar": False})
                                    ]
                                ),
                                html.Div(
                                    style={"flex": "1", "background": CARD, "padding": "14px",
                                           "borderRadius": "10px", "border": "1px solid #1E2D45"},
                                    children=[
                                        html.P("DISTRIBUTION DES PRIX J+N", style={"color": MUTED,
                                            "fontSize": "10px", "letterSpacing": "1px", "margin": "0 0 6px 0"}),
                                        dcc.Graph(id="graph-dist", style={"height": "260px"},
                                                  config={"displayModeBar": False})
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),

        # FOOTER
        html.Div(
            style={"background": "#050A14", "padding": "10px 24px",
                   "textAlign": "center", "borderTop": "1px solid #1E2D45"},
            children=[
                html.Span("Quantpy - Outil d'aide a la reflexion. Ne constitue pas un conseil en investissement.",
                          style={"color": MUTED, "fontSize": "10px", "letterSpacing": "1px"})
            ]
        ),
    ]
)

# ============================================================
# CALLBACKS
# ============================================================

@app.callback(
    Output("paire", "options"),
    Output("paire", "value"),
    Input("categorie", "value")
)
def update_paires(cat):
    mapping = {"major": MAJOR_PAIRS, "minor": MINOR_PAIRS, "exotic": EXOTIC_PAIRS}
    pairs   = mapping[cat]
    opts    = [{"label": k, "value": k} for k in pairs]
    return opts, list(pairs.keys())[0]


@app.callback(
    Output("kpi-cards",    "children"),
    Output("graph-prix",   "figure"),
    Output("graph-rsi",    "figure"),
    Output("graph-mc",     "figure"),
    Output("graph-dist",   "figure"),
    Output("store-result", "data"),
    Input("btn-analyser",  "n_clicks"),
    State("paire",   "value"),
    State("nb-sim",  "value"),
    State("jours",   "value"),
    State("lot",     "value"),
    State("capital", "value"),
    prevent_initial_call=False
)
def analyser(n, paire, nb_sim, jours, lot, capital):
    r = run_analysis(
        paire   or "EUR/USD",
        nb_sim  or 500,
        jours   or 30,
        lot     or 1.0,
        capital or 10000
    )

    score_col  = GREEN if r["score"] >= 80 else "#F5A623" if r["score"] >= 55 else RED
    signal_col = GREEN if r["signal"] == "LONG" else RED if r["signal"] == "SHORT" else "#F5A623"

    def kpi(label, value, sub="", color=TEXT):
        return html.Div(
            style={"background": CARD, "padding": "14px 18px", "borderRadius": "10px",
                   "border": "1px solid #1E2D45", "minWidth": "120px", "flex": "1"},
            children=[
                html.P(label, style={"color": MUTED, "fontSize": "9px", "letterSpacing": "1px",
                                     "margin": "0 0 4px 0", "textTransform": "uppercase"}),
                html.P(value, style={"color": color, "fontSize": "20px", "fontWeight": "800",
                                     "margin": 0, "fontFamily": "Georgia, serif"}),
                html.P(sub,   style={"color": MUTED, "fontSize": "9px", "margin": 0})
            ]
        )

    cards = [
        kpi("Prix",        f"{r['prix']:.4f}",       paire),
        kpi("Signal",      r["signal"],               r["confiance"],             signal_col),
        kpi("Score",       f"{r['score']}/100",       r["confiance"],             score_col),
        kpi("RSI",         f"{r['rsi']:.1f}",         r["rsi_lbl"]),
        kpi("Stop Loss",   f"{r['sl']:.4f}",          f"{r['pips_sl']:.0f} pips", RED),
        kpi("Take Profit", f"{r['tp']:.4f}",          f"{r['pips_tp']:.0f} pips", GREEN),
        kpi("R/R",         f"1:{r['ratio_rr']:.2f}",  "Risk/Reward"),
        kpi("Gain Pot.",   f"+{r['gain_lot']:.0f}$",  f"sur {lot} lot(s)",        GREEN),
    ]

    df    = r["df"]
    dates = df.index.astype(str).tolist()

    layout_base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=10), margin=dict(l=40, r=10, t=10, b=30),
        xaxis=dict(showgrid=False, color=MUTED),
        yaxis=dict(showgrid=True, gridcolor="#1E2D45", color=MUTED)
    )

    fig_prix = go.Figure(layout=go.Layout(**layout_base))
    fig_prix.add_trace(go.Scatter(x=dates, y=df["Close"].values.flatten(),
        mode="lines", line=dict(color=ACCENT, width=1.5), name="Prix"))
    fig_prix.add_trace(go.Scatter(x=dates, y=df["MM20"].values.flatten(),
        mode="lines", line=dict(color="#F5A623", width=1.5, dash="dot"), name="MM20"))
    fig_prix.add_trace(go.Scatter(x=dates, y=df["MM50"].values.flatten(),
        mode="lines", line=dict(color="#A855F7", width=1.5, dash="dot"), name="MM50"))
    fig_prix.update_layout(legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"))

    fig_rsi = go.Figure(layout=go.Layout(**layout_base))
    fig_rsi.add_trace(go.Scatter(x=dates, y=df["RSI"].values.flatten(),
        mode="lines", line=dict(color=ACCENT, width=1.5), name="RSI",
        fill="tozeroy", fillcolor="rgba(0,120,200,0.08)"))
    fig_rsi.add_hline(y=70, line_dash="dot", line_color=RED,   opacity=0.7)
    fig_rsi.add_hline(y=30, line_dash="dot", line_color=GREEN, opacity=0.7)
    fig_rsi.add_hline(y=50, line_dash="dot", line_color=MUTED, opacity=0.3)
    fig_rsi.update_yaxes(range=[0, 100])

    fig_mc = go.Figure(layout=go.Layout(**layout_base))
    sims   = r["sims"]
    step   = max(1, r["nb_sim"] // 80)
    for s in range(0, r["nb_sim"], step):
        fig_mc.add_trace(go.Scatter(y=sims[:, s], mode="lines",
            line=dict(color=ACCENT, width=0.3), opacity=0.15, showlegend=False))
    fig_mc.add_hline(y=r["p10"], line_color=RED,    line_dash="dash", line_width=1.5,
        annotation_text=f"P10: {r['p10']:.4f}", annotation_font_color=RED)
    fig_mc.add_hline(y=r["p50"], line_color=ACCENT, line_dash="dash", line_width=1.5,
        annotation_text=f"P50: {r['p50']:.4f}", annotation_font_color=ACCENT)
    fig_mc.add_hline(y=r["p90"], line_color=GREEN,  line_dash="dash", line_width=1.5,
        annotation_text=f"P90: {r['p90']:.4f}", annotation_font_color=GREEN)
    fig_mc.add_hline(y=r["sl"], line_color=RED,   line_width=2,
        annotation_text="SL", annotation_font_color=RED)
    fig_mc.add_hline(y=r["tp"], line_color=GREEN, line_width=2,
        annotation_text="TP", annotation_font_color=GREEN)

    fig_dist = go.Figure(layout=go.Layout(**layout_base))
    fig_dist.add_trace(go.Histogram(x=r["prix_finaux"], nbinsx=50,
        marker_color=ACCENT, opacity=0.7, name="Distribution"))
    fig_dist.add_vline(x=r["p10"],  line_color=RED,    line_dash="dash", line_width=2)
    fig_dist.add_vline(x=r["p50"],  line_color=ACCENT, line_dash="dash", line_width=2)
    fig_dist.add_vline(x=r["p90"],  line_color=GREEN,  line_dash="dash", line_width=2)
    fig_dist.add_vline(x=r["prix"], line_color="white", line_width=2)

    store = {
        "pair": r["pair"], "prix": r["prix"], "signal": r["signal"],
        "score": r["score"], "confiance": r["confiance"],
        "rsi": r["rsi"], "rsi_lbl": r["rsi_lbl"], "tendance": r["tendance"],
        "mm20": r["mm20"], "mm50": r["mm50"],
        "prob_hausse": r["prob_hausse"], "prob_baisse": r["prob_baisse"],
        "p10": r["p10"], "p50": r["p50"], "p90": r["p90"],
        "sl": r["sl"], "tp": r["tp"],
        "pips_sl": r["pips_sl"], "pips_tp": r["pips_tp"], "ratio_rr": r["ratio_rr"],
        "gain_lot": r["gain_lot"], "perte_lot": r["perte_lot"],
        "gain_cap": r["gain_cap"], "perte_cap": r["perte_cap"],
        "nb_sim": r["nb_sim"], "jours": r["jours"],
        "lot": r["lot"], "capital": r["capital"]
    }

    return cards, fig_prix, fig_rsi, fig_mc, fig_dist, store


@app.callback(
    Output("download-pdf", "data"),
    Input("btn-pdf", "n_clicks"),
    State("store-result", "data"),
    prevent_initial_call=True
)
def generer_pdf(n, data):
    if not data:
        return None

    COULEUR_BRAND = (0, 43, 92)
    COULEUR_GRIS  = (240, 240, 245)

    class QuantpyPDF(FPDF):
        def header(self):
            self.set_fill_color(*COULEUR_BRAND)
            self.rect(0, 0, 210, 18, "F")
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(255, 255, 255)
            self.set_xy(10, 4)
            self.cell(0, 6, SOCIETE, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(180, 210, 240)
            self.set_xy(10, 10)
            self.cell(0, 5, SLOGAN)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(180, 210, 240)
            self.set_xy(130, 6)
            self.cell(70, 5, datetime.datetime.now().strftime("%d/%m/%Y %H:%M"), align="R")
            self.set_text_color(0, 0, 0)
            self.ln(12)

        def footer(self):
            self.set_y(-12)
            self.set_fill_color(*COULEUR_BRAND)
            self.rect(0, 285, 210, 15, "F")
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(180, 210, 240)
            self.set_xy(10, 287)
            self.cell(130, 5, "Ce rapport est un outil d'aide a la reflexion. Ne constitue pas un conseil en investissement.")
            self.set_xy(140, 287)
            self.cell(60, 5, f"Page {self.page_no()}", align="R")

        def section_title(self, title):
            self.set_fill_color(*COULEUR_BRAND)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 11)
            self.cell(0, 9, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
            self.set_text_color(0, 0, 0)
            self.ln(3)

    pdf = QuantpyPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*COULEUR_BRAND)
    pdf.cell(0, 12, f"ANALYSE - {data['pair']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7,
        f"{data['nb_sim']} simulations | Horizon {data['jours']}j | Lot {data['lot']} | Capital {data['capital']} USD",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(8)

    pdf.section_title("SCORE DE CONFIANCE")
    score_color = (0, 150, 0) if data["score"] >= 80 else (180, 120, 0) if data["score"] >= 55 else (200, 0, 0)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*score_color)
    pdf.cell(0, 10, f"{data['score']}/100 - {data['confiance']} - Signal : {data['signal']}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.section_title("ANALYSE DETAILLEE")
    rows = [
        ("Prix actuel",        f"{data['prix']:.4f}"),
        ("Tendance MM",        f"{data['tendance']} (MM20:{data['mm20']:.4f} / MM50:{data['mm50']:.4f})"),
        ("RSI (14j)",          f"{data['rsi']:.1f} - {data['rsi_lbl']}"),
        ("Probabilite hausse", f"{data['prob_hausse']:.1f}%"),
        ("Probabilite baisse", f"{data['prob_baisse']:.1f}%"),
        ("P10 / P50 / P90",    f"{data['p10']:.4f} / {data['p50']:.4f} / {data['p90']:.4f}"),
        ("Stop Loss (P10)",    f"{data['sl']:.4f} ({data['pips_sl']:.0f} pips) | Risque: -{data['perte_lot']:.0f} USD"),
        ("Take Profit (P90)",  f"{data['tp']:.4f} ({data['pips_tp']:.0f} pips) | Gain: +{data['gain_lot']:.0f} USD"),
        ("Ratio Risk/Reward",  f"1:{data['ratio_rr']:.2f}"),
        ("Gain potentiel",     f"+{data['gain_cap']:.0f} USD sur {data['capital']} USD investis"),
        ("Risque maximum",     f"-{data['perte_cap']:.0f} USD sur {data['capital']} USD investis"),
    ]

    for j, (label, val) in enumerate(rows):
        fill = j % 2 == 0
        if fill:
            pdf.set_fill_color(*COULEUR_GRIS)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(70, 7, f"  {label}", fill=fill)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, val, fill=fill, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf_path = f"/tmp/rapport_{data['pair'].replace('/', '')}.pdf"
    pdf.output(pdf_path)
    return dcc.send_file(pdf_path)


if __name__ == "__main__":
    app.run(debug=False, port=8050)