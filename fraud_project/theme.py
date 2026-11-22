"""Forest green + cream visual system for the Streamlit dashboard."""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(251,247,240,0.55)",
    font=dict(color="#1A241E", family="Outfit, sans-serif", size=13),
    colorway=["#1B4332", "#40916C", "#C4A35A", "#8C2F39", "#52796F", "#9C7A32"],
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(251,247,240,0.8)", bordercolor="#E7DCC8", borderwidth=1),
)

COLORS = {
    "pine": "#143D2B",
    "forest": "#1B4332",
    "moss": "#2D6A4F",
    "leaf": "#40916C",
    "cream": "#F4EFE4",
    "ivory": "#FBF7F0",
    "sand": "#E7DCC8",
    "brass": "#C4A35A",
    "ink": "#1A241E",
    "burgundy": "#8C2F39",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Outfit:wght@300;400;500;600&display=swap');

:root {
  --pine: #143D2B;
  --forest: #1B4332;
  --moss: #2D6A4F;
  --leaf: #40916C;
  --sage: #95B8A0;
  --cream: #F4EFE4;
  --ivory: #FBF7F0;
  --sand: #E7DCC8;
  --brass: #C4A35A;
  --brass-dark: #9C7A32;
  --ink: #1A241E;
  --muted: #5C6B62;
  --burgundy: #8C2F39;
}

html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(1200px 500px at 10% -10%, rgba(64, 145, 108, 0.12), transparent 50%),
    radial-gradient(900px 400px at 100% 0%, rgba(196, 163, 90, 0.10), transparent 45%),
    linear-gradient(180deg, #FBF7F0 0%, #F4EFE4 55%, #EFE6D6 100%);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #143D2B 0%, #1B4332 55%, #244F3A 100%);
  border-right: 1px solid #C4A35A;
}

[data-testid="stSidebar"] * {
  color: #F4EFE4 !important;
}

[data-testid="stSidebar"] .stRadio label {
  padding: 0.45rem 0.7rem;
  border-radius: 8px;
}

[data-testid="stSidebar"] hr {
  border-color: rgba(196, 163, 90, 0.35);
}

.brand-mark {
  font-family: 'Cormorant Garamond', serif;
  letter-spacing: 0.18em;
  font-size: 0.78rem;
  color: #C4A35A;
  text-transform: uppercase;
  margin-bottom: 0.2rem;
}

.brand-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.85rem;
  line-height: 1.1;
  color: #FBF7F0;
  margin: 0 0 0.35rem 0;
}

.brand-sub {
  font-size: 0.82rem;
  color: #C9D8CE;
  line-height: 1.45;
  margin-bottom: 1.2rem;
}

.hero {
  background: linear-gradient(120deg, #143D2B 0%, #1B4332 48%, #2D6A4F 100%);
  color: #FBF7F0;
  border: 1px solid #C4A35A;
  border-radius: 18px;
  padding: 1.6rem 1.8rem 1.4rem;
  margin-bottom: 1.2rem;
  box-shadow: 0 18px 40px rgba(20, 61, 43, 0.18);
}

.hero h1 {
  font-family: 'Cormorant Garamond', serif;
  font-size: 2.4rem;
  font-weight: 600;
  margin: 0 0 0.4rem 0;
  color: #FBF7F0;
}

.hero p {
  margin: 0;
  max-width: 62ch;
  color: #DCE6DF;
  font-weight: 300;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.9rem;
  margin: 0.4rem 0 1.1rem;
}

.kpi {
  background: var(--ivory);
  border: 1px solid var(--sand);
  border-radius: 14px;
  padding: 1rem 1.05rem 0.95rem;
  box-shadow: 0 8px 24px rgba(26, 36, 30, 0.04);
  position: relative;
  overflow: hidden;
}

.kpi::after {
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  background: linear-gradient(180deg, var(--brass), var(--moss));
}

.kpi-label {
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.35rem;
}

.kpi-value {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.85rem;
  color: var(--pine);
  line-height: 1;
}

.kpi-hint {
  margin-top: 0.4rem;
  font-size: 0.78rem;
  color: var(--muted);
}

.panel {
  background: rgba(251, 247, 240, 0.78);
  border: 1px solid var(--sand);
  border-radius: 16px;
  padding: 1.1rem 1.2rem;
  margin-bottom: 1rem;
}

.badge {
  display: inline-block;
  padding: 0.18rem 0.6rem;
  border-radius: 999px;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.badge-critical { background: #8C2F39; color: #FBF7F0; }
.badge-elevated { background: #9C7A32; color: #FBF7F0; }
.badge-watch { background: #52796F; color: #FBF7F0; }
.badge-clear { background: #2D6A4F; color: #FBF7F0; }

.stButton>button {
  background: linear-gradient(180deg, #2D6A4F, #1B4332);
  color: #FBF7F0;
  border: 1px solid #C4A35A;
  border-radius: 10px;
  font-weight: 500;
  letter-spacing: 0.04em;
  padding: 0.45rem 1rem;
}

.stButton>button:hover {
  background: linear-gradient(180deg, #40916C, #1B4332);
  border-color: #d4b56c;
  color: #fff;
}

div[data-testid="stMetric"] {
  background: var(--ivory);
  border: 1px solid var(--sand);
  padding: 0.7rem 0.85rem;
  border-radius: 12px;
}

.stDataFrame, [data-testid="stTable"] {
  border: 1px solid var(--sand);
  border-radius: 12px;
}

@media (max-width: 1100px) {
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""
