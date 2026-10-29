"""Visual theme for the Streamlit app (website-style layout, no sidebar)."""

# Palette: ink #0b1437 | blue #3d5afe | sky #e8ecff | paper #f5f7ff | line #dfe4f5 | mint #12b886
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'DM Sans', -apple-system, 'Segoe UI', sans-serif !important;
}
h1, h2, h3, h4, .display { font-family: 'Sora', 'DM Sans', sans-serif !important; }
.stApp { background: #f5f7ff; color: #0b1437; }

/* Website feel: no sidebar, no Streamlit chrome */
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], #MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { max-width: 1080px; padding-top: 1rem; padding-bottom: 3rem; }

/* Navbar */
.nav { display: flex; align-items: center; justify-content: space-between; padding: 10px 4px 18px 4px; }
.nav-brand { display: flex; align-items: center; gap: 10px; font-family: 'Sora', sans-serif; font-weight: 700; font-size: 17px; color: #0b1437; }
.logo {
    width: 34px; height: 34px; border-radius: 10px; background: #3d5afe; color: #fff;
    display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 16px;
}
.status {
    display: flex; align-items: center; gap: 8px; background: #fff; border: 1px solid #dfe4f5;
    border-radius: 999px; padding: 7px 14px; font-size: 13px; font-weight: 600; color: #475075;
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: #b3bbd9; }
.status.on .dot { background: #12b886; box-shadow: 0 0 0 4px rgba(18,184,134,0.18); }
.status.on { color: #0b6b4f; }

/* Hero */
.hero {
    background-color: #0b1437;
    background-image: radial-gradient(rgba(255,255,255,0.09) 1.2px, transparent 1.2px);
    background-size: 22px 22px;
    border-radius: 24px; padding: 56px 48px 48px 48px; margin-bottom: 22px;
}
.hero h1 {
    font-size: 3.3rem !important; line-height: 1.05; font-weight: 800 !important;
    letter-spacing: -0.035em; color: #ffffff !important; margin: 0 0 16px 0; padding: 0;
}
.hero p { color: #b9c3ee; font-size: 1.12rem; max-width: 560px; margin: 0 0 28px 0; line-height: 1.55; }
.steps { display: flex; flex-wrap: wrap; gap: 10px; }
.step {
    display: flex; align-items: center; gap: 10px; background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14); border-radius: 999px; padding: 8px 16px 8px 8px;
    color: #e6eaff; font-size: 14px; font-weight: 600;
}
.step i {
    width: 24px; height: 24px; border-radius: 50%; background: #3d5afe; color: #fff;
    display: flex; align-items: center; justify-content: center; font-style: normal; font-size: 12px; font-weight: 700;
}

/* Upload panel */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff; border: 1px solid #dfe4f5 !important; border-radius: 20px;
}
[data-testid="stFileUploader"] section {
    background: #f5f7ff; border: 1.5px dashed #aab6f5; border-radius: 14px; padding: 22px;
}
.panel-title { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.05rem; margin-bottom: 4px; }
.panel-text { color: #5b6485; font-size: 0.93rem; margin-bottom: 12px; line-height: 1.5; }

/* Stat tiles */
.tile { background: #ffffff; border: 1px solid #dfe4f5; border-radius: 16px; padding: 16px 20px; }
.tile b { display: block; font-family: 'Sora', sans-serif; font-size: 1.7rem; font-weight: 700; color: #0b1437; }
.tile span { font-size: 0.85rem; color: #5b6485; font-weight: 500; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; border: none; padding: 0; }
.stTabs [data-baseweb="tab"] {
    height: 42px; border-radius: 999px; padding: 0 22px; background: #e8ecff;
    color: #3a4470; font-weight: 700; border: none;
}
.stTabs [aria-selected="true"] { background: #0b1437 !important; color: #ffffff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: #ffffff; color: #2f49e0; border: 1px solid #c3ccf7; border-radius: 12px;
    font-weight: 700; padding: 0.55rem 1.1rem; transition: all 0.15s ease;
}
.stButton > button p, .stDownloadButton > button p { color: inherit; }
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #e8ecff; border-color: #3d5afe; color: #2f49e0;
}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: #3d5afe; color: #ffffff; border: 1px solid #3d5afe;
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background: #2f49e0; border-color: #2f49e0; color: #ffffff;
}
.stButton > button:focus-visible { outline: 2px solid #3d5afe; outline-offset: 2px; }

.stButton > button { width: 100%; }
/* Inputs & chat */
[data-baseweb="input"] { border-radius: 12px !important; }
[data-testid="stChatInput"] { border-radius: 16px; border: 1px solid #c9d1f3; background: #fff; }
[data-testid="stChatMessage"] {
    background: #ffffff; border: 1px solid #dfe4f5; border-radius: 16px;
    padding: 14px 18px; margin-bottom: 12px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: #eef1ff; border-color: #d3dbfb; }
[data-testid="stExpander"] { background: #fafbff; border: 1px solid #dfe4f5 !important; border-radius: 12px; }

/* Empty state & headings */
.empty { background: #fff; border: 1px dashed #c3ccf7; border-radius: 18px; text-align: center; padding: 38px 24px; margin-bottom: 14px; }
.empty .icon { font-size: 30px; margin-bottom: 6px; }
.empty h4 { margin: 0 0 6px 0; font-size: 1.05rem; }
.empty p { margin: 0; color: #5b6485; font-size: 0.93rem; }
.section-title { font-family: 'Sora', sans-serif; font-size: 1.3rem; font-weight: 700; margin: 0 0 2px 0; }
.section-sub { color: #5b6485; font-size: 0.93rem; margin-bottom: 1rem; }
.foot { text-align: center; color: #8189aa; font-size: 0.85rem; margin-top: 2.5rem; }
/* Warnings & small captions */
.files { color: #5b6485; font-size: 0.88rem; margin-top: 8px; }
.lang-note { color: #8189aa; font-size: 0.85rem; margin: 6px 0 0 0; }
</style>
"""
