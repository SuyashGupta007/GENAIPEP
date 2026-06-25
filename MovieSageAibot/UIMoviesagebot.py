from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import json, re
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

# ── Model & prompt (identical to original) ─────────────────────────────────────
model = ChatMistralAI(model="mistral-small-2603")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are MovieSage AI, an expert movie analyst and information extraction assistant.
    Your responsibilities:
    1. Carefully read and understand the entire movie description.
    2. Extract all important movie-related information.
    3. Generate a concise and engaging plot summary.
    4. Identify key characters, themes, and notable facts.
    5. Extract factual information only from the provided text.
    6. If information is not available, return null.
    7. Never hallucinate or invent details.
    8. Return ONLY valid JSON.
    9. Do not include markdown, explanations, comments, or additional text.
    Extract the following fields:
    - title, genre, director, writers, producers, cast, release_year, runtime
    - language, country, plot_summary, main_characters, themes
    - notable_facts, awards, box_office, rating, keywords
    """,
        ),
        (
            "human",
            "Analyze the following movie description and extract all relevant information.\n\nMovie Description:\n{movie_description}",
        ),
    ]
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MovieSage AI", page_icon="🎬", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: transparent !important;
    font-family: 'Space Grotesk', sans-serif;
    color: #dde8f0;
}
[data-testid="stAppViewContainer"] {
    background: #07000f !important;
    min-height: 100vh;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Animated bg mesh ── */
.bg-mesh {
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 60% 40% at 20% 10%, rgba(0,245,212,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 80% 80%, rgba(255,0,110,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 40% 60% at 60% 40%, rgba(157,78,221,0.05) 0%, transparent 60%);
}

/* ── Horizontal scan line sweep ── */
.scan-sweep {
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background: repeating-linear-gradient(
        180deg,
        transparent 0px, transparent 3px,
        rgba(0,245,212,0.012) 3px, rgba(0,245,212,0.012) 4px
    );
}

/* ── Main wrap ── */
.main-wrap {
    position: relative; z-index: 1;
    max-width: 900px; margin: 0 auto;
    padding: 0 1.5rem 6rem;
}

/* ══ HERO ══ */
.hero {
    padding: 3.5rem 0 3rem;
    text-align: center;
    position: relative;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.22em;
    text-transform: uppercase; color: #00f5d4;
    margin-bottom: 20px;
    opacity: 0.7;
}
/* Chromatic aberration — the signature element */
.glitch-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 7vw, 5rem);
    font-weight: 800;
    color: #e0f7ff;
    letter-spacing: -0.03em;
    line-height: 0.95;
    position: relative;
    display: inline-block;
    text-shadow:
        -3px 0 0 rgba(255,0,110,0.6),
         3px 0 0 rgba(0,245,212,0.6);
}
.glitch-title .accent { color: #00f5d4; }
.hero-sub {
    font-size: 0.85rem; color: #3a4a6a;
    margin-top: 14px; letter-spacing: 0.02em;
    font-family: 'JetBrains Mono', monospace;
}

/* ══ NAV BAR ══ */
.nav-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 18px;
    background: rgba(0,245,212,0.04);
    border: 1px solid rgba(0,245,212,0.1);
    border-radius: 12px;
    margin-bottom: 2rem;
}
.nav-left {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #3a4a6a;
    letter-spacing: 0.1em;
}
.nav-dot { display: inline-block; width:7px; height:7px; border-radius:50%; margin-right:8px; background:#00f5d4; animation: pulse 2s infinite; }
.nav-right {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #3a4a6a;
    letter-spacing: 0.08em;
}
.nav-right span { color: #ff006e; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

/* ══ INPUT CARD ══ */
.input-card {
    background: rgba(0,245,212,0.03);
    border: 1px solid rgba(0,245,212,0.15);
    border-radius: 20px;
    padding: 1.8rem 2rem 1.6rem;
    margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.input-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00f5d4, #9d4edd, #ff006e, transparent);
}
.input-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; letter-spacing: 0.18em;
    text-transform: uppercase; color: #00f5d4;
    opacity: 0.6; margin-bottom: 14px;
}

[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(0,245,212,0.2) !important;
    border-radius: 14px !important;
    color: #dde8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.75 !important;
    resize: vertical !important;
    caret-color: #00f5d4 !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(0,245,212,0.45) !important;
    box-shadow: 0 0 0 3px rgba(0,245,212,0.07), 0 0 20px rgba(0,245,212,0.05) !important;
    outline: none !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: #1e2a3a !important; }
[data-testid="stTextArea"] label { display: none !important; }

/* ── Analyse button ── */
div[data-testid="stVerticalBlock"] .stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.07em !important;
    color: #07000f !important;
    background: linear-gradient(90deg, #00f5d4, #00c8ff) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(0,245,212,0.25), 0 0 0 0 rgba(0,245,212,0) !important;
    width: 100% !important;
}
div[data-testid="stVerticalBlock"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 36px rgba(0,245,212,0.4) !important;
}

/* ══ RESULT SECTION ══ */
/* Title card */
.result-title-card {
    position: relative; overflow: hidden;
    padding: 2.2rem 2.4rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 22px;
    margin-bottom: 1.25rem;
}
.result-title-card::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(0,245,212,0.06) 0%, transparent 50%, rgba(255,0,110,0.04) 100%);
    border-radius: inherit;
}
.decoded-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.2em;
    text-transform: uppercase; color: #00f5d4;
    opacity: 0.55; margin-bottom: 12px;
}
.result-movie-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800; color: #e0f7ff;
    letter-spacing: -0.025em; line-height: 1.05;
    margin-bottom: 16px;
    text-shadow: -2px 0 0 rgba(255,0,110,0.3), 2px 0 0 rgba(0,245,212,0.3);
}
.meta-strip { display: flex; flex-wrap: wrap; gap: 8px; }
.meta-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; padding: 4px 11px;
    border-radius: 6px; letter-spacing: 0.06em;
    border: 1px solid rgba(0,245,212,0.2);
    background: rgba(0,245,212,0.07);
    color: #00f5d4;
}
.meta-tag.pink {
    border-color: rgba(255,0,110,0.25);
    background: rgba(255,0,110,0.07);
    color: #ff6eb0;
}
.meta-tag.purple {
    border-color: rgba(157,78,221,0.25);
    background: rgba(157,78,221,0.07);
    color: #c084fc;
}

/* Plot */
.plot-block {
    border-left: 3px solid #00f5d4;
    padding: 1.2rem 1.6rem;
    background: rgba(0,245,212,0.03);
    border-radius: 0 16px 16px 0;
    margin-bottom: 1.25rem;
}
.plot-block .plot-hd {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; letter-spacing: 0.18em;
    text-transform: uppercase; color: #00f5d4;
    opacity: 0.55; margin-bottom: 10px;
}
.plot-block .plot-body {
    font-size: 0.92rem; line-height: 1.8;
    color: #9ab0c8; font-style: italic;
}

/* Data grid */
.data-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
.data-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }

/* Data cell */
.dcell {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.2rem 1.4rem;
    position: relative; overflow: hidden;
}
.dcell::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1.5px;
    background: linear-gradient(90deg, #00f5d4, transparent);
    opacity: 0.4;
}
.dcell.pink-top::after { background: linear-gradient(90deg, #ff006e, transparent); }
.dcell.purple-top::after { background: linear-gradient(90deg, #9d4edd, transparent); }

.dcell-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.16em;
    text-transform: uppercase; color: #3a4a6a;
    margin-bottom: 9px;
}
.dcell-value {
    font-size: 0.88rem; color: #9ab0c8; line-height: 1.65;
}
.dcell-value strong { color: #dde8f0; font-weight: 600; }

/* Chip cloud */
.chip-cloud { display: flex; flex-wrap: wrap; gap: 6px; }
.chip {
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 10px; border-radius: 5px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #4a6080;
}
.chip.c { background: rgba(0,245,212,0.07); border-color: rgba(0,245,212,0.18); color: #00c8a8; }
.chip.p { background: rgba(255,0,110,0.07); border-color: rgba(255,0,110,0.18); color: #ff6eb0; }
.chip.v { background: rgba(157,78,221,0.07); border-color: rgba(157,78,221,0.18); color: #c084fc; }

/* Awards & facts list */
.list-item {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.86rem; color: #7a90a8; line-height: 1.6;
}
.list-item:last-child { border-bottom: none; }
.list-bullet { color: #00f5d4; flex-shrink: 0; font-size: 0.7rem; margin-top: 4px; }
.list-bullet.p { color: #ff006e; }

/* JSON expander */
[data-testid="stExpander"] {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(0,245,212,0.08) !important;
    border-radius: 14px !important;
    margin-top: 1rem;
}
[data-testid="stExpander"] details summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10.5px !important; color: #3a4a6a !important;
    letter-spacing: 0.1em !important;
}
.json-pre {
    background: #030208;
    border-radius: 10px; padding: 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #2a3a52;
    overflow-x: auto; white-space: pre;
    line-height: 1.7; margin-top: 8px;
}

/* Empty state */
.empty-state { text-align: center; padding: 4rem 1rem; }
.empty-icon { font-size: 3rem; opacity: 0.15; margin-bottom: 16px; }
.empty-text { font-size: 0.82rem; color: #1e2a3a; line-height: 1.7; font-family: 'JetBrains Mono', monospace; }

/* Spinner */
[data-testid="stSpinner"] p {
    color: #3a4a6a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important; letter-spacing: 0.1em !important;
}

::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,245,212,0.15); border-radius: 3px; }
</style>

<div class="bg-mesh"></div>
<div class="scan-sweep"></div>
""",
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "raw_json" not in st.session_state:
    st.session_state.raw_json = ""


def v(d, k, fallback="—"):
    val = d.get(k)
    if val is None or val == "" or val == []:
        return fallback
    return val


def chips(items, style=""):
    if not isinstance(items, list):
        items = [items] if items and items != "—" else []
    if not items:
        return "<span style='color:#1e2a3a;font-size:10px'>—</span>"
    return "".join(f'<span class="chip {style}">{i}</span>' for i in items)


def list_rows(items, bullet_class=""):
    if not isinstance(items, list):
        items = [items] if items and items != "—" else []
    if not items:
        return "<span style='color:#1e2a3a;font-size:10px'>—</span>"
    return "".join(
        f'<div class="list-item"><span class="list-bullet {bullet_class}">▸</span>{i}</div>'
        for i in items
    )


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
st.markdown(
    """
<div class="hero">
    <div class="hero-eyebrow">◈ intelligent movie data extraction</div>
    <div class="glitch-title">Movie<span class="accent">Sage</span></div>
    <div class="hero-sub">// paste a paragraph → receive structured intelligence //</div>
</div>
""",
    unsafe_allow_html=True,
)

# Nav bar
st.markdown(
    """
<div class="nav-bar">
    <div class="nav-left"><span class="nav-dot"></span>SYSTEM READY</div>
    <div class="nav-right">MODEL <span>mistral-small-2603</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# ── INPUT CARD ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="input-card"><div class="input-label">⬡ movie description input</div>',
    unsafe_allow_html=True,
)

description = st.text_area(
    label="movie_input",
    placeholder="Paste a raw paragraph about any movie — cast, plot, production, awards, box office… the more detail, the richer the decoded output.",
    height=150,
    label_visibility="collapsed",
)

col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    run = st.button("◈  DECODE MOVIE", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── RUN MODEL ─────────────────────────────────────────────────────────────────
if run:
    if not description.strip():
        st.warning("Paste a movie description first.")
    else:
        with st.spinner("DECODING SIGNAL ···"):
            fp = prompt.invoke({"movie_description": description})
            res = model.invoke(fp)
            raw = re.sub(r"^```(?:json)?\s*", "", res.content.strip())
            raw = re.sub(r"\s*```$", "", raw).strip()
            try:
                st.session_state.result = json.loads(raw)
                st.session_state.raw_json = json.dumps(
                    st.session_state.result, indent=2
                )
            except json.JSONDecodeError:
                st.session_state.result = None
                st.session_state.raw_json = raw
                st.error("Parse error — raw output shown below.")
                st.code(raw)

# ── RESULTS ───────────────────────────────────────────────────────────────────
data = st.session_state.result

if data is None and not st.session_state.raw_json:
    st.markdown(
        """
    <div class="empty-state">
        <div class="empty-icon">⬡</div>
        <div class="empty-text">awaiting signal input<br>paste a movie description above to begin decoding</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

elif data:
    title = v(data, "title", "UNKNOWN TITLE")
    year = v(data, "release_year")
    runtime = v(data, "runtime")
    language = v(data, "language")
    country = v(data, "country")
    rating = v(data, "rating")
    genre = v(data, "genre", [])
    plot = v(data, "plot_summary")
    director = v(data, "director")
    writers = v(data, "writers", [])
    producers = v(data, "producers", [])
    cast = v(data, "cast", [])
    chars = v(data, "main_characters", [])
    themes = v(data, "themes", [])
    keywords = v(data, "keywords", [])
    awards = v(data, "awards", [])
    facts = v(data, "notable_facts", [])
    box_off = v(data, "box_office")

    genre_list = genre if isinstance(genre, list) else ([genre] if genre != "—" else [])

    # ── Title card ──
    genre_pills = "".join(
        f'<span class="meta-tag purple">{g}</span>' for g in genre_list
    )
    year_pill = f'<span class="meta-tag">📅 {year}</span>' if year != "—" else ""
    rt_pill = f'<span class="meta-tag">⏱ {runtime}</span>' if runtime != "—" else ""
    lang_pill = (
        f'<span class="meta-tag pink">🌐 {language}</span>' if language != "—" else ""
    )
    rating_pill = (
        f'<span class="meta-tag pink">⭐ {rating}</span>' if rating != "—" else ""
    )

    st.markdown(
        f"""
    <div class="result-title-card">
        <div class="decoded-label">⬡ decoded · movie intelligence</div>
        <div class="result-movie-title">{title}</div>
        <div class="meta-strip">
            {year_pill}{rt_pill}{lang_pill}{rating_pill}{genre_pills}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Plot ──
    if plot != "—":
        st.markdown(
            f"""
        <div class="plot-block">
            <div class="plot-hd">plot summary</div>
            <div class="plot-body">{plot}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Row 1: Director + Box Office ──
    st.markdown(
        f"""
    <div class="data-grid">
        <div class="dcell">
            <div class="dcell-label">Director</div>
            <div class="dcell-value"><strong>{director}</strong></div>
        </div>
        <div class="dcell pink-top">
            <div class="dcell-label">Box Office</div>
            <div class="dcell-value"><strong>{box_off}</strong></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Row 2: Writers + Producers ──
    st.markdown(
        f"""
    <div class="data-grid">
        <div class="dcell">
            <div class="dcell-label">Writers</div>
            <div class="chip-cloud">{chips(writers, "c")}</div>
        </div>
        <div class="dcell purple-top">
            <div class="dcell-label">Producers</div>
            <div class="chip-cloud">{chips(producers, "v")}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Cast (full width) ──
    st.markdown(
        f"""
    <div class="dcell" style="margin-bottom:1rem">
        <div class="dcell-label">Cast</div>
        <div class="chip-cloud">{chips(cast, "c")}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Main characters ──
    if isinstance(chars, list) and chars:
        st.markdown(
            f"""
        <div class="dcell pink-top" style="margin-bottom:1rem">
            <div class="dcell-label">Main Characters</div>
            <div class="chip-cloud">{chips(chars, "p")}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Themes + Keywords ──
    st.markdown(
        f"""
    <div class="data-grid">
        <div class="dcell purple-top">
            <div class="dcell-label">Themes</div>
            <div class="chip-cloud">{chips(themes, "v")}</div>
        </div>
        <div class="dcell">
            <div class="dcell-label">Keywords</div>
            <div class="chip-cloud">{chips(keywords)}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Awards + Notable facts ──
    awards_list = (
        awards if isinstance(awards, list) else ([awards] if awards != "—" else [])
    )
    facts_list = facts if isinstance(facts, list) else ([facts] if facts != "—" else [])

    st.markdown(
        f"""
    <div class="data-grid">
        <div class="dcell pink-top">
            <div class="dcell-label">Awards</div>
            {list_rows(awards_list, "p")}
        </div>
        <div class="dcell">
            <div class="dcell-label">Notable Facts</div>
            {list_rows(facts_list)}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Raw JSON ──
    with st.expander("◈  RAW JSON OUTPUT"):
        st.markdown(
            f'<div class="json-pre">{st.session_state.raw_json}</div>',
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)
