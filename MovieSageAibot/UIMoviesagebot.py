from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import json
import re
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
    - title
    - genre
    - director
    - writers
    - producers
    - cast
    - release_year
    - runtime
    - language
    - country
    - plot_summary
    - main_characters
    - themes
    - notable_facts
    - awards
    - box_office
    - rating
    - keywords
    """,
        ),
        (
            "human",
            """
    Analyze the following movie description and extract all relevant information.

    Movie Description:
    {movie_description}
    """,
        ),
    ]
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MovieSage AI", page_icon="🎬", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: transparent !important;
    font-family: 'Inter', sans-serif;
    color: #e2d9c8;
}
[data-testid="stAppViewContainer"] {
    background: #0a0804 !important;
    min-height: 100vh;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── Scanline overlay ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.08) 2px,
        rgba(0,0,0,0.08) 4px
    );
    pointer-events: none;
    z-index: 999;
}

/* ── Grain texture ── */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none; opacity: 0.35; z-index: 998;
}

/* ── Projection glow ── */
.proj-glow {
    position: fixed; top: -200px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 400px;
    background: radial-gradient(ellipse, rgba(255,180,50,0.08) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}

/* ── Film strip header ── */
.filmstrip {
    position: relative; overflow: hidden;
    background: #111008;
    border-bottom: 2px solid #2a2215;
    padding: 0;
    margin-bottom: 0;
}
.filmstrip-holes {
    display: flex; align-items: center;
    gap: 0; background: #0a0804;
    padding: 6px 0;
}
.hole {
    width: 18px; height: 14px;
    background: #0a0804;
    border: 1.5px solid #2a2215;
    border-radius: 3px;
    flex-shrink: 0;
    margin: 0 6px;
}
.strip-spacer { flex: 1; }

.header-content {
    padding: 2rem 3rem 2.5rem;
    position: relative; z-index: 1;
    display: flex; align-items: flex-end;
    justify-content: space-between;
}
.brand-block {}
.brand-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.18em;
    text-transform: uppercase; color: #c97b2a;
    margin-bottom: 6px;
}
.brand-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem; line-height: 0.9;
    color: #f5e6c8; letter-spacing: 0.04em;
}
.brand-title span { color: #e8981e; }
.brand-sub {
    font-size: 0.78rem; color: #5a4e3a; margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
}
.model-badge {
    background: rgba(232,152,30,0.1);
    border: 1px solid rgba(232,152,30,0.25);
    border-radius: 8px; padding: 8px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px; color: #c97b2a;
    letter-spacing: 0.06em; text-align: right;
}
.model-badge .mb-val { color: #e8981e; font-size: 12px; font-weight: 500; display: block; margin-top: 3px; }

/* ── Main layout ── */
.page-body {
    position: relative; z-index: 1;
    max-width: 1280px; margin: 0 auto;
    padding: 2rem 2rem 4rem;
}

/* ── Input panel ── */
.input-panel {
    background: rgba(232,152,30,0.04);
    border: 1px solid rgba(232,152,30,0.12);
    border-radius: 16px; padding: 1.8rem 2rem;
    margin-bottom: 2rem;
}
.input-panel .panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.14em;
    text-transform: uppercase; color: #5a4e3a;
    margin-bottom: 12px;
}

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(232,152,30,0.2) !important;
    border-radius: 12px !important;
    color: #e2d9c8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    resize: vertical !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(232,152,30,0.5) !important;
    box-shadow: 0 0 0 3px rgba(232,152,30,0.08) !important;
    outline: none !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: #3a3020 !important; }
[data-testid="stTextArea"] label { color: #7a6a4a !important; font-size: 0.8rem !important; }

/* ── Button ── */
div[data-testid="stVerticalBlock"] .stButton > button {
    background: linear-gradient(135deg, #e8981e, #c97b2a) !important;
    color: #0a0804 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2.5rem !important;
    transition: all 0.17s ease !important;
    box-shadow: 0 4px 20px rgba(232,152,30,0.25) !important;
}
div[data-testid="stVerticalBlock"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(232,152,30,0.4) !important;
}

/* ── Result: movie header card ── */
.movie-header-card {
    background: linear-gradient(135deg, rgba(232,152,30,0.08), rgba(201,123,42,0.04));
    border: 1px solid rgba(232,152,30,0.2);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.movie-header-card::after {
    content: '🎬';
    position: absolute; right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem; opacity: 0.06;
}
.movie-title-display {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem; color: #f5e6c8;
    letter-spacing: 0.06em; line-height: 1; margin-bottom: 8px;
}
.movie-meta-row {
    display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px;
}
.meta-pill {
    background: rgba(232,152,30,0.1);
    border: 1px solid rgba(232,152,30,0.2);
    border-radius: 6px; padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px; color: #c97b2a; letter-spacing: 0.05em;
}
.rating-big {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem; color: #e8981e;
}

/* ── Section cards ── */
.section-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
}
.section-card::before {
    content: '';
    position: absolute; left: 0; top: 20%; bottom: 20%;
    width: 3px; border-radius: 0 3px 3px 0;
    background: linear-gradient(180deg, #e8981e, #c97b2a);
    opacity: 0.6;
}
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; letter-spacing: 0.15em;
    text-transform: uppercase; color: #5a4e3a;
    margin-bottom: 10px;
}
.sec-value {
    font-size: 0.9rem; color: #c8baa0; line-height: 1.65;
}
.sec-value strong { color: #e2d9c8; }

/* ── Tag chips ── */
.tags-wrap { display: flex; flex-wrap: wrap; gap: 7px; }
.tag-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 6px; padding: 4px 10px;
    font-size: 0.78rem; color: #8a7a5a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
}
.tag-chip.amber {
    background: rgba(232,152,30,0.08);
    border-color: rgba(232,152,30,0.18);
    color: #c97b2a;
}

/* ── Plot summary ── */
.plot-card {
    background: rgba(232,152,30,0.04);
    border: 1px solid rgba(232,152,30,0.12);
    border-radius: 16px; padding: 1.6rem 2rem;
    margin-bottom: 1rem;
}
.plot-card .plot-text {
    font-size: 0.95rem; line-height: 1.85;
    color: #c8baa0; font-style: italic;
}
.plot-card .plot-text::before { content: '\201C'; color: #e8981e; font-size: 1.4rem; margin-right: 4px; }
.plot-card .plot-text::after  { content: '\201D'; color: #e8981e; font-size: 1.4rem; margin-left: 4px; }

/* ── Grid ── */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
.three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }

/* ── JSON toggle ── */
.json-block {
    background: #080604;
    border: 1px solid rgba(232,152,30,0.1);
    border-radius: 12px; padding: 1.2rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; color: #6a5c3a;
    overflow-x: auto; white-space: pre-wrap; line-height: 1.6;
    margin-top: 0.5rem;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: #5a4e3a !important; font-size: 0.8rem !important; font-family: 'JetBrains Mono', monospace !important; }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 4rem 2rem;
    color: #2a2215;
}
.empty-state .es-icon { font-size: 3.5rem; margin-bottom: 16px; opacity: 0.4; }
.empty-state p { font-size: 0.82rem; color: #3a3020; line-height: 1.6; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2215; border-radius: 3px; }

/* ── Streamlit expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: #5a4e3a !important; font-size: 0.8rem !important; }
</style>

<div class="proj-glow"></div>
""",
    unsafe_allow_html=True,
)

# ── Film strip header ──────────────────────────────────────────────────────────
holes_html = "".join(['<div class="hole"></div>' for _ in range(40)])
st.markdown(
    f"""
<div class="filmstrip">
    <div class="filmstrip-holes">{holes_html}</div>
    <div class="header-content">
        <div class="brand-block">
            <div class="brand-eyebrow">▶ Powered by Mistral AI · LangChain</div>
            <div class="brand-title">Movie<span>Sage</span></div>
            <div class="brand-sub">// intelligent movie data extraction //</div>
        </div>
        <div class="model-badge">
            MODEL
            <span class="mb-val">mistral-small-2603</span>
        </div>
    </div>
    <div class="filmstrip-holes">{holes_html}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "raw_json" not in st.session_state:
    st.session_state.raw_json = ""


# ── Helper ────────────────────────────────────────────────────────────────────
def val(data, key, fallback="—"):
    v = data.get(key)
    if v is None or v == "" or v == []:
        return fallback
    if isinstance(v, list):
        return v
    return str(v)


def list_to_chips(items, amber=False):
    if not isinstance(items, list) or not items:
        return "<span style='color:#3a3020;font-size:0.8rem'>Not available</span>"
    cls = "tag-chip amber" if amber else "tag-chip"
    return "".join(f'<span class="{cls}">{i}</span>' for i in items)


# ── Input panel ───────────────────────────────────────────────────────────────
st.markdown('<div class="page-body">', unsafe_allow_html=True)
st.markdown('<div class="input-panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="panel-label">🎬 movie description</div>', unsafe_allow_html=True
)

description = st.text_area(
    label="Movie description",
    placeholder="Paste a movie paragraph here — cast, plot, production details, awards… the more detail, the richer the extraction.",
    height=160,
    label_visibility="collapsed",
)

col_btn, col_space = st.columns([1, 3])
with col_btn:
    analyse = st.button("ANALYSE MOVIE", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Run model (mirrors original logic exactly) ────────────────────────────────
if analyse:
    if not description.strip():
        st.warning("Paste a movie description first.")
    else:
        with st.spinner("Projecting intelligence… reading the reel…"):
            final_prompt = prompt.invoke({"movie_description": description})
            res = model.invoke(final_prompt)
            raw = res.content.strip()
            # strip markdown fences if model wraps in ```json
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            try:
                st.session_state.result = json.loads(raw)
                st.session_state.raw_json = json.dumps(
                    st.session_state.result, indent=2
                )
            except json.JSONDecodeError:
                st.session_state.result = None
                st.session_state.raw_json = raw
                st.error(
                    "MovieSage returned unexpected output. Raw response shown below."
                )
                st.code(raw, language="text")

# ── Result display ─────────────────────────────────────────────────────────────
data = st.session_state.result

if data is None and not st.session_state.raw_json:
    st.markdown(
        """
    <div class="empty-state">
        <div class="es-icon">🎞️</div>
        <p>Paste a movie description above and hit <strong style="color:#5a4e3a">Analyse Movie</strong>.<br>
        MovieSage will extract title, cast, themes, awards, box office, and more — returned as structured data.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

elif data:
    # ── Movie header ──
    title = val(data, "title", "Unknown Title")
    year = val(data, "release_year")
    runtime = val(data, "runtime")
    lang = val(data, "language")
    country = val(data, "country")
    rating = val(data, "rating")
    genre_raw = val(data, "genre", [])
    genre_chips = list_to_chips(
        genre_raw if isinstance(genre_raw, list) else [genre_raw], amber=True
    )

    st.markdown(
        f"""
    <div class="movie-header-card">
        <div class="movie-title-display">{title}</div>
        <div class="tags-wrap" style="margin-top:10px">{genre_chips}</div>
        <div class="movie-meta-row">
            {'<span class="meta-pill">📅 ' + year + "</span>" if year != "—" else ""}
            {'<span class="meta-pill">⏱ ' + runtime + "</span>" if runtime != "—" else ""}
            {'<span class="meta-pill">🌐 ' + lang + "</span>" if lang != "—" else ""}
            {'<span class="meta-pill">📍 ' + country + "</span>" if country != "—" else ""}
            {'<span class="meta-pill rating-big">⭐ ' + rating + "</span>" if rating != "—" else ""}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Plot summary ──
    plot = val(data, "plot_summary")
    if plot != "—":
        st.markdown(
            f"""
        <div class="plot-card">
            <div class="sec-label">Plot Summary</div>
            <div class="plot-text">{plot}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Director / Writers / Producers ──
    director = val(data, "director")
    writers = val(data, "writers", [])
    producers = val(data, "producers", [])
    cast = val(data, "cast", [])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
        <div class="section-card">
            <div class="sec-label">Director</div>
            <div class="sec-value"><strong>{director}</strong></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        writers_chips = list_to_chips(
            writers if isinstance(writers, list) else [writers]
        )
        st.markdown(
            f"""
        <div class="section-card">
            <div class="sec-label">Writers</div>
            <div class="tags-wrap">{writers_chips}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        producers_chips = list_to_chips(
            producers if isinstance(producers, list) else [producers]
        )
        st.markdown(
            f"""
        <div class="section-card">
            <div class="sec-label">Producers</div>
            <div class="tags-wrap">{producers_chips}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        box_office = val(data, "box_office")
        st.markdown(
            f"""
        <div class="section-card">
            <div class="sec-label">Box Office</div>
            <div class="sec-value"><strong>{box_office}</strong></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Cast ──
    cast_chips = list_to_chips(cast if isinstance(cast, list) else [cast], amber=True)
    st.markdown(
        f"""
    <div class="section-card">
        <div class="sec-label">Cast</div>
        <div class="tags-wrap">{cast_chips}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Main characters ──
    chars_raw = val(data, "main_characters", [])
    if isinstance(chars_raw, list) and chars_raw:
        chars_chips = list_to_chips(chars_raw)
        st.markdown(
            f"""
        <div class="section-card">
            <div class="sec-label">Main Characters</div>
            <div class="tags-wrap">{chars_chips}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Themes & Keywords ──
    themes = val(data, "themes", [])
    keywords = val(data, "keywords", [])

    col3, col4 = st.columns(2)
    with col3:
        themes_chips = list_to_chips(
            themes if isinstance(themes, list) else [themes], amber=True
        )
        st.markdown(
            f"""
        <div class="section-card">
            <div class="sec-label">Themes</div>
            <div class="tags-wrap">{themes_chips}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        kw_chips = list_to_chips(keywords if isinstance(keywords, list) else [keywords])
        st.markdown(
            f"""
        <div class="section-card">
            <div class="sec-label">Keywords</div>
            <div class="tags-wrap">{kw_chips}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Awards ──
    awards_raw = val(data, "awards", [])
    if isinstance(awards_raw, list) and awards_raw:
        awards_html = "".join(
            f'<div class="sec-value" style="margin-bottom:4px">🏆 {a}</div>'
            for a in awards_raw
        )
    elif awards_raw != "—":
        awards_html = f'<div class="sec-value">🏆 {awards_raw}</div>'
    else:
        awards_html = (
            '<span style="color:#3a3020;font-size:0.8rem">Not available</span>'
        )

    st.markdown(
        f"""
    <div class="section-card">
        <div class="sec-label">Awards</div>
        {awards_html}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Notable facts ──
    facts_raw = val(data, "notable_facts", [])
    if isinstance(facts_raw, list) and facts_raw:
        facts_html = "".join(
            f'<div class="sec-value" style="margin-bottom:5px">· {f}</div>'
            for f in facts_raw
        )
    elif facts_raw != "—":
        facts_html = f'<div class="sec-value">· {facts_raw}</div>'
    else:
        facts_html = '<span style="color:#3a3020;font-size:0.8rem">Not available</span>'

    st.markdown(
        f"""
    <div class="section-card">
        <div class="sec-label">Notable Facts</div>
        {facts_html}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Raw JSON expander ──
    with st.expander("View raw JSON output"):
        st.markdown(
            f'<div class="json-block">{st.session_state.raw_json}</div>',
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)  # page-body
