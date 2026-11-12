"""ClaimGuard's custom UI with a bidirectional Python inference bridge."""
from pathlib import Path
import sys

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import ui_pipeline

# Keep a long-running Streamlit process in sync when the upload limit changes.
CONFIG = ui_pipeline.CONFIG
MAX_FILES = int(CONFIG.get("max_files", 20))
ui_pipeline.MAX_FILES = MAX_FILES
AnalysisJob = ui_pipeline.AnalysisJob

st.set_page_config(page_title="ClaimGuard Vision", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:#0F172A;}
#MainMenu, footer, header {visibility:hidden;}
.block-container {padding:0!important;max-width:100%!important;}
iframe {display:block;border:none;width:100%;}
</style>
""", unsafe_allow_html=True)

# Serve dist directly so uploaded files and model results can travel both ways.
claim_ui = components.declare_component("claimguard_vision", path=str(ROOT / "dist"))

if "roi_threshold" not in st.session_state:
    st.session_state.roi_threshold = CONFIG["roi_threshold"]


@st.fragment(run_every=1)
def render_workspace():
    job = st.session_state.get("analysis_job")
    snapshot = job.snapshot() if job else None
    event = claim_ui(job=snapshot, threshold=st.session_state.roi_threshold,
                     max_files=MAX_FILES, key="claim_workspace", default=None)
    if not isinstance(event, dict) or event.get("id") == st.session_state.get("last_ui_event"):
        return
    st.session_state.last_ui_event = event.get("id")
    if event.get("action") == "analyze" and isinstance(event.get("id"), str):
        if job:
            job.cancelled.set()
        job = AnalysisJob(event)
        st.session_state.analysis_job = job
        job.start()
    elif event.get("action") == "cancel":
        if job:
            job.cancelled.set()
        st.session_state.analysis_job = None
    elif event.get("action") == "set_roi_threshold":
        try:
            threshold = float(event.get("threshold"))
        except (TypeError, ValueError):
            return
        if 0 <= threshold <= 1:
            st.session_state.roi_threshold = threshold


render_workspace()
