import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.query_engine import build_agent, is_slurm_related

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IGS Grid Assistant",
    page_icon="🖥️",
    layout="centered",
)

st.markdown("""
<style>
/* Wrap long lines inside code blocks */
pre, code {
    white-space: pre-wrap !important;
    word-break: break-word !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🖥️ IGS Grid Assistant")
st.caption("Ask me anything about the IGS computational grid and Slurm.")

# ── Load engine (cached — only built once per session) ────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base...")
def get_engine():
    return build_agent()

engine = get_engine()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "lab" not in st.session_state:
    st.session_state.lab = None

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Your Lab / Group")
    lab_options = [
        "Not selected",
        "IGS IFX",
        "General",
        "Maryland Genomics / GRC",
        "Fertig Lab",
        "Hotopp Lab",
        "EDN",
        "Rasko Lab",
        "Silva Lab",
        "Ravel Lab",
        "Serre Lab",
        "Ernst Group",
        "Diagnostics Radiology & Nuclear Medicine",
        "Dept. of Pain & Translational Symptom Science",
        "Dept. of Pharmacology / Wolff Lab",
    ]
    selected_lab = st.selectbox("Select your lab:", lab_options)
    if selected_lab != "Not selected":
        st.session_state.lab = selected_lab
        st.success(f"Lab set: {selected_lab}")

    st.divider()
    st.markdown("**Need more help?**")
    st.markdown("Contact the grid admins or post in the grid support Slack channel.")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.lab = None
        st.session_state.pending_query = None
        st.rerun()

# ── Suggested queries (shown only when chat is empty) ────────────────────────

SUGGESTED_QUERIES = [
    "How do I access the grid for the first time?",
    "My job failed with exceeded memory limit — what do I do?",
    "How do I request GPU resources for my job?",
    "Why is my job still pending after I submitted it?",
]

if not st.session_state.messages:
    st.markdown("**Common questions to get started:**")
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTED_QUERIES):
        if cols[i % 2].button(suggestion, use_container_width=True):
            st.session_state.pending_query = suggestion
            st.rerun()

# ── Chat history display ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input — always rendered ──────────────────────────────────────────────

typed_input = st.chat_input("Ask a question about the IGS grid or Slurm...")
st.caption("<span style='color: #aaaaaa; font-size: 0.75em;'>Responses are AI-generated and may not always be accurate. Always verify critical information.</span>", unsafe_allow_html=True)

# Pick up pending query from suggestion button, or typed input
prompt = st.session_state.pending_query or typed_input
if st.session_state.pending_query:
    st.session_state.pending_query = None

if prompt:

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Greeting check
    greetings = {"hi", "hello", "hey", "howdy", "greetings", "good morning", "good afternoon"}
    if prompt.strip().lower().rstrip("!,.") in greetings:
        reply = "Hi! I'm the IGS Grid Assistant. Ask me anything about using the IGS computational grid or Slurm — job submission, GPU resources, troubleshooting, and more."
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    # Guardrail check
    elif not is_slurm_related(prompt):
        reply = "I can only help with IGS grid and Slurm-related questions. For other topics, please refer to the appropriate resource."
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    else:
        # Inject lab context into query if known
        augmented_prompt = prompt
        if st.session_state.lab and st.session_state.lab != "Not selected":
            augmented_prompt = f"[User is in: {st.session_state.lab}] {prompt}"

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("_Thinking..._")
            full_reply = ""
            for token in engine.stream_chat(augmented_prompt).response_gen:
                full_reply += token
                response_placeholder.markdown(full_reply + "▌")
            response_placeholder.markdown(full_reply)

        st.session_state.messages.append({"role": "assistant", "content": full_reply})
