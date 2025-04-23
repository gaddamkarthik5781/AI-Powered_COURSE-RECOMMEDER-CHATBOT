import os, re
from dotenv import load_dotenv
import streamlit as st
from Retrieval import get_top_k_courses

# 1. Load secrets
load_dotenv()

# 2. Page config
st.set_page_config(
    page_title="AI Course Recommender",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 3. Custom CSS
st.markdown(
    """
    <style>
      .main > div.appview-container {
        max-width: 850px;
        margin: auto;
      }
      .justified {
        text-align: justify;
      }

      /* Button spacing with flex container */
      .button-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 8px;
        margin-bottom: 16px;
      }

      .btn-link {
        background-color: #1e3a5f;
        color: #f0f0f0;
        padding: 8px 14px;
        border-radius: 4px;
        text-decoration: none;
        border: 1px solid #90cdf4;
        font-size: 0.95em;
      }
      .btn-link:hover {
        background-color: #34597d;
      }

      .info-note {
        color: #f0f0f0; /* Light color for dark theme */
        font-weight: 500;
        margin-top: 10px;
      }

      .footer-msg {
        text-align: center;
        font-size: 1.1em;
        color: #f0f0f0; /* Light color for dark theme */
        margin-top: 40px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# 4. Header
st.markdown("<h1 style='text-align:center;'>🤖 AI Course Recommender Chatbot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#ccc;'>"
    "Enter any course‐related query below, and get back top recommendations drawn from our Pinecone‐backed vector index."
    "</p>",
    unsafe_allow_html=True,
)

# 5. Search bar
query = st.text_input(
    label="What would you like to learn today?",
    placeholder="Type here…, for e.g.: “Best web development courses”",
    key="search_input"
)
search_clicked = st.button("🔍 Search", help="Click to find courses")

# 6. Perform search
if search_clicked and query:
    with st.spinner("Fetching your personalized recommendations…"):
        results = get_top_k_courses(query, top_k=5)

    if not results:
        st.warning("No courses found. Try rephrasing your query.")
    else:
        st.success("Here are the top recommendations for you:")
        for match in results:
            meta = match.metadata
            title = meta.get("title", "Unknown Course")
            desc  = meta.get("description", "")
            if "MAIN FEATURES" in desc.upper():
                parts = re.split(r"(MAIN FEATURES OF THE PROGRAM[:]? )", desc, flags=re.I)
                intro = parts[0].strip()
                features = parts[-1].strip().splitlines()
                st.markdown(f"### {title}")
                if intro:
                    st.markdown(f"<div class='justified'>{intro}</div>", unsafe_allow_html=True)
                st.markdown("**Main Features:**")
                for line in features:
                    clean = line.lstrip("•- ").strip()
                    if clean:
                        st.markdown(f"- {clean}")
            else:
                st.markdown(f"### {title}")
                st.markdown(f"<div class='justified'>{desc}</div>", unsafe_allow_html=True)

            # ℹ️ note about curriculum
            st.markdown(
                "<p class='info-note'><br>ℹ️ Curriculum info is on the same overview page—just toggle it there.</p>",
                unsafe_allow_html=True,
            )

            # Action buttons with proper spacing
            curriculum = meta.get("curriculum_link", "")
            fee        = meta.get("course_fee_link", "")
            enquire    = meta.get("enquire_link", "")
            demo       = meta.get("demo_link", "")

            btns_html = "<div class='button-container'>"
            if curriculum:
                btns_html += f"<a class='btn-link' href='{curriculum}' target='_blank'>📚 Curriculum</a>"
            if fee:
                btns_html += f"<a class='btn-link' href='{fee}' target='_blank'>💰 Course Fee</a>"
            if enquire:
                btns_html += f"<a class='btn-link' href='{enquire}' target='_blank'>✉️ Enquire Now</a>"
            if demo:
                btns_html += f"<a class='btn-link' href='{demo}' target='_blank'>🎓 Free Demo</a>"
            btns_html += "</div>"

            st.markdown(btns_html, unsafe_allow_html=True)
            st.markdown("---")

        # 7. Footer message
        st.markdown(
            "<p class='footer-msg'>🎉 Keep up the curiosity! Your next big skill is just one click away. 🦸‍♂️🦸‍♀️</p>",
            unsafe_allow_html=True,
        )
