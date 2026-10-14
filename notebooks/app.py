
import re
from pathlib import Path

import joblib
import streamlit as st


# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
)


# -------------------- FIND SAVED MODELS --------------------

APP_DIR = Path(__file__).resolve().parent

PROJECT_DIRS = [
    APP_DIR,
    APP_DIR.parent,
    *APP_DIR.parents,
]

MODEL_DIRS = []

for project_dir in PROJECT_DIRS:
    MODEL_DIRS.extend([
        project_dir / "notebooks" / "models",
        project_dir / "models",
    ])

MODEL_DIRS = list(dict.fromkeys(MODEL_DIRS))


def find_model_files():
    for folder in MODEL_DIRS:
        category_path = folder / "ticket_type_model.joblib"
        priority_path = folder / "ticket_priority_model.joblib"

        if category_path.is_file() and priority_path.is_file():
            return category_path, priority_path

    checked = "\n".join(str(folder) for folder in MODEL_DIRS)

    raise FileNotFoundError(
        "Unable to find both saved model files.\n\n"
        "Expected files:\n"
        "ticket_type_model.joblib\n"
        "ticket_priority_model.joblib\n\n"
        "Folders checked:\n" + checked
    )


CATEGORY_MODEL_PATH, PRIORITY_MODEL_PATH = find_model_files()


# -------------------- LOAD MODELS --------------------

@st.cache_resource
def load_models():
    category_model = joblib.load(CATEGORY_MODEL_PATH)
    priority_model = joblib.load(PRIORITY_MODEL_PATH)

    return category_model, priority_model


# -------------------- TEXT CLEANING --------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\{.*?\}", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------- CUSTOM STYLE --------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #2563eb);
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }

    .result-card {
        padding: 22px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        margin-bottom: 12px;
    }

    .result-label {
        color: #64748b;
        font-size: 14px;
    }

    .result-value {
        color: #172554;
        font-size: 23px;
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.title("🎫 Ticket AI")
    st.caption("Support Ticket Classification")

    st.divider()

    st.write("**Features**")
    st.write("• Ticket type prediction")
    st.write("• Priority prediction")
    st.write("• Text-based classification")

    st.divider()

    st.caption("Machine Learning Portfolio Project")


# -------------------- HEADER --------------------

st.markdown(
    """
    <div class="hero">
        <h1>Support Ticket Classifier</h1>
        <p>
            Enter a customer's support request to predict its
            ticket category and priority.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- LOAD STATUS --------------------

try:
    category_model, priority_model = load_models()

    st.success("Both saved models loaded successfully.")

except Exception as error:
    st.error("Unable to load the saved models.")
    st.code(str(error))
    st.stop()


# -------------------- INPUT FORM --------------------

st.subheader("Submit a Support Ticket")

with st.form("ticket_form"):
    subject = st.text_input(
        "Ticket Subject",
        placeholder="Example: Unable to access my account",
    )

    description = st.text_area(
        "Ticket Description",
        placeholder=(
            "Describe the customer's problem in detail..."
        ),
        height=160,
    )

    submitted = st.form_submit_button(
        "Classify Ticket",
        use_container_width=True,
    )


# -------------------- PREDICTION --------------------

if submitted:
    if not subject.strip() or not description.strip():
        st.warning(
            "Please enter both the ticket subject and description."
        )

    else:
        combined_text = subject.strip() + " " + description.strip()
        processed_text = clean_text(combined_text)

        try:
            predicted_category = category_model.predict(
                [processed_text]
            )[0]

            predicted_priority = priority_model.predict(
                [processed_text]
            )[0]

            st.divider()
            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Predicted Ticket Type
                        </div>
                        <div class="result-value">
                            {predicted_category}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Predicted Ticket Priority
                        </div>
                        <div class="result-value">
                            {predicted_priority}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.info(
                "These predictions are experimental. "
                "The current models have low test accuracy, "
                "so verify the results manually before using them "
                "to make real support decisions."
            )

        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "Support Ticket Classification | "
    "Python • Scikit-learn • Streamlit"
)
