import os
import time

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from PIL import Image as PILImage


# Load .env for local development. Streamlit Cloud uses st.secrets.
load_dotenv()

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Add it to Streamlit secrets or your local .env file.")
    st.stop()

GEMINI_MODEL_IDS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
]

genai.configure(api_key=GOOGLE_API_KEY)

query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging.

Analyze the medical image and structure your response as follows:

### 1. Image Type & Region
- Identify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Specify anatomical region and positioning
- Evaluate image quality and technical adequacy

### 2. Key Findings
- Highlight primary observations systematically
- Identify abnormalities with detailed descriptions
- Include measurements where relevant

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses
- Support findings with evidence
- Highlight urgent findings

### 4. Patient-Friendly Explanation
- Explain findings simply
- Avoid difficult medical jargon

### 5. Research Context
- Include recent medical insights and treatment references

Use proper markdown formatting.
"""


def analyze_medical_image(image_path):
    image = PILImage.open(image_path)

    width, height = image.size
    aspect_ratio = width / height

    new_width = 500
    new_height = int(new_width / aspect_ratio)

    resized_image = image.resize((new_width, new_height))
    if resized_image.mode != "RGB":
        resized_image = resized_image.convert("RGB")

    temp_path = "temp_resized_image.png"
    resized_image.save(temp_path)

    try:
        last_error = ""

        for attempt, model_id in enumerate(GEMINI_MODEL_IDS, start=1):
            try:
                model = genai.GenerativeModel(model_id)
                response = model.generate_content([query, resized_image])

                if response and getattr(response, "text", None):
                    return response.text

                last_error = f"{model_id} returned an empty response."

            except Exception as model_error:
                error_message = str(model_error)
                last_error = error_message
                is_busy = (
                    "UNAVAILABLE" in error_message
                    or "503" in error_message
                    or "high demand" in error_message.lower()
                )

                if is_busy and attempt < len(GEMINI_MODEL_IDS):
                    time.sleep(3)
                    continue

                raise

        return f"Analysis Error: {last_error}"

    except Exception as e:
        error_message = str(e)
        if "RESOURCE_EXHAUSTED" in error_message or "quota" in error_message.lower() or "429" in error_message:
            return (
                "### Gemini quota exceeded\n\n"
                "Your API key is valid, but the Google project connected to this key has no available Gemini quota right now.\n\n"
                "Fix it by doing one of these:\n"
                "- Wait a few minutes and try again.\n"
                "- Check usage at https://ai.dev/rate-limit.\n"
                "- Enable billing for the Google Cloud project connected to this API key.\n"
                "- Create a new API key in a different Google AI Studio project that has available quota.\n"
            )
        if "UNAVAILABLE" in error_message or "503" in error_message or "high demand" in error_message.lower():
            return (
                "### Gemini is temporarily unavailable\n\n"
                "Google says this model is experiencing high demand. This is usually temporary.\n\n"
                "Wait 2-5 minutes, then click Analyze Image again. If it keeps happening, try a new API key/project or enable billing.\n"
            )
        return f"Analysis Error: {error_message}"

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


st.set_page_config(
    page_title="Medical Image Analysis",
    layout="centered",
)

st.title("Medical Image Analysis Tool")

st.markdown(
    """
Upload a medical image like:
- X-ray
- MRI
- CT Scan
- Ultrasound

The AI system will analyze the image and generate a detailed report.
"""
)

st.sidebar.header("Upload Medical Image")

uploaded_file = st.sidebar.file_uploader(
    "Choose Image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True,
    )

    if st.sidebar.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            file_extension = uploaded_file.name.split(".")[-1]
            image_path = f"temp_image.{file_extension}"

            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            report = analyze_medical_image(image_path)

            st.subheader("Analysis Report")
            st.markdown(report)

            if os.path.exists(image_path):
                os.remove(image_path)

else:
    st.warning("Please upload a medical image.")
