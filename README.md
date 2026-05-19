# AI Medical Image Detection Agent

A Streamlit demo that uploads a medical image and asks Gemini to generate a structured imaging report.

## Run locally

```bash
pip install -r requirements.txt
streamlit run medical.py
```

Create a `.env` file for local use:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

## Create a live demo link with Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Click **New app**.
4. Select this repository: `Vaibhav-Lomte14/AI-medical-image-detection-agent`.
5. Set **Main file path** to `medical.py`.
6. Open **Advanced settings** and add this secret:

```toml
GOOGLE_API_KEY = "your_google_gemini_api_key"
```

7. Click **Deploy**.

Streamlit will give you a public URL like:

```text
https://your-app-name.streamlit.app
```

Do not commit your `.env` file or API key to GitHub.
