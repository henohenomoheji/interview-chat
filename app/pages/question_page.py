import base64
import streamlit as st
from src.utils import embedding_dify_iframe


def set_background(image_path: str) -> None:
    """Set the Streamlit app background image via inline CSS."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/png;base64,{encoded}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.title("Flexche")
    set_background("contents/flexshe_screen.png")
    embedding_dify_iframe()


if __name__ == "__main__":
    main()
