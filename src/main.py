import streamlit as st

st.set_page_config(
    page_title="My Streamlit App",
    page_icon="👋",
)

def main():
    st.title("Welcome to My Streamlit App")
    st.write("This is the main page of the application.")

if __name__ == "__main__":
    main()