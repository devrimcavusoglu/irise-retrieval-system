"""
Simple UI for IRise retrieval system
Main concepts: https://docs.streamlit.io/library/get-started/main-concepts
"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Retrieval App", layout="wide")


@st.cache
def get_results(query: str):
    pass


def show_results(results):
    table = pd.DataFrame({
        "index": [0, 1, 2, 3],
        "docs" : ["doc1", "doc2", "doc3", "doc4"],
    })
    # st.table(table)
    st.dataframe(table)


def main():
    query = st.text_input("Search 🔎", key="query", placeholder="Enter a query")
    submit = st.button("Submit", key="submit")

    if submit:
        r = get_results(query)
        show_results(r)


if __name__ == "__main__":
    main()
