"""
Simple UI for IRise retrieval system
Main concepts: https://docs.streamlit.io/library/get-started/main-concepts
"""
import pandas as pd
import streamlit as st

from irise import INDEX_DIR
from irise.pipeline import SearchPipeline

st.set_page_config(page_title="Retrieval App", layout="wide")


def init_pipeline():
    if "_pipeline" not in st.session_state:
        st.session_state._pipeline = SearchPipeline(index_path=INDEX_DIR / "irise_index_advanced")


def show_results(results):
    table = pd.DataFrame.from_dict(results)
    st.markdown(table.to_markdown())


def main():
    beta = st.slider("lambda", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                     help="0.0 equals TF-IDF, 1.0 equals BM25F.")
    query = st.text_input("Search 🔎", key="query", placeholder="Enter a query")
    submit = st.button("Submit", key="submit")

    if submit:
        results = st.session_state._pipeline(query, beta=beta)
        show_results(results)


if __name__ == "__main__":
    init_pipeline()
    main()
