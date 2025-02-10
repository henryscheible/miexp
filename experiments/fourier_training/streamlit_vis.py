import pandas as pd
import streamlit as st

st.title("Fourier Component Training Results")


# Read the CSV file
@st.cache_data()
def get_metadata_df() -> pd.DataFrame:
    return pd.read_csv("results/bulk_metadata.csv")


def get_events_df() -> pd.DataFrame:
    return pd.read_csv("results/bulk_events.csv", index_col=0)


metadata_df = get_metadata_df()
events_df = get_events_df()

# Display the dataframe
st.write("Training Runs:")
selection = st.dataframe(metadata_df, selection_mode="single-row", on_select="rerun")
if len(selection["selection"]["rows"]) > 0:  # type: ignore
    row = selection["selection"]["rows"][0]  # type: ignore
    st.write(metadata_df.loc[selection["selection"]["rows"], "uuid"])  # type: ignore

    st.dataframe(events_df)
