import pandas as pd
import streamlit as st

st.title("Fourier Component Training Results")

# Read the CSV file
metadata_df = pd.read_csv("results/bulk_metadata.csv")

# Display the dataframe
st.write("Training Runs:")
st.dataframe(metadata_df)
