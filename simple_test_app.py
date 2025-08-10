import streamlit as st
import numpy as np

st.title("Simple Test App")
st.write("This is a basic test to verify Streamlit deployment works.")

# Simple calculation
x = st.slider("Select a number", 0, 100, 50)
y = x * 2
st.write(f"Double of {x} is {y}")

# Simple plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
ax.set_title("Simple Test Plot")
st.pyplot(fig)
