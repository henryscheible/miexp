from streamlit.testing.v1 import AppTest

at = AppTest.from_file("streamlit_vis.py")
at.run(timeout=10)
assert not at.exception
