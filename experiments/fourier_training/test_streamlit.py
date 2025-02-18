from streamlit.testing.v1 import AppTest

at = AppTest.from_file("streamlit_vis.py")
at.run()
assert not at.exception
