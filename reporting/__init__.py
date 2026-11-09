"""Static reporting: the result-file parser and the matplotlib figures.

Separate from `visualise/`, which is the Dash dashboard. The dashboard is uploaded to
the Hugging Face Space and `deploy/requirements.txt` carries no matplotlib, so the two
cannot share a package. Nothing under `visualise/` may import this one.
"""
