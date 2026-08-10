# Documentation is built with MkDocs Material (see mkdocs.yml).
# The previous Jupyter Book target was dead: docs/_config.yml and docs/_toc.yml
# no longer exist, so `make html` had been failing.

.PHONY: all html serve clean

all: html

# Build the static site into site/
html:
	mkdocs build

# Live preview on http://127.0.0.1:8000 with auto-reload
serve:
	mkdocs serve

clean:
	rm -rf site
