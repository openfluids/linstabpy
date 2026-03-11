PYTHON := uv run --no-sync python

.PHONY: figs paper clean

figs:
	$(PYTHON) figs/example_grid.py
	$(PYTHON) figs/scaling_reference.py
	$(PYTHON) figs/case1_dashboard.py

paper: figs
	pdflatex -interaction=nonstopmode manuscript.tex
	bibtex manuscript
	pdflatex -interaction=nonstopmode manuscript.tex
	pdflatex -interaction=nonstopmode manuscript.tex

clean:
	rm -f manuscript.aux manuscript.bbl manuscript.blg manuscript.fdb_latexmk manuscript.fls manuscript.log manuscript.out
