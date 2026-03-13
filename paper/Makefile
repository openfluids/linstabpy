PYTHON := uv run --no-sync python
PDFLATEX := pdflatex -interaction=nonstopmode

.PHONY: all figs brief supp clean

all: brief supp

figs:
	$(PYTHON) figs/example_grid.py

brief:
	$(PDFLATEX) brief.tex
	bibtex brief
	$(PDFLATEX) brief.tex
	$(PDFLATEX) brief.tex

supp: figs
	$(PDFLATEX) supp.tex
	bibtex supp
	$(PDFLATEX) supp.tex
	$(PDFLATEX) supp.tex

clean:
	rm -f brief.aux brief.bbl brief.blg brief.fdb_latexmk brief.fls brief.log brief.out
	rm -f supp.aux supp.bbl supp.blg supp.fdb_latexmk supp.fls supp.log supp.out
