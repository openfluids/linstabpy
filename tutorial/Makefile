TEX=pdflatex
TEXFLAGS=-interaction=nonstopmode -halt-on-error
TARGET=tutorial

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex
	$(TEX) $(TEXFLAGS) $(TARGET).tex
	$(TEX) $(TEXFLAGS) $(TARGET).tex

clean:
	rm -f $(TARGET).aux $(TARGET).log $(TARGET).out $(TARGET).listing $(TARGET).pdf

.PHONY: all clean
