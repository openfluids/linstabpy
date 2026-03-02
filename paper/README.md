# Metascience in Aerospace manuscript package (`paper`)

This directory is the active paper workspace for submission to:
- `Metascience in Aerospace` (AIMS Press)

## Files

- `manuscript.tex`: single working manuscript file (equations, methods, results story).
- `refs.bib`: bibliography.
- `aims.cls`, `logo.pdf`, `logobottom.pdf`: official class/assets copied from AIMS MINA template.
- `template/`: downloaded official MINA template archive and original files.

## Build

Draft PDF:

```bash
cd paper
pdflatex -interaction=nonstopmode manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```

## Confirmed journal guidance used

- MINA instructions for authors (official content endpoint):
  - `https://www.aimspress.com/data/news/news-detail-data?newsId=2460`
- MINA journal page (APC/APF listing):
  - `https://www.aimspress.com/journal/mina`
- Official MINA TeX template zip:
  - `https://www.aimspress.com/aimspress-upload/news/mina/202411503338785.zip`

## Submission-relevant constraints (current)

- LaTeX template available and accepted.
- Suggested LaTeX line numbering for review (`lineno`, pagewise).
- Abstract under 300 words (target 200-300).
- 5-10 keywords.
- No fixed manuscript length limit.
- MINA page currently states APC/APF `0 USD`.
