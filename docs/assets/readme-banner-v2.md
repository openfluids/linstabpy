# README Banner v2

Asset: `docs/assets/readme-banner-v2.jpg` (1408x469, 3:1, 138 KB)

Tool/model: xAI Grok CLI, built-in `image_gen` tool, plus local compositing.

Replaces `linstabpy-banner.png`. Adopts the shared language now used across the
openfluids repositories — `dynachaos`, `fftkit`, `chaos-atlas`, `openmodalpy`,
`dsgbr` — all 3:1, all on a charcoal ground with a warm off-white lowercase
wordmark and cyan/teal structure with coral accents.

Referenced from the README by **relative path**, not by a
`raw.githubusercontent.com` URL. This repository is private, so a raw URL would
fail to authenticate for anyone viewing the README on GitHub. The sibling public
repositories use absolute URLs instead, so their banners also render on PyPI.

## Approach

The wordmark is **not** generated. Image models render short lowercase words
unpredictably, and accepting whatever letterforms come back is most of what
makes a generated banner look cheap. The artwork is generated deliberately
textless and the type is set locally in Lato Light, sized to a fixed fraction of
the frame width so names of different lengths carry comparable optical weight.

## Subject

An eigenvalue spectrum in the complex plane: a dense continuous branch of stable
eigenvalues sweeping away in teal below a horizontal neutral axis, and a handful
of isolated eigenvalues sitting clearly above that axis, glowing coral — the few
unstable modes among many stable ones.

This is the canonical figure of global linear stability analysis, and it is what
this package computes. Growth rate above zero is the whole question; putting the
neutral axis in the frame makes the image say something rather than decorate.

## Prompt (artwork only, no text)

```text
A stunning abstract scientific artwork, wide 2:1 landscape: an eigenvalue
spectrum plotted in the complex plane. Hundreds of fine luminous points arranged
into elegant sweeping branches and arcs — a dense continuous branch curving away
below, and a scattering of discrete isolated points above it. A single long
horizontal line runs across the frame as a neutral axis, and a small number of
brilliant isolated points sit clearly ABOVE that line, glowing hot coral and
amber, distinctly separated from the teal cloud below, as though they were the
few unstable modes among many stable ones. Deep near-black charcoal ground with
a very faint fine grid, volumetric glow, atmospheric depth of field, fine film
grain, rich deep blacks and luminous highlights. Cinematic, precise, elegant,
expensive, gallery-quality scientific data art. ABSOLUTELY NO TEXT [...] Leave
the left third dark, calm and completely empty as negative space.
```

## Post-processing

- Returned 1408x704, centre-cropped to 1408x469. `image_gen` rejects a 3:1
  request and falls back to 2:1, so prompts must state that the image will be
  letterboxed and keep all subject matter in the central band.
- Wordmark composited locally: Lato Light, auto-sized to 28% of frame width
  (91 px here), tracking 6% of point size, warm off-white `#F7F3EC`, with a wide
  blurred dark halo underneath. Placed high in the frame so it clears the
  neutral axis rather than being struck through by it.
- JPEG q95, no chroma subsampling: 138 KB, against 717 KB for the previous PNG.

## Rejected alternatives

- **Amplifying global mode** — alternating cyan and coral lobes growing
  downstream, which is what a resolvent response mode looks like when plotted.
  Physically the most apt of the three, but the saturated primary cyan and
  salmon read as candy next to the restraint of the other banners.
- **Sparse operator structure** — the banded block sparsity pattern of the
  discretized Jacobian, which would have tied to the memory argument that
  motivated the package. It came back repetitive and wallpaper-like, with a hard
  vertical seam where the grid began.
