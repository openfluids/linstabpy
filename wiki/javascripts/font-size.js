// Refined Font Size Stepper Controller for LinStabPy Documentation
(function () {
  const SIZES = [
    { label: "80%",  value: "80%" },
    { label: "90%",  value: "90%" },
    { label: "100%", value: "100%" },
    { label: "110%", value: "110%" },
    { label: "120%", value: "120%" }
  ];
  const DEFAULT_INDEX = 2; // 100%

  function getSavedIndex() {
    const raw = localStorage.getItem("linstabpy-font-size-idx");
    const idx = parseInt(raw, 10);
    return !isNaN(idx) && idx >= 0 && idx < SIZES.length ? idx : DEFAULT_INDEX;
  }

  function applySize(sizeVal) {
    document.documentElement.style.fontSize = sizeVal;
  }

  function renderStepper() {
    let currentIdx = getSavedIndex();
    applySize(SIZES[currentIdx].value);

    let container = document.getElementById("font-stepper-container");
    if (!container) {
      const parent = document.querySelector(".md-header__inner");
      if (!parent) return;

      container = document.createElement("div");
      container.id = "font-stepper-container";
      container.className = "cds-stepper";

      container.innerHTML = `
        <button type="button" class="cds-stepper-btn cds-stepper-dec" aria-label="Decrease text size" title="Decrease text size">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>
        </button>
        <button type="button" class="cds-stepper-label" aria-label="Reset text size to 100%" title="Reset text size to default (100%)">
          <span class="cds-stepper-text">${SIZES[currentIdx].label}</span>
        </button>
        <button type="button" class="cds-stepper-btn cds-stepper-inc" aria-label="Increase text size" title="Increase text size">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        </button>
      `;

      // Insert adjacent to palette or search
      const palette = document.querySelector('[data-md-component="palette"]');
      if (palette && palette.parentNode) {
        palette.parentNode.insertBefore(container, palette);
      } else {
        parent.appendChild(container);
      }

      const decBtn = container.querySelector(".cds-stepper-dec");
      const incBtn = container.querySelector(".cds-stepper-inc");
      const labelBtn = container.querySelector(".cds-stepper-label");

      decBtn.addEventListener("click", () => {
        let idx = getSavedIndex();
        if (idx > 0) {
          idx--;
          localStorage.setItem("linstabpy-font-size-idx", idx.toString());
          applySize(SIZES[idx].value);
          container.querySelector(".cds-stepper-text").textContent = SIZES[idx].label;
          updateState(container, idx);
        }
      });

      incBtn.addEventListener("click", () => {
        let idx = getSavedIndex();
        if (idx < SIZES.length - 1) {
          idx++;
          localStorage.setItem("linstabpy-font-size-idx", idx.toString());
          applySize(SIZES[idx].value);
          container.querySelector(".cds-stepper-text").textContent = SIZES[idx].label;
          updateState(container, idx);
        }
      });

      labelBtn.addEventListener("click", () => {
        const idx = DEFAULT_INDEX;
        localStorage.setItem("linstabpy-font-size-idx", idx.toString());
        applySize(SIZES[idx].value);
        container.querySelector(".cds-stepper-text").textContent = SIZES[idx].label;
        updateState(container, idx);
      });
    }

    updateState(container, currentIdx);
  }

  function updateState(container, idx) {
    const textSpan = container.querySelector(".cds-stepper-text");
    if (textSpan) textSpan.textContent = SIZES[idx].label;

    const decBtn = container.querySelector(".cds-stepper-dec");
    const incBtn = container.querySelector(".cds-stepper-inc");
    if (decBtn) decBtn.disabled = (idx === 0);
    if (incBtn) incBtn.disabled = (idx === SIZES.length - 1);
  }

  if (window.document$) {
    window.document$.subscribe(renderStepper);
  } else {
    document.addEventListener("DOMContentLoaded", renderStepper);
  }
})();
