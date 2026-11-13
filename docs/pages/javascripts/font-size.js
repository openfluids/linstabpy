// Refined Font Size Stepper Controller for LinStabPy Documentation
(function () {
  const SIZES = [
    { label: "80%",  scale: "0.85" },
    { label: "90%",  scale: "0.92" },
    { label: "100%", scale: "1" },
    { label: "110%", scale: "1.08" },
    { label: "120%", scale: "1.16" }
  ];
  const DEFAULT_INDEX = 2; // 100%

  function getSavedIndex() {
    const raw = localStorage.getItem("linstabpy-font-size-idx");
    const idx = parseInt(raw, 10);
    return !isNaN(idx) && idx >= 0 && idx < SIZES.length ? idx : DEFAULT_INDEX;
  }

  function applySize(scaleVal) {
    if (scaleVal === "1") {
      document.documentElement.style.removeProperty("--cds-content-scale");
    } else {
      document.documentElement.style.setProperty("--cds-content-scale", scaleVal);
    }
  }

  function setupStepper() {
    const currentIdx = getSavedIndex();
    if (currentIdx !== DEFAULT_INDEX) {
      applySize(SIZES[currentIdx].scale);
    }

    const container = document.getElementById("font-stepper-container");
    if (!container) return;

    if (!container._initialized) {
      container._initialized = true;

      const decBtn = container.querySelector(".cds-stepper-dec");
      const incBtn = container.querySelector(".cds-stepper-inc");
      const labelBtn = container.querySelector(".cds-stepper-label");

      if (decBtn) {
        decBtn.addEventListener("click", () => {
          let idx = getSavedIndex();
          if (idx > 0) {
            idx--;
            localStorage.setItem("linstabpy-font-size-idx", idx.toString());
            applySize(SIZES[idx].scale);
            updateState(container, idx);
          }
        });
      }

      if (incBtn) {
        incBtn.addEventListener("click", () => {
          let idx = getSavedIndex();
          if (idx < SIZES.length - 1) {
            idx++;
            localStorage.setItem("linstabpy-font-size-idx", idx.toString());
            applySize(SIZES[idx].scale);
            updateState(container, idx);
          }
        });
      }

      if (labelBtn) {
        labelBtn.addEventListener("click", () => {
          const idx = DEFAULT_INDEX;
          localStorage.setItem("linstabpy-font-size-idx", idx.toString());
          applySize(SIZES[idx].scale);
          updateState(container, idx);
        });
      }
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
    window.document$.subscribe(setupStepper);
  } else {
    document.addEventListener("DOMContentLoaded", setupStepper);
  }
})();
