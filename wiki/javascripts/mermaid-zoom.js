/**
 * Mermaid Diagram Pan, Zoom & Fullscreen Lightbox Controller
 * LinStabPy Documentation - Anthropic/Stripe-grade interactive schematics
 */

// 1. Early shadow root hook so we can inspect and control Mermaid diagrams
(function () {
  if (!Element.prototype._attachShadowOrig) {
    const orig = Element.prototype.attachShadow;
    Element.prototype._attachShadowOrig = orig;
    Element.prototype.attachShadow = function (init) {
      const root = orig.call(this, { ...init, mode: 'open' });
      this.__shadowRoot = root;
      return root;
    };
  }
})();

(function () {
  'use strict';

  // Sizing and zoom configurations
  const CARD_ZOOM_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5];
  let modalInstance = null;

  /* ==========================================================================
     Modal Lightbox Singleton
     ========================================================================== */
  function getOrCreateModal() {
    if (modalInstance) return modalInstance;

    const modal = document.createElement('div');
    modal.id = 'mermaid-lightbox-modal';
    modal.className = 'mermaid-modal';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-label', 'Mermaid Schematic Inspection Lightbox');

    modal.innerHTML = `
      <div class="mermaid-modal-header">
        <div class="mermaid-modal-title-wrap">
          <span class="mermaid-modal-title">Schematic Inspection</span>
          <span class="mermaid-modal-tag">Interactive Vector</span>
        </div>
        <div class="mermaid-modal-controls">
          <div class="mermaid-modal-btn-group">
            <button type="button" class="mermaid-modal-btn btn-zoom-out" title="Zoom Out (– or Wheel Down)" aria-label="Zoom Out">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>
            </button>
            <button type="button" class="mermaid-modal-btn btn-zoom-val" title="Toggle 100% / Fit" aria-label="Reset zoom">100%</button>
            <button type="button" class="mermaid-modal-btn btn-zoom-in" title="Zoom In (+ or Wheel Up)" aria-label="Zoom In">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            </button>
          </div>
          <button type="button" class="mermaid-modal-btn btn-fit" title="Fit diagram to viewport (0)">Fit</button>
          <button type="button" class="mermaid-modal-btn btn-100" title="Natural 100% size">1:1</button>
          <span class="mermaid-modal-hint">Drag to pan • Wheel to zoom • Esc to exit</span>
          <button type="button" class="mermaid-modal-btn btn-close" title="Close Lightbox (Esc)" aria-label="Close modal">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
      </div>
      <div class="mermaid-modal-viewport">
        <div class="mermaid-modal-stage"></div>
      </div>
    `;

    document.body.appendChild(modal);

    const viewport = modal.querySelector('.mermaid-modal-viewport');
    const stage = modal.querySelector('.mermaid-modal-stage');
    const titleEl = modal.querySelector('.mermaid-modal-title');
    const zoomValBtn = modal.querySelector('.btn-zoom-val');
    const zoomInBtn = modal.querySelector('.btn-zoom-in');
    const zoomOutBtn = modal.querySelector('.btn-zoom-out');
    const fitBtn = modal.querySelector('.btn-fit');
    const oneToOneBtn = modal.querySelector('.btn-100');
    const closeBtn = modal.querySelector('.btn-close');

    // Modal state
    let naturalW = 1200;
    let naturalH = 600;
    let zoom = 1.0;
    let fitZoom = 1.0;
    let panX = 0;
    let panY = 0;
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let lastTouchDist = 0;

    function updateTransform(smooth = false) {
      stage.style.transition = smooth ? 'transform 0.18s cubic-bezier(0.16, 1, 0.3, 1)' : 'none';
      stage.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
      zoomValBtn.textContent = `${Math.round(zoom * 100)}%`;
    }

    function calculateFitZoom() {
      const vRect = viewport.getBoundingClientRect();
      const padding = 60;
      const availW = Math.max(100, vRect.width - padding * 2);
      const availH = Math.max(100, vRect.height - padding * 2);
      return Math.min(availW / naturalW, availH / naturalH, 1.2);
    }

    function centerStage(targetZoom, smooth = false) {
      zoom = targetZoom;
      const vRect = viewport.getBoundingClientRect();
      panX = (vRect.width - naturalW * zoom) / 2;
      panY = (vRect.height - naturalH * zoom) / 2;
      updateTransform(smooth);
    }

    function zoomAtPoint(factor, mouseX, mouseY) {
      const vRect = viewport.getBoundingClientRect();
      const originX = mouseX !== undefined ? mouseX - vRect.left : vRect.width / 2;
      const originY = mouseY !== undefined ? mouseY - vRect.top : vRect.height / 2;

      const newZoom = Math.max(0.1, Math.min(8.0, zoom * factor));
      panX = originX - (originX - panX) * (newZoom / zoom);
      panY = originY - (originY - panY) * (newZoom / zoom);
      zoom = newZoom;
      updateTransform(false);
    }

    // Wheel zoom
    viewport.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 0.87;
      zoomAtPoint(factor, e.clientX, e.clientY);
    }, { passive: false });

    // Drag to pan
    viewport.addEventListener('mousedown', (e) => {
      if (e.target.closest('button')) return;
      isDragging = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
      viewport.classList.add('is-dragging');
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      panX = e.clientX - startX;
      panY = e.clientY - startY;
      updateTransform(false);
    });

    window.addEventListener('mouseup', () => {
      if (isDragging) {
        isDragging = false;
        viewport.classList.remove('is-dragging');
      }
    });

    // Touch support (pinch to zoom and pan)
    viewport.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        isDragging = true;
        startX = e.touches[0].clientX - panX;
        startY = e.touches[0].clientY - panY;
      } else if (e.touches.length === 2) {
        isDragging = false;
        lastTouchDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
      }
    }, { passive: true });

    viewport.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1 && isDragging) {
        panX = e.touches[0].clientX - startX;
        panY = e.touches[0].clientY - startY;
        updateTransform(false);
      } else if (e.touches.length === 2) {
        const dist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        if (lastTouchDist > 0) {
          const factor = dist / lastTouchDist;
          const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
          const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
          zoomAtPoint(factor, midX, midY);
        }
        lastTouchDist = dist;
      }
    }, { passive: true });

    viewport.addEventListener('touchend', () => {
      isDragging = false;
      lastTouchDist = 0;
    });

    // Double click to toggle Fit / 100%
    viewport.addEventListener('dblclick', (e) => {
      if (e.target.closest('button')) return;
      if (Math.abs(zoom - fitZoom) < 0.05) {
        centerStage(1.0, true);
      } else {
        fitZoom = calculateFitZoom();
        centerStage(fitZoom, true);
      }
    });

    // Controls
    zoomInBtn.addEventListener('click', () => zoomAtPoint(1.25));
    zoomOutBtn.addEventListener('click', () => zoomAtPoint(0.8));
    zoomValBtn.addEventListener('click', () => {
      if (Math.abs(zoom - 1.0) < 0.05) {
        fitZoom = calculateFitZoom();
        centerStage(fitZoom, true);
      } else {
        centerStage(1.0, true);
      }
    });
    fitBtn.addEventListener('click', () => {
      fitZoom = calculateFitZoom();
      centerStage(fitZoom, true);
    });
    oneToOneBtn.addEventListener('click', () => centerStage(1.0, true));

    function closeModal() {
      modal.classList.remove('is-open');
      document.documentElement.classList.remove('mermaid-modal-active');
      stage.innerHTML = '';
      window.removeEventListener('keydown', handleKey);
    }

    function handleKey(e) {
      if (e.key === 'Escape') closeModal();
      else if (e.key === '+' || e.key === '=') zoomAtPoint(1.2);
      else if (e.key === '-' || e.key === '_') zoomAtPoint(0.83);
      else if (e.key === '0') {
        fitZoom = calculateFitZoom();
        centerStage(fitZoom, true);
      } else if (e.key === 'ArrowLeft') {
        panX += 60; updateTransform(true);
      } else if (e.key === 'ArrowRight') {
        panX -= 60; updateTransform(true);
      } else if (e.key === 'ArrowUp') {
        panY += 60; updateTransform(true);
      } else if (e.key === 'ArrowDown') {
        panY -= 60; updateTransform(true);
      }
    }

    closeBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
      if (e.target === viewport) closeModal();
    });

    modalInstance = {
      open: (svgElement, title) => {
        titleEl.textContent = title || 'LinStabPy Schematic';
        stage.innerHTML = '';

        const clone = svgElement.cloneNode(true);
        clone.style.maxWidth = 'none';
        clone.style.width = '100%';
        clone.style.height = '100%';
        stage.appendChild(clone);

        const vb = svgElement.viewBox && svgElement.viewBox.baseVal;
        naturalW = vb && vb.width ? vb.width : (svgElement.scrollWidth || 1200);
        naturalH = vb && vb.height ? vb.height : (svgElement.scrollHeight || 600);

        stage.style.width = `${naturalW}px`;
        stage.style.height = `${naturalH}px`;

        modal.classList.add('is-open');
        document.documentElement.classList.add('mermaid-modal-active');
        window.addEventListener('keydown', handleKey);

        requestAnimationFrame(() => {
          fitZoom = calculateFitZoom();
          centerStage(fitZoom, false);
        });
      },
      close: closeModal
    };

    return modalInstance;
  }

  /* ==========================================================================
     In-Card Diagram Enhancement
     ========================================================================== */
  function findPrecedingHeading(el) {
    let current = el.parentElement;
    while (current && !current.matches('.md-content__inner, article, main')) {
      let prev = el.previousElementSibling;
      while (prev) {
        if (/^H[1-4]$/i.test(prev.tagName)) {
          return prev.textContent.replace(/¶|\s*#\s*$/g, '').trim();
        }
        prev = prev.previousElementSibling;
      }
      el = current;
      current = current.parentElement;
    }
    return document.title.split(' - ')[0] || 'Architecture Schematic';
  }

  function setupDiagramCard(mermaidEl) {
    if (mermaidEl.closest('.mermaid-card')) return;

    const root = mermaidEl.shadowRoot || mermaidEl.__shadowRoot;
    if (!root) return;

    const svg = root.querySelector('svg');
    if (!svg) {
      // SVG not rendered yet; observe shadow root
      const obs = new MutationObserver(() => {
        if (root.querySelector('svg')) {
          obs.disconnect();
          setupDiagramCard(mermaidEl);
        }
      });
      obs.observe(root, { childList: true, subtree: true });
      return;
    }

    const vb = svg.viewBox && svg.viewBox.baseVal;
    const naturalW = vb && vb.width ? vb.width : (svg.scrollWidth || 1200);
    const naturalH = vb && vb.height ? vb.height : (svg.scrollHeight || 600);

    const title = findPrecedingHeading(mermaidEl);

    // Create Card Wrapper
    const card = document.createElement('div');
    card.className = 'mermaid-card';

    // Toolbar
    const toolbar = document.createElement('div');
    toolbar.className = 'mermaid-toolbar';
    toolbar.setAttribute('role', 'toolbar');
    toolbar.setAttribute('aria-label', 'Schematic zoom and display controls');

    toolbar.innerHTML = `
      <div class="mermaid-card-btn-group">
        <button type="button" class="mermaid-tb-btn btn-card-zoom-out" title="Zoom Out" aria-label="Zoom Out">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>
        </button>
        <button type="button" class="mermaid-tb-btn btn-card-zoom-val" title="Click to toggle Fit / 100%" aria-label="Toggle Fit / 100%">Fit</button>
        <button type="button" class="mermaid-tb-btn btn-card-zoom-in" title="Zoom In" aria-label="Zoom In">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        </button>
      </div>
      <button type="button" class="mermaid-tb-btn btn-card-fullscreen" title="Open Fullscreen Lightbox (Double-click diagram)" aria-label="Fullscreen Lightbox">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="15 3 21 3 21 9"></polyline>
          <polyline points="9 21 3 21 3 15"></polyline>
          <line x1="21" y1="3" x2="14" y2="10"></line>
          <line x1="3" y1="21" x2="10" y2="14"></line>
        </svg>
      </button>
    `;

    // Scroll Container
    const scrollWrap = document.createElement('div');
    scrollWrap.className = 'mermaid-scroll-wrap';

    // Insert wrapper into DOM
    mermaidEl.parentNode.insertBefore(card, mermaidEl);
    card.appendChild(toolbar);
    card.appendChild(scrollWrap);
    scrollWrap.appendChild(mermaidEl);

    // Zoom state
    let mode = 'fit'; // 'fit' or number scale
    let currentScale = 1.0;

    const zoomValBtn = toolbar.querySelector('.btn-card-zoom-val');
    const zoomInBtn = toolbar.querySelector('.btn-card-zoom-in');
    const zoomOutBtn = toolbar.querySelector('.btn-card-zoom-out');
    const fsBtn = toolbar.querySelector('.btn-card-fullscreen');

    function applyCardZoom(newMode, scaleVal) {
      mode = newMode;
      if (mode === 'fit') {
        mermaidEl.style.width = '100%';
        svg.style.width = '100%';
        svg.style.maxWidth = `${naturalW}px`;
        svg.style.height = 'auto';
        zoomValBtn.textContent = 'Fit';
        card.classList.remove('is-zoomed');
        scrollWrap.scrollLeft = 0;
      } else {
        currentScale = scaleVal;
        const targetW = Math.round(naturalW * currentScale);
        mermaidEl.style.width = 'max-content';
        svg.style.width = `${targetW}px`;
        svg.style.maxWidth = 'none';
        svg.style.height = 'auto';
        zoomValBtn.textContent = `${Math.round(currentScale * 100)}%`;
        card.classList.add('is-zoomed');
      }
    }

    // Default: Fit to card
    applyCardZoom('fit', 1.0);

    zoomInBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (mode === 'fit') {
        applyCardZoom('custom', 1.0);
      } else {
        const next = CARD_ZOOM_STEPS.find(s => s > currentScale + 0.05) || (currentScale + 0.25);
        applyCardZoom('custom', Math.min(3.0, next));
      }
    });

    zoomOutBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (mode === 'fit') {
        applyCardZoom('custom', 0.75);
      } else {
        const prev = [...CARD_ZOOM_STEPS].reverse().find(s => s < currentScale - 0.05);
        if (!prev || prev < 0.5) {
          applyCardZoom('fit', 1.0);
        } else {
          applyCardZoom('custom', prev);
        }
      }
    });

    zoomValBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (mode === 'fit') {
        applyCardZoom('custom', 1.0);
      } else {
        applyCardZoom('fit', 1.0);
      }
    });

    fsBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const modal = getOrCreateModal();
      modal.open(svg, title);
    });

    // Double click to open fullscreen lightbox
    scrollWrap.addEventListener('dblclick', (e) => {
      if (e.target.closest('.mermaid-toolbar')) return;
      const modal = getOrCreateModal();
      modal.open(svg, title);
    });

    // Drag-to-pan in scroll container
    let isPanning = false;
    let panStartX = 0;
    let panStartY = 0;
    let initialScrollLeft = 0;
    let initialScrollTop = 0;

    scrollWrap.addEventListener('mousedown', (e) => {
      if (e.target.closest('.mermaid-toolbar')) return;
      if (scrollWrap.scrollWidth <= scrollWrap.clientWidth && scrollWrap.scrollHeight <= scrollWrap.clientHeight) {
        return;
      }
      isPanning = true;
      scrollWrap.classList.add('is-grabbing');
      panStartX = e.pageX - scrollWrap.offsetLeft;
      panStartY = e.pageY - scrollWrap.offsetTop;
      initialScrollLeft = scrollWrap.scrollLeft;
      initialScrollTop = scrollWrap.scrollTop;
    });

    window.addEventListener('mousemove', (e) => {
      if (!isPanning) return;
      e.preventDefault();
      const x = e.pageX - scrollWrap.offsetLeft;
      const y = e.pageY - scrollWrap.offsetTop;
      scrollWrap.scrollLeft = initialScrollLeft - (x - panStartX);
      scrollWrap.scrollTop = initialScrollTop - (y - panStartY);
    });

    window.addEventListener('mouseup', () => {
      if (isPanning) {
        isPanning = false;
        scrollWrap.classList.remove('is-grabbing');
      }
    });
  }

  function enhanceAllDiagrams() {
    const mermaidElements = document.querySelectorAll('.mermaid');
    mermaidElements.forEach(setupDiagramCard);
  }

  // Subscribe to Material for MkDocs instant navigation
  if (typeof document$ !== 'undefined') {
    document$.subscribe(() => {
      enhanceAllDiagrams();
      // Polling check for asynchronous Mermaid rendering
      let checks = 0;
      const timer = setInterval(() => {
        enhanceAllDiagrams();
        checks++;
        if (checks >= 10) clearInterval(timer);
      }, 300);
    });
  } else {
    document.addEventListener('DOMContentLoaded', () => {
      enhanceAllDiagrams();
      let checks = 0;
      const timer = setInterval(() => {
        enhanceAllDiagrams();
        checks++;
        if (checks >= 10) clearInterval(timer);
      }, 300);
    });
  }
})();
