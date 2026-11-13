// Sidebar Panel Toggle Controller for LinStabPy Documentation
// Provides interactive collapse/expand toggles for left (navigation) and right (TOC) sidebars
// with localStorage persistence, smooth transitions without page-load CLS, and keyboard shortcuts.
(function () {
  'use strict';

  // Prevent console errors from GitHub API requests on private repo
  if (!window._fetchPatched) {
    const origFetch = window.fetch;
    window.fetch = function (url, options) {
      if (typeof url === "string" && url.includes("api.github.com/repos/openfluids/linstabpy")) {
        return Promise.resolve(new Response(JSON.stringify({ stargazers_count: 0, forks_count: 0 }), {
          status: 200,
          headers: { "Content-Type": "application/json" }
        }));
      }
      return origFetch.apply(this, arguments);
    };
    window._fetchPatched = true;
  }

  const STORAGE_KEY_PRIMARY = "linstabpy-sidebar-primary";
  const STORAGE_KEY_SECONDARY = "linstabpy-sidebar-secondary";

  function getSavedState(key) {
    return localStorage.getItem(key) === "collapsed" ? "collapsed" : "expanded";
  }

  function setSavedState(key, state) {
    localStorage.setItem(key, state);
  }

  function triggerTransition() {
    document.body.classList.add("sidebar-transitioning");
    setTimeout(() => {
      document.body.classList.remove("sidebar-transitioning");
    }, 260);
  }

  function applySidebarState(target, state) {
    if (state === "collapsed") {
      document.documentElement.setAttribute(`data-sidebar-${target}`, "collapsed");
    } else {
      document.documentElement.removeAttribute(`data-sidebar-${target}`);
    }
  }

  // Early application to avoid flash of un-collapsed sidebar
  applySidebarState("primary", getSavedState(STORAGE_KEY_PRIMARY));
  applySidebarState("secondary", getSavedState(STORAGE_KEY_SECONDARY));

  function updateToggleIcons() {
    const leftBtn = document.getElementById("toggle-sidebar-primary");
    if (leftBtn) {
      const isCollapsed = document.documentElement.getAttribute("data-sidebar-primary") === "collapsed";
      leftBtn.classList.toggle("is-collapsed", isCollapsed);
      leftBtn.title = isCollapsed
        ? "Expand navigation sidebar (Ctrl+B)"
        : "Collapse navigation sidebar (Ctrl+B)";
    }

    const rightBtn = document.getElementById("toggle-sidebar-secondary");
    if (rightBtn) {
      const isCollapsed = document.documentElement.getAttribute("data-sidebar-secondary") === "collapsed";
      rightBtn.classList.toggle("is-collapsed", isCollapsed);
      rightBtn.title = isCollapsed
        ? "Expand table of contents (Ctrl+Shift+B)"
        : "Collapse table of contents (Ctrl+Shift+B)";
    }
  }

  function bindToggle(id, target) {
    const btn = document.getElementById(id);
    if (!btn || btn._bound) return;
    btn._bound = true;

    btn.addEventListener("click", () => {
      triggerTransition();
      const storageKey = target === "primary" ? STORAGE_KEY_PRIMARY : STORAGE_KEY_SECONDARY;
      const current = getSavedState(storageKey);
      const next = current === "collapsed" ? "expanded" : "collapsed";
      setSavedState(storageKey, next);
      applySidebarState(target, next);
      updateToggleIcons();
    });
  }

  function mountToggles() {
    // 1. Bind Server-rendered Header Buttons
    bindToggle("toggle-sidebar-primary", "primary");
    bindToggle("toggle-sidebar-secondary", "secondary");

    // 2. In-Sidebar Inner Collapse Buttons
    const primaryTitle = document.querySelector(".md-sidebar--primary .md-nav--primary > .md-nav__title");
    if (primaryTitle && !primaryTitle.querySelector(".cds-sidebar-inner-toggle")) {
      const innerToggle = document.createElement("button");
      innerToggle.type = "button";
      innerToggle.className = "cds-sidebar-inner-toggle";
      innerToggle.title = "Collapse sidebar (Ctrl+B)";
      innerToggle.setAttribute("aria-label", "Collapse navigation sidebar");
      innerToggle.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="11 17 6 12 11 7"/><polyline points="18 17 13 12 18 7"/></svg>`;
      innerToggle.addEventListener("click", (e) => {
        e.stopPropagation();
        const btn = document.getElementById("toggle-sidebar-primary");
        if (btn) btn.click();
      });
      primaryTitle.appendChild(innerToggle);
    }

    const secondaryTitle = document.querySelector(".md-sidebar--secondary .md-nav--secondary > .md-nav__title");
    if (secondaryTitle && !secondaryTitle.querySelector(".cds-sidebar-inner-toggle")) {
      const innerToggle = document.createElement("button");
      innerToggle.type = "button";
      innerToggle.className = "cds-sidebar-inner-toggle";
      innerToggle.title = "Collapse table of contents (Ctrl+Shift+B)";
      innerToggle.setAttribute("aria-label", "Collapse table of contents");
      innerToggle.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="13 17 18 12 13 7"/><polyline points="6 17 11 12 6 7"/></svg>`;
      innerToggle.addEventListener("click", (e) => {
        e.stopPropagation();
        const btn = document.getElementById("toggle-sidebar-secondary");
        if (btn) btn.click();
      });
      secondaryTitle.appendChild(innerToggle);
    }

    // 3. Accessibility compliance: add missing ARIA labels for Lighthouse 100
    const searchEl = document.querySelector(".md-search");
    if (searchEl && !searchEl.getAttribute("aria-label")) {
      searchEl.setAttribute("aria-label", "Search documentation");
    }

    const progressEl = document.querySelector(".md-progress");
    if (progressEl && !progressEl.getAttribute("aria-label")) {
      progressEl.setAttribute("aria-label", "Page loading progress");
    }

    // Synchronize initial state
    applySidebarState("primary", getSavedState(STORAGE_KEY_PRIMARY));
    applySidebarState("secondary", getSavedState(STORAGE_KEY_SECONDARY));
    updateToggleIcons();
  }

  // Keyboard shortcut listener: Ctrl+B / Cmd+B for left, Ctrl+Shift+B / Cmd+Shift+B for right
  document.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName) || document.activeElement?.isContentEditable) {
      return;
    }
    const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;
    const modifier = isMac ? e.metaKey : e.ctrlKey;

    if (modifier && !e.shiftKey && e.key.toLowerCase() === "b") {
      e.preventDefault();
      const btn = document.getElementById("toggle-sidebar-primary");
      if (btn) btn.click();
    } else if (modifier && e.shiftKey && e.key.toLowerCase() === "b") {
      e.preventDefault();
      const btn = document.getElementById("toggle-sidebar-secondary");
      if (btn) btn.click();
    }
  });

  if (window.document$) {
    window.document$.subscribe(mountToggles);
  } else {
    document.addEventListener("DOMContentLoaded", mountToggles);
  }
})();
