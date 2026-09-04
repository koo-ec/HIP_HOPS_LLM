/* Page-aware links to external assistants; no API keys or background requests. */
(function () {
  "use strict";
  const widget = document.querySelector(".ask-ai");
  if (!widget) return;
  const trigger = widget.querySelector(".ask-ai__trigger");
  const panel = widget.querySelector(".ask-ai__panel");
  const promptField = widget.querySelector("#ask-ai-prompt");
  const context = widget.querySelector(".ask-ai__context");
  const status = widget.querySelector(".ask-ai__status");

  // Use the current version on RTD or Pages, but a public URL in local previews.
  const current = new URL(window.location.href);
  const publicHost = current.hostname === "koorosh-aslansefat.com" ||
    current.hostname === "koo-ec.github.io" ||
    /(^|\.)readthedocs\.(io|org)$/.test(current.hostname);
  const home = publicHost
    ? new URL(widget.dataset.docsHome, current)
    : new URL("index.html", widget.dataset.publicHome);
  const page = publicHost
    ? new URL(current.pathname, current.origin)
    : new URL(widget.dataset.page + ".html", widget.dataset.publicHome);
  const prompt = [
    "Help me understand HIP-HOPS-LLM: " + widget.dataset.pageTitle + ".",
    "Current documentation page: " + page.href,
    "Documentation home: " + home.href,
    "Source repository: https://github.com/koo-ec/HIP_HOPS_LLM",
    "",
    "The project combines HiP-HOPS failure propagation with HIP-LLM's hierarchical imprecise reliability assessment for LLM-based agentic systems. It connects system architecture, fault trees, minimal cut sets, FMEA, operational-profile calibration and Bayesian networks.",
    "Explain this page, give a relevant example, and suggest what to read next. Cite the documentation. Distinguish measured reliability intervals from placeholder probabilities. If you cannot access a source, say so and ask me to paste the relevant section."
  ].join("\n");
  promptField.value = prompt;
  trigger.hidden = false;

  function close(restoreFocus) {
    panel.hidden = true;
    trigger.setAttribute("aria-expanded", "false");
    if (restoreFocus) trigger.focus();
  }

  async function copyPrompt() {
    try {
      if (!navigator.clipboard || !window.isSecureContext) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(prompt);
      status.textContent = "Prompt copied. Paste it into your chosen assistant.";
    } catch (_) {
      context.open = true;
      promptField.focus();
      promptField.select();
      status.textContent = "Copy the selected prompt manually, then paste it into your assistant.";
    }
  }

  trigger.addEventListener("click", function () {
    if (!panel.hidden) return close(false);
    panel.hidden = false;
    trigger.setAttribute("aria-expanded", "true");
    panel.querySelector("[data-provider]").focus();
  });
  widget.querySelector(".ask-ai__close").addEventListener("click", () => close(true));
  widget.querySelector(".ask-ai__copy").addEventListener("click", copyPrompt);
  widget.querySelectorAll("[data-provider]").forEach(function (link) {
    // Keep native link navigation synchronous, including modifier-click behaviour.
    link.addEventListener("click", copyPrompt);
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !panel.hidden) {
      event.preventDefault();
      close(true);
    }
  });
  document.addEventListener("click", function (event) {
    if (!panel.hidden && !widget.contains(event.target)) close(false);
  });
  widget.addEventListener("focusout", function (event) {
    if (event.relatedTarget && !widget.contains(event.relatedTarget)) close(false);
  });
})();
