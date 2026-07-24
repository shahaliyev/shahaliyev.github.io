(function () {
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }

    return new Promise(function (resolve, reject) {
      var textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      try {
        document.execCommand("copy");
        document.body.removeChild(textarea);
        resolve();
      } catch (err) {
        document.body.removeChild(textarea);
        reject(err);
      }
    });
  }

  function showCopied(button) {
    button.classList.add("is-copied");
    button.title = "Copied!";
    button.setAttribute("aria-label", "Copied!");

    setTimeout(function () {
      button.classList.remove("is-copied");
      button.title = "Copy";
      button.setAttribute("aria-label", "Copy BibTeX");
    }, 1400);
  }

  function initResearchBibtex() {
    var panel = document.getElementById("research-panel");
    if (!panel) return;

    panel.querySelectorAll(".research-bibtex-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var id = btn.getAttribute("data-bibtex-dialog");
        var dialog = id ? document.getElementById(id) : null;
        if (dialog && typeof dialog.showModal === "function") {
          dialog.showModal();
        }
      });
    });

    panel.querySelectorAll(".research-bibtex-dialog").forEach(function (dialog) {
      var closeBtn = dialog.querySelector(".research-bibtex-close");
      var copyBtn = dialog.querySelector(".research-bibtex-copy");
      var code = dialog.querySelector(".research-bibtex-code");

      if (closeBtn) {
        closeBtn.addEventListener("click", function () {
          dialog.close();
        });
      }

      dialog.addEventListener("click", function (event) {
        if (event.target === dialog) dialog.close();
      });

      if (copyBtn && code) {
        copyBtn.addEventListener("click", function () {
          copyText(code.textContent || "").then(function () {
            showCopied(copyBtn);
          });
        });
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initResearchBibtex);
  } else {
    initResearchBibtex();
  }
})();
