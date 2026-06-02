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

  function showCopied(button, labels) {
    if (!button) return;

    var icon = button.querySelector("i");
    var defaultTitle = button.getAttribute("data-default-title") || button.title;
    var defaultLabel = button.getAttribute("aria-label") || defaultTitle;
    var copiedTitle = labels.copiedTitle || "Copied!";
    var copiedLabel = labels.copiedLabel || copiedTitle;

    if (icon) {
      var defaultIcon = button.getAttribute("data-default-icon") || icon.className;
      if (!button.getAttribute("data-default-icon")) {
        button.setAttribute("data-default-icon", defaultIcon);
      }
      icon.className = "fas fa-check";
    }

    button.classList.add("is-copied");
    button.title = copiedTitle;
    button.setAttribute("aria-label", copiedLabel);

    window.setTimeout(function () {
      if (icon) {
        icon.className = button.getAttribute("data-default-icon");
      }
      button.classList.remove("is-copied");
      button.title = defaultTitle;
      button.setAttribute("aria-label", defaultLabel);
    }, 1800);
  }

  function bindCopyButton(button, getText, labels) {
    if (!button || button.getAttribute("data-copy-bound") === "true") return;

    button.setAttribute("data-copy-bound", "true");
    if (!button.getAttribute("data-default-title")) {
      button.setAttribute("data-default-title", button.title || "");
    }

    button.addEventListener("click", function () {
      var text = getText(button);
      if (!text) return;

      copyText(text)
        .then(function () {
          showCopied(button, labels);
        })
        .catch(function () {});
    });
  }

  function initNoteCopyButtons() {
    document.querySelectorAll(".note-copy-link").forEach(function (button) {
      bindCopyButton(
        button,
        function (btn) {
          return btn.getAttribute("data-copy-url");
        },
        { copiedTitle: "Link copied!", copiedLabel: "Link copied" }
      );
    });

    document.querySelectorAll(".note-copy-text").forEach(function (button) {
      bindCopyButton(
        button,
        function (btn) {
          var card = btn.closest(".note-card");
          var source = card ? card.querySelector(".note-copy-plain") : null;
          return source ? source.value : "";
        },
        { copiedTitle: "Text copied!", copiedLabel: "Text copied" }
      );
    });
  }

  function initPostCopyButtons() {
    document.querySelectorAll(".post-copy-link").forEach(function (button) {
      bindCopyButton(
        button,
        function () {
          return window.location.href.split("#")[0];
        },
        { copiedTitle: "Link copied!", copiedLabel: "Link copied" }
      );
    });

    document.querySelectorAll(".post-copy-text").forEach(function (button) {
      bindCopyButton(
        button,
        function (btn) {
          var source = document.getElementById(btn.getAttribute("data-copy-target"));
          return source ? source.value : "";
        },
        { copiedTitle: "Text copied!", copiedLabel: "Text copied" }
      );
    });
  }

  function init() {
    initNoteCopyButtons();
    initPostCopyButtons();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  window.initNoteCopyButtons = initNoteCopyButtons;
})();
