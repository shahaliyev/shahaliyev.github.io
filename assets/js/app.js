(function () {
  function getScrollTop() {
    return Math.max(
      window.pageYOffset || 0,
      document.documentElement.scrollTop || 0,
      document.body.scrollTop || 0
    );
  }

  function scrollToTop() {
    if (getScrollTop() === 0) return;

    var root = document.scrollingElement || document.documentElement;

    function scrollInstant() {
      window.scrollTo(0, 0);
      document.documentElement.scrollTop = 0;
      document.body.scrollTop = 0;
      if (root) {
        root.scrollTop = 0;
      }
    }

    var usedSmooth = false;

    try {
      window.scrollTo({ top: 0, left: 0, behavior: "smooth" });
      usedSmooth = true;
    } catch (e) {
      scrollInstant();
      return;
    }

    if (root && root.scrollTo) {
      try {
        root.scrollTo({ top: 0, left: 0, behavior: "smooth" });
        usedSmooth = true;
      } catch (e) {
        scrollInstant();
        return;
      }
    }

    if (!usedSmooth) {
      scrollInstant();
      return;
    }

    window.setTimeout(function () {
      if (getScrollTop() > 5) {
        scrollInstant();
      }
    }, 120);
  }

  window.scrollToTop = scrollToTop;

  function initBlockquoteFootnotes() {
    document.querySelectorAll("blockquote").forEach(function (blockquote) {
      var refs = Array.prototype.slice.call(
        blockquote.querySelectorAll('sup[role="doc-noteref"]')
      );
      if (!refs.length) return;

      var target = blockquote.querySelector("p:last-child") || blockquote;
      var group = document.createElement("span");
      group.className = "blockquote-footnote-refs";

      refs.forEach(function (ref) {
        ref.parentNode.removeChild(ref);
      });

      refs.forEach(function (ref, index) {
        if (index === 0) {
          group.appendChild(document.createTextNode("\u2009"));
        }
        group.appendChild(ref);
      });

      target.appendChild(group);
    });
  }

  function initFootnoteTooltips() {
    var links = document.querySelectorAll('a.footnote[href^="#fn"]');
    if (!links.length) return;

    var activeLink = null;
    var hideTimer = null;
    var tip = document.createElement("div");
    tip.className = "footnote-tooltip";
    tip.setAttribute("role", "tooltip");
    tip.hidden = true;
    document.body.appendChild(tip);

    function getBounds() {
      var main = document.querySelector("main");
      var rect = main ? main.getBoundingClientRect() : null;
      var pad = 8;
      if (rect && rect.width > 40) {
        return {
          left: rect.left + pad,
          right: rect.right - pad,
          width: Math.max(120, rect.width - pad * 2)
        };
      }
      return {
        left: pad,
        right: window.innerWidth - pad,
        width: Math.max(120, window.innerWidth - pad * 2)
      };
    }

    function footnoteHtml(href) {
      var id = href.replace(/^#/, "");
      var note = document.getElementById(id);
      if (!note) return "";
      var clone = note.cloneNode(true);
      clone.querySelectorAll(".reversefootnote").forEach(function (el) {
        el.remove();
      });
      var html = "";
      Array.prototype.forEach.call(clone.children, function (child) {
        html += child.outerHTML || child.textContent || "";
      });
      if (!html) {
        html = "<p>" + (clone.textContent || "").replace(/\s+/g, " ").trim() + "</p>";
      }
      return html.trim();
    }

    function positionTip(link) {
      var bounds = getBounds();
      var anchor = link.getBoundingClientRect();
      var maxWidth = Math.min(20 * 16, bounds.width);
      tip.style.width = "auto";
      tip.style.maxWidth = maxWidth + "px";
      tip.hidden = false;
      tip.classList.add("is-visible");

      var tipRect = tip.getBoundingClientRect();
      var width = Math.min(tipRect.width, maxWidth);
      var left = anchor.left + anchor.width / 2 - width / 2;
      left = Math.max(bounds.left, Math.min(left, bounds.right - width));

      var gap = 8;
      var top = anchor.top - tipRect.height - gap;
      if (top < 8) {
        top = anchor.bottom + gap;
      }

      tip.style.left = left + "px";
      tip.style.top = top + "px";
      tip.style.width = width + "px";
    }

    function showTip(link) {
      var html = footnoteHtml(link.getAttribute("href") || "");
      if (!html) return;
      clearTimeout(hideTimer);
      activeLink = link;
      tip.innerHTML = html;
      tip.setAttribute("id", "footnote-tooltip-live");
      link.setAttribute("aria-describedby", "footnote-tooltip-live");
      positionTip(link);
    }

    function hideTip() {
      hideTimer = setTimeout(function () {
        if (activeLink) {
          activeLink.removeAttribute("aria-describedby");
          activeLink = null;
        }
        tip.classList.remove("is-visible");
        tip.hidden = true;
        tip.innerHTML = "";
      }, 120);
    }

    links.forEach(function (link) {
      link.addEventListener("mouseenter", function () {
        showTip(link);
      });
      link.addEventListener("mouseleave", hideTip);
      link.addEventListener("focus", function () {
        showTip(link);
      });
      link.addEventListener("blur", hideTip);
    });

    tip.addEventListener("mouseenter", function () {
      clearTimeout(hideTimer);
    });
    tip.addEventListener("mouseleave", hideTip);

    window.addEventListener(
      "scroll",
      function () {
        if (activeLink && tip.classList.contains("is-visible")) {
          positionTip(activeLink);
        }
      },
      { passive: true }
    );

    window.addEventListener("resize", function () {
      if (activeLink && tip.classList.contains("is-visible")) {
        positionTip(activeLink);
      }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && tip.classList.contains("is-visible")) {
        hideTip();
        if (activeLink) activeLink.blur();
      }
    });
  }

  function initScrollButton() {
    var scrollBtn = document.getElementById("scroll-btn");
    if (!scrollBtn) return;

    var threshold = 200;
    var lastScrollTop = getScrollTop();
    var visible = false;
    var ticking = false;

    function setVisible(show) {
      if (show === visible) return;
      visible = show;
      scrollBtn.classList.toggle("is-visible", show);
      scrollBtn.setAttribute("aria-hidden", show ? "false" : "true");
      scrollBtn.tabIndex = show ? 0 : -1;
    }

    function updateScrollButton() {
      var scrollTop = getScrollTop();

      if (scrollTop <= threshold) {
        setVisible(false);
      } else if (scrollTop < lastScrollTop) {
        setVisible(true);
      } else if (scrollTop > lastScrollTop) {
        setVisible(false);
      }

      lastScrollTop = scrollTop;
    }

    function onScroll() {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(function () {
        updateScrollButton();
        ticking = false;
      });
    }

    function handleActivate(event) {
      if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") {
        return;
      }
      if (event.type === "keydown") {
        event.preventDefault();
      }
      scrollToTop();
      lastScrollTop = 0;
      setVisible(false);
    }

    scrollBtn.addEventListener("click", handleActivate);
    scrollBtn.addEventListener("keydown", handleActivate);

    window.addEventListener("scroll", onScroll, { passive: true });
    document.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    updateScrollButton();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      initScrollButton();
      initBlockquoteFootnotes();
      initFootnoteTooltips();
    });
  } else {
    initScrollButton();
    initBlockquoteFootnotes();
    initFootnoteTooltips();
  }
})();
