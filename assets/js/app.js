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
    document.addEventListener("DOMContentLoaded", initScrollButton);
  } else {
    initScrollButton();
  }
})();
