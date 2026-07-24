(function () {
  function initResearchFilters() {
    var panel = document.getElementById("research-panel");
    if (!panel) return;

    var buttons = panel.querySelectorAll(".research-nav .writings-nav-btn");
    var items = panel.querySelectorAll(".research-item");
    if (!buttons.length || !items.length) return;

    function applyFilter(keyword) {
      var filter = keyword || "all";

      buttons.forEach(function (btn) {
        btn.classList.toggle("active", (btn.getAttribute("data-keyword") || "") === filter);
      });

      items.forEach(function (item) {
        var keywords = (item.getAttribute("data-keywords") || "")
          .split(",")
          .map(function (part) {
            return part.trim();
          })
          .filter(Boolean);
        var show = filter === "all" || keywords.indexOf(filter) !== -1;
        item.style.display = show ? "" : "none";
      });
    }

    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        applyFilter(btn.getAttribute("data-keyword") || "all");
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initResearchFilters);
  } else {
    initResearchFilters();
  }
})();
