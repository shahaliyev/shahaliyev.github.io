(function () {
  function filterNotes(tag) {
    var items = document.querySelectorAll(".note-item");
    var buttons = document.querySelectorAll(".notes-nav .writings-nav-btn");
    var filter = (tag || "all").toLowerCase();

    buttons.forEach(function (btn) {
      btn.classList.remove("active");
    });

    var matchBtn = Array.from(buttons).find(function (btn) {
      return btn.textContent.trim().toLowerCase() === filter;
    });
    if (matchBtn) {
      matchBtn.classList.add("active");
    }

    items.forEach(function (item) {
      var tags = (item.getAttribute("data-tags") || "").split(",");
      var show = filter === "all" || tags.includes(filter);
      item.style.display = show ? "block" : "none";
    });
  }

  function getNotesTagFromUrl() {
    try {
      var value = new URLSearchParams(window.location.search).get("notes_tag");
      return value ? value.trim().toLowerCase() : null;
    } catch (err) {
      return null;
    }
  }

  function applyNotesTagFromUrl() {
    var tag = getNotesTagFromUrl();
    if (tag) {
      filterNotes(tag);
    }
  }

  window.filterNotes = filterNotes;
  window.getNotesTagFromUrl = getNotesTagFromUrl;
  window.applyNotesTagFromUrl = applyNotesTagFromUrl;
})();
