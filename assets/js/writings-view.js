(function () {
  function initWritingsHub() {
    var hub = document.querySelector(".writings-hub");
    if (!hub) return;

    var lang = hub.getAttribute("data-lang") || "en";
    var notesEnabled = hub.getAttribute("data-notes-enabled") === "true";
    var pageTitle = hub.getAttribute("data-page-title") || "";
    var writingsPanel = document.getElementById("writings-panel");
    var notesPanel = document.getElementById("notes-panel");
    var toggleButtons = hub.querySelectorAll(".writings-view-btn");
    var searchContainer = hub.querySelector(".search-container");
    var searchInput = document.getElementById("search-input");

    if (!writingsPanel) return;

    var labels = {
      writings: hub.getAttribute("data-label-writings") || "Posts",
      notes: hub.getAttribute("data-label-notes") || "Notes"
    };

    var placeholders = {
      writings: lang === "az" ? "Yazılarda axtar..." : "Search posts...",
      notes: lang === "az" ? "Qeydlərdə axtar..." : "Search notes..."
    };

    var paths = {
      writings: lang === "az" ? "/az/writings.html" : "/writings.html",
      notes: "/notes/"
    };

    var searchIndex = null;
    var currentView = hub.getAttribute("data-initial-view") || "writings";

    function filterIndex(scope) {
      if (!searchIndex) return [];
      return searchIndex.filter(function (item) {
        if (!notesEnabled && item.type === "note") return false;
        if (lang === "az" && item.type === "note") return false;
        if (item.lang && item.lang !== lang) return false;
        if (scope === "writings") return item.type === "writing";
        if (scope === "notes") return item.type === "note";
        return true;
      });
    }

    function initSearch(scope) {
      if (!searchInput || typeof SimpleJekyllSearch === "undefined") return;

      var resultsContainer = document.getElementById("results-container");
      if (!resultsContainer) return;

      searchInput.value = "";
      resultsContainer.innerHTML = "";
      if (searchContainer) {
        searchContainer.setAttribute("data-search-scope", scope);
      }
      searchInput.placeholder = placeholders[scope] || placeholders.writings;

      SimpleJekyllSearch({
        searchInput: searchInput,
        resultsContainer: resultsContainer,
        json: filterIndex(scope)
      });
    }

    function updateToggleButtons(view) {
      if (!toggleButtons.length) return;

      toggleButtons.forEach(function (btn) {
        var isActive = btn.getAttribute("data-view") === view;
        btn.classList.toggle("active", isActive);
        btn.classList.toggle("is-posts", isActive && view === "writings");
        btn.classList.toggle("is-writings", isActive && view === "writings");
        btn.classList.toggle("is-notes", isActive && view === "notes");
        btn.setAttribute("aria-pressed", isActive ? "true" : "false");
        if (notesEnabled) {
          btn.setAttribute("aria-selected", isActive ? "true" : "false");
        }
      });
    }

    function setView(view, updateHistory) {
      if (view === "notes" && !notesEnabled) {
        view = "writings";
      }
      if (view !== "writings" && view !== "notes") return;
      currentView = view;

      writingsPanel.classList.toggle("is-active", view === "writings");
      if (notesPanel) {
        notesPanel.classList.toggle("is-active", view === "notes");
      }

      updateToggleButtons(view);

      if (notesEnabled) {
        document.title = labels[view];
      } else if (pageTitle) {
        document.title = pageTitle;
      }

      if (searchIndex) {
        initSearch(view);
      }

      if (!notesEnabled || updateHistory === false) return;

      if (window.history && window.history.replaceState) {
        var onNotesPage = location.pathname.indexOf("/notes") !== -1;
        if (onNotesPage) {
          window.history.replaceState({ view: view }, "", view === "notes" ? paths.notes : paths.writings);
        } else {
          window.history.replaceState(
            { view: view },
            "",
            paths.writings + (view === "notes" ? "#notes" : "")
          );
        }
      }
    }

    function finishInit() {
      if (notesEnabled && (location.pathname.indexOf("/notes") !== -1 || location.hash === "#notes")) {
        currentView = "notes";
      } else {
        currentView = "writings";
      }

      setView(currentView, false);
    }

    if (notesEnabled) {
      window.addEventListener("popstate", function () {
        if (location.pathname.indexOf("/notes") !== -1 || location.hash === "#notes") {
          setView("notes", false);
        } else {
          setView("writings", false);
        }
      });

      toggleButtons.forEach(function (btn) {
        btn.addEventListener("click", function () {
          setView(btn.getAttribute("data-view"));
        });
      });
    }

    fetch("/search.json")
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        searchIndex = data;
        finishInit();
      })
      .catch(function () {
        finishInit();
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initWritingsHub);
  } else {
    initWritingsHub();
  }
})();
