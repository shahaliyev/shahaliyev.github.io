(function () {
  function initWritingsHub() {
    var hub = document.querySelector(".writings-hub");
    if (!hub) return;

    var lang = hub.getAttribute("data-lang") || "en";
    var notesEnabled = hub.getAttribute("data-notes-enabled") === "true";
    var researchEnabled = hub.getAttribute("data-research-enabled") === "true";
    var pageTitle = hub.getAttribute("data-page-title") || "";
    var writingsPanel = document.getElementById("writings-panel");
    var notesPanel = document.getElementById("notes-panel");
    var researchPanel = document.getElementById("research-panel");
    var toggleButtons = hub.querySelectorAll(".writings-view-btn");
    var searchContainer = hub.querySelector(".search-container");
    var searchInput = document.getElementById("search-input");

    if (!writingsPanel) return;

    var labels = {
      writings: hub.getAttribute("data-label-writings") || "Writings",
      notes: hub.getAttribute("data-label-notes") || "Notes",
      research: hub.getAttribute("data-label-research") || "Research"
    };

    var placeholders = {
      writings: lang === "az" ? "Yazılarda axtar..." : "Search writings...",
      notes: lang === "az" ? "Qeydlərdə axtar..." : "Search notes...",
      research: lang === "az" ? "Tədqiqatda axtar..." : "Search research..."
    };

    var isHome = hub.getAttribute("data-is-home") === "true";
    var hubBase = hub.getAttribute("data-hub-base");
    if (!hubBase) {
      hubBase = lang === "az" ? "/az/" : "/";
    }
    var hubHome = hubBase.replace(/\/?$/, "/");

    var viewHashes = {
      writings: "#writings",
      notes: "#notes",
      research: "#research"
    };

    var searchIndex = null;
    var currentView = hub.getAttribute("data-initial-view") || "writings";

    function getNotesTagFromUrl() {
      if (typeof window.getNotesTagFromUrl === "function") {
        return window.getNotesTagFromUrl();
      }
      return null;
    }

    function hashToView(hash) {
      var h = hash || "";
      if (h === "#notes" || h === "#writings-hub") return "notes";
      if (h === "#research") return "research";
      if (h === "#writings") return "writings";
      return null;
    }

    function isViewEnabled(view) {
      if (view === "notes") return notesEnabled;
      if (view === "research") return researchEnabled;
      return view === "writings";
    }

    function scrollToWritingsHub() {
      var el = document.getElementById("writings-hub");
      if (!el) return;
      window.requestAnimationFrame(function () {
        var top = el.getBoundingClientRect().top + window.scrollY;
        window.scrollTo({ top: top, behavior: "smooth" });
      });
    }

    function resolveViewFromLocation() {
      var fromHash = hashToView(location.hash);
      if (fromHash && isViewEnabled(fromHash)) {
        return fromHash;
      }
      if (notesEnabled && isHome && getNotesTagFromUrl() !== null) {
        return "notes";
      }
      return "writings";
    }

    function filterIndex(scope) {
      if (!searchIndex) return [];
      return searchIndex.filter(function (item) {
        if (!notesEnabled && item.type === "note") return false;
        if (!researchEnabled && item.type === "research") return false;
        if (lang === "az" && item.type === "note") return false;
        if (item.lang && item.lang !== lang && item.type === "writing") return false;
        if (scope === "writings") return item.type === "writing";
        if (scope === "notes") return item.type === "note";
        if (scope === "research") return item.type === "research";
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
        btn.classList.toggle("is-research", isActive && view === "research");
        btn.setAttribute("aria-pressed", isActive ? "true" : "false");
        btn.setAttribute("aria-selected", isActive ? "true" : "false");
      });
    }

    function setView(view, updateHistory) {
      if (!isViewEnabled(view)) {
        view = "writings";
      }
      if (view !== "writings" && view !== "notes" && view !== "research") return;
      currentView = view;

      writingsPanel.classList.toggle("is-active", view === "writings");
      if (notesPanel) {
        notesPanel.classList.toggle("is-active", view === "notes");
      }
      if (researchPanel) {
        researchPanel.classList.toggle("is-active", view === "research");
      }

      updateToggleButtons(view);

      if ((notesEnabled || researchEnabled) && !isHome) {
        document.title = labels[view] || pageTitle;
      } else if (pageTitle) {
        document.title = pageTitle;
      }

      if (searchIndex) {
        initSearch(view);
      }

      if (updateHistory === false || !isHome) return;

      if (window.history && window.history.replaceState) {
        var hash = viewHashes[view] || "";
        var query = window.location.search || "";
        // Drop notes_tag when leaving notes view
        if (view !== "notes" && query.indexOf("notes_tag=") !== -1) {
          try {
            var params = new URLSearchParams(query);
            params.delete("notes_tag");
            var next = params.toString();
            query = next ? "?" + next : "";
          } catch (err) {
            query = "";
          }
        }
        window.history.replaceState({ view: view }, "", hubHome + query + hash);
      }
    }

    function finishInit() {
      currentView = resolveViewFromLocation();
      setView(currentView, false);

      if (currentView === "notes") {
        var notesTag = getNotesTagFromUrl();
        if (notesTag && typeof window.applyNotesTagFromUrl === "function") {
          window.applyNotesTagFromUrl();
        }
      }

      if (currentView === "notes" || currentView === "research" || getNotesTagFromUrl()) {
        scrollToWritingsHub();
      }

      // Normalize legacy #writings-hub → #notes
      if (isHome && location.hash === "#writings-hub" && notesEnabled && window.history && window.history.replaceState) {
        window.history.replaceState({ view: "notes" }, "", hubHome + (window.location.search || "") + "#notes");
      }
    }

    window.addEventListener("popstate", function () {
      var view = resolveViewFromLocation();
      setView(view, false);
      if (view === "notes") {
        var notesTag = getNotesTagFromUrl();
        if (notesTag && typeof window.applyNotesTagFromUrl === "function") {
          window.applyNotesTagFromUrl();
        }
      }
      if (view === "notes" || view === "research") {
        scrollToWritingsHub();
      }
    });

    toggleButtons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        setView(btn.getAttribute("data-view"));
      });
    });

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
