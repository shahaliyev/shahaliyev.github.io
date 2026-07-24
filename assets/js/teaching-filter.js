(function () {
  function semesterSortKey(label) {
    var match = String(label || "").match(/^(Fall|Spring|Summer)\s+(\d{4})$/i);
    if (!match) return Number.MAX_SAFE_INTEGER;
    var season = { spring: 1, summer: 2, fall: 3 }[match[1].toLowerCase()] || 0;
    return parseInt(match[2], 10) * 10 + season;
  }

  function initTeachingFilters() {
    var panel = document.getElementById("teaching-panel");
    if (!panel) return;

    var buttons = panel.querySelectorAll(".teaching-nav .writings-nav-btn");
    var items = panel.querySelectorAll(".teaching-item");
    var select = document.getElementById("teaching-semester-select");
    if (!buttons.length || !items.length || !select) return;

    var currentCourse = "all";
    var currentSemester = "all";
    var semesterSet = {};
    var allCourseBtn = panel.querySelector('.teaching-nav .writings-nav-btn[data-course="all"]');

    items.forEach(function (item) {
      var raw = item.getAttribute("data-semesters") || "";
      raw.split(",").forEach(function (part) {
        var value = part.trim();
        if (value) semesterSet[value] = true;
      });
    });

    Object.keys(semesterSet)
      .sort(function (a, b) {
        return semesterSortKey(a) - semesterSortKey(b);
      })
      .forEach(function (semester) {
        var option = document.createElement("option");
        option.value = semester;
        option.textContent = semester;
        select.appendChild(option);
      });

    function setActiveCourseButton(course) {
      buttons.forEach(function (btn) {
        btn.classList.toggle("active", (btn.getAttribute("data-course") || "") === course);
      });
    }

    function applyFilters() {
      items.forEach(function (item) {
        var course = item.getAttribute("data-course") || "";
        var semesters = (item.getAttribute("data-semesters") || "")
          .split(",")
          .map(function (part) {
            return part.trim();
          })
          .filter(Boolean);
        var courseMatch = currentCourse === "all" || course === currentCourse;
        var semesterMatch =
          currentSemester === "all" || semesters.indexOf(currentSemester) !== -1;
        item.style.display = courseMatch && semesterMatch ? "" : "none";
      });
    }

    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        currentCourse = btn.getAttribute("data-course") || "all";
        currentSemester = "all";
        select.value = "all";
        setActiveCourseButton(currentCourse);
        applyFilters();
      });
    });

    select.addEventListener("change", function () {
      currentSemester = select.value || "all";
      currentCourse = "all";
      if (allCourseBtn) {
        setActiveCourseButton("all");
      }
      applyFilters();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initTeachingFilters);
  } else {
    initTeachingFilters();
  }
})();
