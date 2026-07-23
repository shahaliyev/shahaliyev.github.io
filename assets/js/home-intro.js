(function () {
  var CREDIT = "by Lana Chess Photography";

  function initProfileCredit() {
    var figure = document.querySelector(".profile-figure");
    if (!figure) return;

    var credit = figure.querySelector(".profile-credit");
    if (!credit) {
      credit = document.createElement("div");
      credit.className = "profile-credit";
      credit.setAttribute("aria-hidden", "true");
      figure.appendChild(credit);
    }

    credit.hidden = false;
    credit.textContent = CREDIT;

    function moveCredit(event) {
      var rect = figure.getBoundingClientRect();
      var x = event.clientX - rect.left + 14;
      var y = event.clientY - rect.top + 14;
      var maxX = Math.max(8, rect.width - credit.offsetWidth - 8);
      var maxY = Math.max(8, rect.height - credit.offsetHeight - 8);
      credit.style.left = Math.max(8, Math.min(x, maxX)) + "px";
      credit.style.top = Math.max(8, Math.min(y, maxY)) + "px";
    }

    figure.addEventListener("mouseenter", function (event) {
      credit.classList.add("is-visible");
      moveCredit(event);
    });
    figure.addEventListener("mousemove", moveCredit);
    figure.addEventListener("mouseleave", function () {
      credit.classList.remove("is-visible");
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initProfileCredit);
  } else {
    initProfileCredit();
  }
})();
