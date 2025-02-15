

let scrollBtn = document.getElementById("scroll-btn");
let lastScrollTop = 0;

window.onscroll = function() {
  let currentScroll = document.documentElement.scrollTop || document.body.scrollTop;

  if (currentScroll <= 1000) {
   
    scrollBtn.style.display = "none";
  } else if (currentScroll > lastScrollTop) {
   
    scrollBtn.style.display = "none";
  } else {
   
    scrollBtn.style.display = "block";
  }

  lastScrollTop = currentScroll <= 0 ? 0 : currentScroll;
}

function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}
