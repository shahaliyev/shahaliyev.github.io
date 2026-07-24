(function () {
  function loadVideo(root) {
    var id = root.getAttribute('data-youtube-id');
    if (!id || root.getAttribute('data-loaded') === '1') return;
    root.setAttribute('data-loaded', '1');
    var iframe = document.createElement('iframe');
    iframe.src = 'https://www.youtube-nocookie.com/embed/' + encodeURIComponent(id) + '?autoplay=1';
    iframe.title = 'YouTube video';
    iframe.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
    iframe.allowFullscreen = true;
    iframe.loading = 'lazy';
    root.innerHTML = '';
    root.appendChild(iframe);
  }

  document.addEventListener('click', function (e) {
    var btn = e.target.closest('.video-embed-load');
    if (!btn) return;
    var root = btn.closest('.video-embed');
    if (root) loadVideo(root);
  });
})();
