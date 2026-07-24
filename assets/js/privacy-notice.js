(function () {
  var trigger = document.getElementById('privacy-notice-trigger');
  var dialog = document.getElementById('privacy-notice-dialog');
  if (!trigger || !dialog) return;

  trigger.addEventListener('click', function () {
    if (typeof dialog.showModal === 'function') {
      dialog.showModal();
      trigger.setAttribute('aria-expanded', 'true');
    } else {
      dialog.setAttribute('open', '');
      trigger.setAttribute('aria-expanded', 'true');
    }
  });

  dialog.addEventListener('close', function () {
    trigger.setAttribute('aria-expanded', 'false');
  });

  dialog.addEventListener('click', function (e) {
    if (e.target === dialog && typeof dialog.close === 'function') {
      dialog.close();
    }
  });
})();
