function fixNavTitles() {
  var selector = '.md-sidebar--primary .md-nav__title, .md-sidebar--secondary .md-nav__title';
  document.querySelectorAll(selector).forEach(function (el) {
    el.style.setProperty('background',       'transparent', 'important');
    el.style.setProperty('background-color', 'transparent', 'important');
    el.style.setProperty('color',            '#64748b',     'important');
    el.style.setProperty('box-shadow',       'none',        'important');
  });
}

function addWBLogo() {
  if (document.querySelector('.wb-logo-wrapper')) return; // already added
  var esmap = document.querySelector('.md-header__button.md-logo img');
  if (!esmap) return;
  var option = document.querySelector('.md-header__option');
  if (!option) return;
  var wbSrc = esmap.getAttribute('src').replace('esmap.png', 'worldbank.png');
  var wrapper = document.createElement('div');
  wrapper.className = 'wb-logo-wrapper';
  wrapper.innerHTML =
    '<a href="https://www.worldbank.org" target="_blank" rel="noopener" title="The World Bank">' +
    '<img src="' + wbSrc + '" alt="The World Bank" class="wb-header-logo">' +
    '</a>';
  option.insertBefore(wrapper, option.firstChild);
}

function init() {
  addWBLogo();
  fixNavTitles();
}

// Run on load
document.addEventListener('DOMContentLoaded', init);

// Hook into Material's SPA navigation system
if (typeof document$ !== 'undefined') {
  document$.subscribe(function () {
    fixNavTitles();
  });
}

// Fallback: reapply after short delays to catch late renders
setTimeout(fixNavTitles, 200);
setTimeout(fixNavTitles, 800);
