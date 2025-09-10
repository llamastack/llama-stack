document.addEventListener('keydown', function(event) {
  // command+K or ctrl+K
  if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
    event.preventDefault();
    document.querySelector('.search-input, .search-field, input[name="q"]').focus();
  }

  // forward slash
  if (event.key === '/' &&
      !event.target.matches('input, textarea, select')) {
    event.preventDefault();
    document.querySelector('.search-input, .search-field, input[name="q"]').focus();
  }
});
