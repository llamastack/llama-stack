// Horizontal Navigation Bar for Llama Stack Documentation
document.addEventListener('DOMContentLoaded', function() {
    // Create the horizontal navigation HTML
    const navHTML = `
        <nav class="horizontal-nav">
            <div class="nav-container">
                <a href="/" class="nav-brand">Llama Stack</a>
                <ul class="nav-links">
                    <li><a href="/">Docs</a></li>
                    <li><a href="/references/api_reference/">API Reference</a></li>
                    <li><a href="https://github.com/meta-llama/llama-stack" target="_blank" class="github-link">
                        <svg class="github-icon" viewBox="0 0 16 16" aria-hidden="true">
                            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                        </svg>
                        GitHub
                    </a></li>
                </ul>
            </div>
        </nav>
    `;

    // Insert the navigation at the beginning of the body
    document.body.insertAdjacentHTML('afterbegin', navHTML);

    // Update navigation links based on current page
    updateActiveNav();
});

function updateActiveNav() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.horizontal-nav .nav-links a');

    navLinks.forEach(link => {
        // Remove any existing active classes
        link.classList.remove('active');

        // Add active class based on current path
        if (currentPath === '/' && link.getAttribute('href') === '/') {
            link.classList.add('active');
        } else if (currentPath.includes('/references/api_reference/') && link.getAttribute('href').includes('api_reference')) {
            link.classList.add('active');
        }
    });
}
