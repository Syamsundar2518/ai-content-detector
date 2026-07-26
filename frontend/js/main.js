/* =========================================================================
   main.js
   -------
   This file runs on EVERY page. It handles things that are shared across
   the whole site:
     1. Light/Dark mode toggle (and remembering the choice)
     2. Mobile menu open/close
     3. FAQ accordion (click a question to expand the answer)

   "DOMContentLoaded" means: "wait until the page's HTML has fully loaded,
   THEN run this code." This prevents errors from trying to find elements
   that don't exist yet.
   ========================================================================= */

document.addEventListener('DOMContentLoaded', () => {

  /* ---------- 1. DARK / LIGHT MODE ---------- */
  const themeToggleBtn = document.querySelector('.theme-toggle');
  const savedTheme = localStorage.getItem('ai-detector-theme');

  // Apply saved preference on page load (default = dark mode)
  if (savedTheme === 'light') {
    document.body.classList.add('light-mode');
  }

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      document.body.classList.toggle('light-mode');
      const isLight = document.body.classList.contains('light-mode');
      localStorage.setItem('ai-detector-theme', isLight ? 'light' : 'dark');
    });
  }

  /* ---------- 2. MOBILE MENU ---------- */
  const mobileToggle = document.querySelector('.mobile-toggle');
  const navLinks = document.querySelector('.nav-links');

  if (mobileToggle && navLinks) {
    mobileToggle.addEventListener('click', () => {
      const isOpen = navLinks.classList.toggle('mobile-open');
      // Simple inline styles so we don't need extra CSS classes for this
      navLinks.style.display = isOpen ? 'flex' : 'none';
      navLinks.style.flexDirection = 'column';
      navLinks.style.position = 'absolute';
      navLinks.style.top = '72px';
      navLinks.style.left = '0';
      navLinks.style.right = '0';
      navLinks.style.background = 'var(--surface)';
      navLinks.style.padding = '20px 24px';
      navLinks.style.borderBottom = '1px solid var(--border)';
    });
  }

  /* ---------- 3. FAQ ACCORDION ---------- */
  document.querySelectorAll('.faq-item').forEach((item) => {
    const question = item.querySelector('.faq-question');
    const answer = item.querySelector('.faq-answer');

    question.addEventListener('click', () => {
      const isOpen = item.classList.contains('open');

      // Close all other FAQ items first (accordion behavior)
      document.querySelectorAll('.faq-item').forEach((other) => {
        other.classList.remove('open');
        other.querySelector('.faq-answer').style.maxHeight = null;
      });

      // Then open this one, unless it was already open (then leave it closed)
      if (!isOpen) {
        item.classList.add('open');
        answer.style.maxHeight = answer.scrollHeight + 40 + 'px';
      }
    });
  });

  /* ---------- 4. Animate hero meter bars on the home page (if present) ---------- */
  document.querySelectorAll('.meter-fill[data-target]').forEach((el) => {
    const target = el.getAttribute('data-target');
    requestAnimationFrame(() => {
      setTimeout(() => { el.style.width = target + '%'; }, 300);
    });
  });

});
