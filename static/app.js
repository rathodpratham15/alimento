/* ==========================================================
   Alimento — shared interactive behaviors (vanilla JS)
   ========================================================== */

(function () {
  'use strict';

  // ---- Theme toggle ----------------------------------------------------
  const root = document.documentElement;
  function applyTheme(theme) {
    if (theme === 'dark') root.classList.add('dark');
    else root.classList.remove('dark');
    localStorage.setItem('alimento-theme', theme);
    document.querySelectorAll('[data-theme-icon]').forEach((el) => {
      el.className = theme === 'dark'
        ? 'ph ph-sun text-[18px]'
        : 'ph ph-moon text-[18px]';
    });
  }
  window.__alimentoToggleTheme = function () {
    const isDark = root.classList.contains('dark');
    applyTheme(isDark ? 'light' : 'dark');
  };

  // Initialize icon to match current theme
  document.addEventListener('DOMContentLoaded', () => {
    const t = root.classList.contains('dark') ? 'dark' : 'light';
    applyTheme(t);
  });

  // ---- Mobile nav toggle -----------------------------------------------
  window.__alimentoToggleMobileNav = function () {
    const panel = document.getElementById('mobile-nav');
    if (!panel) return;
    panel.classList.toggle('hidden');
  };

  // ---- Tools dropdown ---------------------------------------------------
  window.__alimentoToggleTools = function (e) {
    e && e.stopPropagation();
    const dd = document.getElementById('tools-dropdown');
    if (!dd) return;
    dd.classList.toggle('hidden');
  };
  document.addEventListener('click', (e) => {
    const dd = document.getElementById('tools-dropdown');
    if (!dd || dd.classList.contains('hidden')) return;
    if (!e.target.closest('#tools-wrap')) dd.classList.add('hidden');
  });

  // ---- User dropdown ----------------------------------------------------
  window.__alimentoToggleUser = function (e) {
    e && e.stopPropagation();
    const dd = document.getElementById('user-dropdown');
    if (!dd) return;
    dd.classList.toggle('hidden');
  };
  document.addEventListener('click', (e) => {
    const dd = document.getElementById('user-dropdown');
    if (!dd || dd.classList.contains('hidden')) return;
    if (!e.target.closest('#user-wrap')) dd.classList.add('hidden');
  });

  // ---- Hide-on-scroll-down nav -----------------------------------------
  let lastY = 0;
  let ticking = false;
  function onScroll() {
    const nav = document.getElementById('alimento-nav');
    if (!nav) return;
    const y = window.scrollY;
    if (y < 8) { nav.classList.remove('nav-hidden'); }
    else if (y > lastY + 6) { nav.classList.add('nav-hidden'); }
    else if (y < lastY - 6) { nav.classList.remove('nav-hidden'); }
    lastY = y;
    ticking = false;
  }
  window.addEventListener('scroll', () => {
    if (!ticking) { requestAnimationFrame(onScroll); ticking = true; }
  }, { passive: true });

  // ---- Guest banner dismiss --------------------------------------------
  window.__alimentoDismissGuest = function () {
    const b = document.getElementById('guest-banner');
    if (b) b.style.display = 'none';
  };

  // ---- Scroll reveal ----------------------------------------------------
  const io = ('IntersectionObserver' in window)
    ? new IntersectionObserver((entries) => {
        entries.forEach((en) => {
          if (en.isIntersecting) {
            en.target.classList.add('visible');
            io.unobserve(en.target);
          }
        });
      }, { threshold: 0.08 })
    : null;
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.reveal').forEach((el, i) => {
      el.style.transitionDelay = (i % 6) * 60 + 'ms';
      if (io) io.observe(el); else el.classList.add('visible');
    });
  });

  // ---- Generic open/close helpers --------------------------------------
  window.__alimentoOpen = function (id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('open');
  };
  window.__alimentoClose = function (id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('open');
  };

  // ---- Collapsible cards (AI insights, etc.) ---------------------------
  window.__alimentoToggleCollapse = function (btn) {
    const card = btn.closest('[data-collapsible]');
    if (!card) return;
    const body = card.querySelector('[data-collapsible-body]');
    const icon = btn.querySelector('[data-chevron]');
    if (body.classList.contains('hidden')) {
      body.classList.remove('hidden');
      if (icon) icon.classList.remove('rotate-180');
    } else {
      body.classList.add('hidden');
      if (icon) icon.classList.add('rotate-180');
    }
  };

  // ---- Tabs (Inventory, Progress, etc.) --------------------------------
  window.__alimentoTab = function (group, value, btn) {
    document.querySelectorAll(`[data-tabs="${group}"] [data-tab]`).forEach((el) => {
      el.classList.toggle('active', el.dataset.tab === value);
    });
    document.querySelectorAll(`[data-tab-panel-group="${group}"]`).forEach((el) => {
      el.classList.toggle('hidden', el.dataset.tabPanel !== value);
    });
  };

  // ---- Macro slider live preview ---------------------------------------
  window.__alimentoMacroChange = function () {
    const p = parseInt(document.getElementById('macro-protein')?.value || 0, 10);
    const c = parseInt(document.getElementById('macro-carbs')?.value || 0, 10);
    const f = parseInt(document.getElementById('macro-fat')?.value || 0, 10);
    const cal = parseInt(document.getElementById('cal-goal')?.value || 2000, 10);
    const total = p + c + f;
    const setText = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
    setText('macro-protein-val', p + '%');
    setText('macro-carbs-val', c + '%');
    setText('macro-fat-val', f + '%');
    setText('macro-protein-g', Math.round((cal * p / 100) / 4) + ' g');
    setText('macro-carbs-g', Math.round((cal * c / 100) / 4) + ' g');
    setText('macro-fat-g', Math.round((cal * f / 100) / 9) + ' g');
    const sum = document.getElementById('macro-sum');
    if (sum) {
      sum.textContent = total + '%';
      sum.className = 'tabular font-semibold ' + (total === 100 ? 'text-success' : 'text-danger');
    }
  };

  // ---- Chat send (UI-only) ---------------------------------------------
  window.__alimentoSendChat = function () {
    const input = document.getElementById('chat-input');
    if (!input || !input.value.trim()) return;
    const stream = document.getElementById('chat-stream');
    const empty = document.getElementById('chat-empty');
    if (empty) empty.classList.add('hidden');
    const time = new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    const userMsg = document.createElement('div');
    userMsg.className = 'flex flex-col items-end gap-1 reveal visible';
    userMsg.innerHTML = `<div class="bubble-user">${input.value.replace(/</g,'&lt;')}</div><span class="text-[11px] text-muted">${time}</span>`;
    stream.appendChild(userMsg);
    input.value = '';
    stream.scrollTop = stream.scrollHeight;
    // Mock AI reply
    setTimeout(() => {
      const reply = document.createElement('div');
      reply.className = 'flex flex-col items-start gap-1 reveal visible';
      reply.innerHTML = `<div class="bubble-ai"><span class="text-text">Based on your keto profile and today's macros, try a salmon avocado bowl — it'll bring you back on protein without spiking carbs.</span></div><span class="text-[11px] text-muted">${new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</span>`;
      stream.appendChild(reply);
      stream.scrollTop = stream.scrollHeight;
    }, 600);
  };

  // ---- Suggested prompt chip click -------------------------------------
  window.__alimentoUsePrompt = function (text) {
    const input = document.getElementById('chat-input');
    if (input) { input.value = text; input.focus(); }
  };

})();
