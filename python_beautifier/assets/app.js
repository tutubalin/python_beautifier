/* python_beautifier - tiny, dependency-free page behaviour.
   Everything degrades gracefully: without JS the page is fully readable. */
(function () {
  "use strict";
  var doc = document;
  var root = doc.documentElement;
  var $ = function (s, r) { return (r || doc).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || doc).querySelectorAll(s)); };
  var store = {
    get: function (k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* file:// may forbid storage */ } }
  };

  /* ---- theme ------------------------------------------------------------ */
  var saved = store.get("pb-theme");
  if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);
  function effectiveTheme() {
    var t = root.getAttribute("data-theme");
    if (t === "light" || t === "dark") return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function syncThemeButton() {
    var dark = effectiveTheme() === "dark";
    root.classList.toggle("is-dark", dark);
  }
  var themeBtn = $("#b-theme");
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      store.set("pb-theme", next);
      syncThemeButton();
    });
  }
  syncThemeButton();

  /* ---- toolbar ---------------------------------------------------------- */
  function setAll(collapsed) {
    $$(".card").forEach(function (c) { c.classList.toggle("collapsed", collapsed); });
    if (!collapsed) $$("details.box, details.arm, details.impl").forEach(function (d) { d.open = true; });
  }
  var b;
  if ((b = $("#b-expand"))) b.addEventListener("click", function () { setAll(false); });
  if ((b = $("#b-collapse"))) b.addEventListener("click", function () { setAll(true); });
  function toggleClass(btn, cls, invert) {
    if (!btn) return;
    btn.addEventListener("click", function () {
      var pressed = btn.getAttribute("aria-pressed") !== "false";
      pressed = !pressed;
      btn.setAttribute("aria-pressed", String(pressed));
      doc.body.classList.toggle(cls, invert ? !pressed : pressed);
    });
  }
  toggleClass($("#b-lines"), "no-lines", true);
  toggleClass($("#b-comments"), "no-comments", true);
  toggleClass($("#b-index"), "no-index", true);
  if ((b = $("#b-print"))) b.addEventListener("click", function () { setAll(false); window.print(); });
  if ((b = $("#menu"))) b.addEventListener("click", function () { doc.body.classList.toggle("menu-open"); });
  doc.addEventListener("click", function (e) {
    if (doc.body.classList.contains("menu-open") && !e.target.closest(".side") && !e.target.closest("#menu")) {
      doc.body.classList.remove("menu-open");
    }
    var a = e.target.closest && e.target.closest(".side a[href^='#']");
    if (a) doc.body.classList.remove("menu-open");
  });

  /* ---- card collapse / copy -------------------------------------------- */
  doc.addEventListener("click", function (e) {
    var tog = e.target.closest && e.target.closest(".tog");
    if (tog) {
      var card = tog.closest(".card");
      if (card) card.classList.toggle("collapsed");
      return;
    }
    var cpy = e.target.closest && e.target.closest(".cpy");
    if (cpy) {
      var card2 = cpy.closest(".card");
      var srcEl = $("#pb-source");
      if (!card2 || !srcEl) return;
      var lines;
      try { lines = JSON.parse(srcEl.textContent).split("\n"); } catch (err) { return; }
      var a = parseInt(card2.getAttribute("data-l1"), 10) - 1;
      var z = parseInt(card2.getAttribute("data-l2"), 10);
      var text = lines.slice(a, z).join("\n");
      cpy.classList.add("done");
      setTimeout(function () { cpy.classList.remove("done"); }, 1400);
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).catch(function () { fallbackCopy(text); });
        } else { fallbackCopy(text); }
      } catch (err2) { fallbackCopy(text); }
    }
  });
  function fallbackCopy(text) {
    var ta = doc.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    doc.body.appendChild(ta); ta.select();
    try { doc.execCommand("copy"); } catch (e) { /* ignore */ }
    doc.body.removeChild(ta);
  }

  /* ---- jump to a card: expand it and flash it -------------------------- */
  function reveal(id) {
    var el = id && doc.getElementById(id);
    if (!el) return;
    var p = el;
    while (p) {
      if (p.classList && p.classList.contains("card")) p.classList.remove("collapsed");
      if (p.tagName === "DETAILS") p.open = true;
      p = p.parentElement;
    }
    var card = el.closest ? el.closest(".card") : null;
    if (card && card === el) {
      card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash");
    }
  }
  window.addEventListener("hashchange", function () { reveal(decodeURIComponent(location.hash.slice(1))); });
  if (location.hash) reveal(decodeURIComponent(location.hash.slice(1)));
  doc.addEventListener("click", function (e) {
    var a = e.target.closest && e.target.closest("a[href^='#']");
    if (a && a.getAttribute("href").length > 1 && !a.closest(".side")) {
      setTimeout(function () { reveal(decodeURIComponent(a.getAttribute("href").slice(1))); }, 0);
    }
  });

  /* ---- sidebar: tree toggles, search, scrollspy ------------------------ */
  $$(".ol-t").forEach(function (t) {
    t.addEventListener("click", function () { t.parentElement.classList.toggle("collapsed"); });
  });
  var q = $("#q");
  var items = $$(".outline li");
  var empty = $(".ol-empty");
  function filter() {
    var term = (q.value || "").trim().toLowerCase();
    var any = false;
    items.forEach(function (li) { li.hidden = false; li.classList.remove("hit"); });
    if (term) {
      items.forEach(function (li) {
        var name = li.getAttribute("data-name") || li.textContent.toLowerCase();
        var own = name.indexOf(term) !== -1;
        li.classList.toggle("hit", own);
      });
      items.forEach(function (li) {
        var keep = li.classList.contains("hit") || li.querySelector(".hit");
        li.hidden = !keep;
        if (keep && li.querySelector(":scope > .ol-sub")) li.classList.remove("collapsed");
        if (keep) any = true;
      });
    } else { any = true; }
    $$(".ol-h").forEach(function (h) {
      var ul = h.nextElementSibling;
      h.hidden = !!term && ul && !ul.querySelector("li:not([hidden])");
    });
    if (empty) empty.hidden = any;
  }
  if (q) {
    q.addEventListener("input", filter);
    q.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        var first = $(".outline li:not([hidden]) > a[data-target]");
        if (first) { location.hash = first.getAttribute("data-target"); q.blur(); }
      } else if (e.key === "Escape") { q.value = ""; filter(); q.blur(); }
    });
    doc.addEventListener("keydown", function (e) {
      if (e.key === "/" && !/^(INPUT|TEXTAREA)$/.test((doc.activeElement || {}).tagName || "")) {
        e.preventDefault(); q.focus(); q.select();
      }
    });
  }
  var links = {};
  $$(".outline a[data-target]").forEach(function (a) { links[a.getAttribute("data-target")] = a; });
  var cards = $$(".card.def");
  var current = null;
  function setCurrent(id) {
    if (id === current) return;
    current = id;
    $$(".outline a.on").forEach(function (a) { a.classList.remove("on"); });
    var a = links[id];
    if (a) {
      a.classList.add("on");
      var li = a.parentElement;
      while (li && li.classList) {
        if (li.tagName === "LI" && li.classList.contains("collapsed")) li.classList.remove("collapsed");
        li = li.parentElement;
      }
      var nav = $(".outline");
      if (nav) {
        var r = a.getBoundingClientRect(), nr = nav.getBoundingClientRect();
        if (r.top < nr.top + 40 || r.bottom > nr.bottom - 40) nav.scrollTop += r.top - nr.top - nr.height / 3;
      }
    }
  }
  if ("IntersectionObserver" in window && cards.length) {
    var visible = new Map();
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) visible.set(en.target, en.boundingClientRect.top);
        else visible.delete(en.target);
      });
      var best = null, bestTop = Infinity;
      visible.forEach(function (_, el) {
        var top = Math.abs(el.getBoundingClientRect().top - 90);
        var depth = 0, p = el.parentElement;
        while (p) { if (p.classList && p.classList.contains("card")) depth++; p = p.parentElement; }
        var score = top - depth * 4;
        if (score < bestTop) { bestTop = score; best = el; }
      });
      if (best) setCurrent(best.id);
    }, { rootMargin: "-70px 0px -55% 0px", threshold: [0, 0.01] });
    cards.forEach(function (c) { io.observe(c); });
  }

  /* ---- indexing guide: point at a term and its explanation lights up -- */
  function ixLight(e, on) {
    var t = e.target.closest && e.target.closest("[data-ix]");
    if (!t || (e.relatedTarget && t.contains(e.relatedTarget))) return;
    $$('[data-ix="' + t.getAttribute("data-ix") + '"]').forEach(function (el) { el.classList.toggle("ix-on", on); });
  }
  doc.addEventListener("mouseover", function (e) { ixLight(e, true); });
  doc.addEventListener("mouseout", function (e) { ixLight(e, false); });

  /* ---- hover a variable: highlight its uses inside the same card ------- */
  var hasHL = typeof CSS !== "undefined" && CSS.highlights && typeof Highlight !== "undefined";
  var lastWord = null;
  function wordAt(x, y) {
    var node, off;
    if (doc.caretPositionFromPoint) {
      var p = doc.caretPositionFromPoint(x, y);
      if (!p) return null;
      node = p.offsetNode; off = p.offset;
    } else if (doc.caretRangeFromPoint) {
      var r = doc.caretRangeFromPoint(x, y);
      if (!r) return null;
      node = r.startContainer; off = r.startOffset;
    } else return null;
    if (!node || node.nodeType !== 3) return null;
    var t = node.data, s = off, e = off;
    var isId = function (ch) { return /[A-Za-z0-9_]/.test(ch); };
    while (s > 0 && isId(t[s - 1])) s--;
    while (e < t.length && isId(t[e])) e++;
    var w = t.slice(s, e);
    if (!w || /^[0-9]/.test(w) || w.length < 2) return null;
    return w;
  }
  function clearHL() { if (hasHL) CSS.highlights.delete("pb-var"); lastWord = null; }
  if (hasHL) {
    var timer = null;
    doc.addEventListener("mousemove", function (ev) {
      var t = ev.target;
      if (!t.closest || !t.closest(".flow code.c, .sig, .gc code, .bh code")) { if (lastWord) clearHL(); return; }
      if (t.closest(".tc, .note .nt, .s")) return;
      clearTimeout(timer);
      var x = ev.clientX, y = ev.clientY;
      timer = setTimeout(function () {
        var w = wordAt(x, y);
        if (!w) { clearHL(); return; }
        if (w === lastWord) return;
        var scope = t.closest(".card") || doc.body;
        var ranges = [];
        var re = new RegExp("(^|[^A-Za-z0-9_])(" + w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")(?![A-Za-z0-9_])", "g");
        var walker = doc.createTreeWalker(scope, NodeFilter.SHOW_TEXT, {
          acceptNode: function (n) {
            var p = n.parentElement;
            if (!p) return NodeFilter.FILTER_REJECT;
            if (p.closest("code.c, .sig, .gc, .bh")) {
              if (p.closest(".s, .cm, .tc, .es")) return NodeFilter.FILTER_REJECT;
              return NodeFilter.FILTER_ACCEPT;
            }
            return NodeFilter.FILTER_REJECT;
          }
        });
        var n;
        while ((n = walker.nextNode())) {
          var m;
          re.lastIndex = 0;
          while ((m = re.exec(n.data))) {
            var start = m.index + m[1].length;
            var rg = new Range();
            rg.setStart(n, start); rg.setEnd(n, start + w.length);
            ranges.push(rg);
          }
        }
        if (ranges.length > 1) { CSS.highlights.set("pb-var", new Highlight(...ranges)); lastWord = w; }
        else clearHL();
      }, 60);
    }, { passive: true });
    doc.addEventListener("mouseleave", clearHL);
  }

  /* ---- chord diagram: hover a node to light up its calls --------------- */
  $$(".chords").forEach(function (box) {
    var chords = $$(".cd-chord", box);
    $$(".cd-node a", box).forEach(function (a) {
      var id = (a.getAttribute("href") || "").slice(1);
      a.addEventListener("mouseenter", function () {
        chords.forEach(function (c) {
          var hit = c.getAttribute("data-a") === id || c.getAttribute("data-b") === id;
          c.classList.toggle("hot", hit);
          c.classList.toggle("dim", !hit);
        });
      });
      a.addEventListener("mouseleave", function () {
        chords.forEach(function (c) { c.classList.remove("hot", "dim"); });
      });
    });
  });
})();
