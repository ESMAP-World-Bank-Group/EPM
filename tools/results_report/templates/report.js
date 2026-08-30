/* Black Sea results report — chart layer.
 *
 * Everything is drawn as inline SVG from window.RD, the cache extract.py wrote.
 * No external library, so the file works offline and survives being emailed.
 *
 * A chart is declared in the HTML as
 *     <div class="chart" data-spec='{"type":"stack","kind":"cap",...}'></div>
 * and this file fills it in on load.  Specs stay tiny: they name a scope and a
 * chart type, and the series are derived here from RD.
 */
(function () {
  "use strict";

  var RD = window.RD;
  var NS = "http://www.w3.org/2000/svg";

  /* ------------------------------------------------------------ palette */

  var COLORS = {
    Nuclear: "#C8A8F0", Coal: "#808890", Gas: "#9A7040", CCGT: "#B8921A",
    OCGT: "#C4A820", Diesel: "#6A7888", HFO: "#7A7068", Oil: "#7A7068",
    Biomass: "#52C860", Waste: "#8A9098", Geothermal: "#D4A820",
    Reservoir: "#1E9AF5", ROR: "#5DADE2", Hydro: "#1E9AF5", PSH: "#0D7680",
    Solar: "#FFD700", PV: "#FFD700", CSP: "#E8C547", RPV: "#FFD700",
    "Onshore Wind": "#44DAEC", Wind: "#44DAEC", "Offshore Wind": "#7CC8FA",
    Battery: "#6A7BC8", Storage: "#8290CE",
    Imports: "#2E9EC8", Exports: "#E8C547", "Storage Charge": "#0D7680",
    "Unmet demand": "#D9534F", Demand: "#8B0000",
    "Interconnection capacity": "#7f8fa6"
  };
  var MAPBAND = ["#1B6CA8", "#36B5B5", "#E8C547", "#4DA6FF", "#4169E1",
                 "#85C1E9", "#2E9EC8", "#5EBCBA", "#1A5276", "#7EC8E3",
                 "#14A094", "#4CAFE8", "#EDD770", "#AED6F1", "#1F618D"];
  var SCEN_LABEL = { LC_Baseline: "Baseline", LC_Iso: "Isolated" };

  function colorOf(k, i) {
    return COLORS[k] || MAPBAND[(i || 0) % MAPBAND.length];
  }
  function label(scen) { return SCEN_LABEL[scen] || scen.replace(/^LC_/, ""); }

  /* ------------------------------------------------------------ plumbing */

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) {
      if (attrs[k] !== null && attrs[k] !== undefined) {
        n.setAttribute(k, attrs[k]);
      }
    }
    if (parent) parent.appendChild(n);
    return n;
  }

  /* Deck mode.  A chart pasted two-per-slide is scaled to about half its
     painted width, which takes an 8.5 px axis label down to roughly 4 pt.
     Rather than redraw at a different size, every type size and every margin
     that depends on one is multiplied by FS, so the plot area shrinks and the
     labels survive the scaling. */
  var FS = 1;
  var DECK = 1.55;

  function txt(parent, x, y, s, o) {
    o = o || {};
    var n = el("text", {
      x: r1(x), y: r1(y), "font-size": r1((o.size || 10) * FS),
      "text-anchor": o.anchor || "start", fill: o.fill || "#55627a",
      "font-weight": o.weight || 400,
      stroke: o.halo ? "#ffffff" : null,
      "stroke-width": o.halo ? o.halo : null,
      "stroke-linejoin": o.halo ? "round" : null,
      "paint-order": o.halo ? "stroke" : null,
      transform: o.rotate ? "rotate(" + o.rotate + " " + r1(x) + " " + r1(y) + ")" : null
    }, parent);
    n.textContent = s;
    return n;
  }

  function r1(v) { return Math.round(v * 10) / 10; }

  function fmt(v, dp) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    if (dp === undefined) dp = Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2;
    return v.toFixed(dp).replace(/\.?0+$/, function (m) {
      return m.indexOf(".") === 0 ? "" : m;
    });
  }

  /* One tooltip for the whole document, moved around on hover. */
  var TIP = document.createElement("div");
  TIP.className = "tip";
  document.body.appendChild(TIP);

  function tipOn(html, ev) {
    TIP.innerHTML = html;
    TIP.classList.add("on");
    tipMove(ev);
  }
  function tipMove(ev) {
    var pad = 14, w = TIP.offsetWidth, h = TIP.offsetHeight;
    var x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - pad;
    TIP.style.left = Math.max(4, x) + "px";
    TIP.style.top = Math.max(4, y) + "px";
  }
  function tipOff() { TIP.classList.remove("on"); }

  function hover(node, html) {
    node.addEventListener("mouseenter", function (e) { tipOn(html, e); });
    node.addEventListener("mousemove", tipMove);
    node.addEventListener("mouseleave", tipOff);
  }

  /* Diagonal hatch, used for everything that is not own generation:
     transmission capacity, imports, exports. */
  var HATCH_ID = 0;
  function hatch(defs, color) {
    var id = "h" + (++HATCH_ID);
    var p = el("pattern", {
      id: id, width: 5, height: 5, patternUnits: "userSpaceOnUse",
      patternTransform: "rotate(45)"
    }, defs);
    el("rect", { width: 5, height: 5, fill: color, "fill-opacity": .22 }, p);
    el("line", { x1: 0, y1: 0, x2: 0, y2: 5, stroke: color, "stroke-width": 2.4 }, p);
    return "url(#" + id + ")";
  }

  /* ------------------------------------------------------------- scaling */

  function niceAxis(lo, hi) {
    if (hi <= 0 && lo >= 0) { hi = 1; lo = 0; }
    if (lo > 0) lo = 0;
    if (hi < 0) hi = 0;
    var span = hi - lo || 1;
    var raw = span / 5;
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag;
    var step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
    var top = Math.ceil(hi / step) * step;
    var bot = Math.floor(lo / step) * step;
    var ticks = [];
    for (var v = bot; v <= top + step * 1e-6; v += step) {
      ticks.push(Math.abs(v) < step * 1e-6 ? 0 : v);
    }
    return { lo: bot, hi: top, step: step, ticks: ticks };
  }

  /* Width the key column takes out of the box, gap included. */
  var LEGW = 166;

  /* A chart is declared at the width of the box it sits in; the plot gets what
     is left once the key has taken its column, so type stays the same size in a
     half-width panel as in a full-width one.  Too narrow for a column and the
     key goes back under the plot, where it has the whole width to wrap in. */
  function boxOf(spec, dw, dh) {
    var w = spec.w || dw;
    var side = !spec.legendBelow && w >= 470;
    return { W: side ? w - LEGW : w, H: spec.h || dh, side: side };
  }

  function frame(host, w, h, m, side) {
    host.innerHTML = "";
    var row = document.createElement("div");
    row.className = "chartrow" + (side ? "" : " stackleg");
    var plot = document.createElement("div");
    plot.className = "plot";
    var leg = document.createElement("div");
    leg.className = "legendcol";
    row.appendChild(plot);
    row.appendChild(leg);
    host.appendChild(row);
    host._leg = leg;
    var s = el("svg", { viewBox: "0 0 " + w + " " + h,
                        preserveAspectRatio: "xMidYMid meet" }, plot);
    var defs = el("defs", {}, s);
    return { svg: s, defs: defs, w: w, h: h, m: m,
             iw: w - m.l - m.r, ih: h - m.t - m.b };
  }

  function gridY(f, ax, unit) {
    var y = function (v) {
      return f.m.t + f.ih * (ax.hi - v) / (ax.hi - ax.lo || 1);
    };
    ax.ticks.forEach(function (v) {
      el("line", { x1: f.m.l, y1: r1(y(v)), x2: f.m.l + f.iw, y2: r1(y(v)),
                   stroke: v === 0 ? "#c8d0dc" : "#eef1f6",
                   "stroke-width": v === 0 ? 1.2 : 1 }, f.svg);
      txt(f.svg, f.m.l - 6, y(v) + 3.2, fmt(v), { size: 9, anchor: "end", fill: "#9aa4b2" });
    });
    if (unit) {
      txt(f.svg, f.m.l - 6, f.m.t - 9, unit,
          { size: 9.5, anchor: "end", fill: "#8a94a6", weight: 600 });
    }
    return y;
  }

  function legend(host, items, note) {
    var d = document.createElement("div");
    d.className = "legend";
    if (FS !== 1) {
      d.style.fontSize = r1(11 * FS) + "px";
      d.style.gap = r1(4 * FS) + "px " + r1(13 * FS) + "px";
    }
    if (note) {
      var n0 = document.createElement("span");
      n0.style.color = "#8a94a6";
      n0.textContent = note;
      d.appendChild(n0);
    }
    var seen = {};
    items.filter(function (it) {
      var k = it.name + "|" + it.color + "|" + (it.hatch ? 1 : 0);
      if (seen[k]) return false;
      seen[k] = 1;
      return true;
    }).forEach(function (it) {
      var s = document.createElement("span");
      var sw = it.hatch
        ? "background:repeating-linear-gradient(45deg," + it.color + "," +
          it.color + " 2px,transparent 2px,transparent 4px);border:1px solid " + it.color
        : "background:" + it.color;
      if (FS !== 1) {
        sw += ";width:" + r1(11 * FS) + "px;height:" +
              r1((it.shape === "line" ? 2 : 11) * FS) + "px";
      }
      s.innerHTML = '<i class="' + (it.shape === "line" ? "line" : "") +
                    '" style="' + sw + '"></i>' + it.name;
      d.appendChild(s);
    });
    (host._leg || host).appendChild(d);
    return d;
  }

  /* ============================================================ stack chart
   * Annual capacity or generation: one stacked bar per (year, scenario),
   * with transmission capacity / imports / exports hatched on top.
   */
  function stackChart(host, spec) {
    var years = RD.years, scens = spec.scenarios || RD.scenarios;
    var b0 = boxOf(spec, 980, 340), W = b0.W, H = b0.H;
    var m = { l: r1(46 * FS), r: 14, t: 22, b: r1(54 * FS) };
    var f = frame(host, W, H, m, b0.side);

    var series = spec.series;                       // [{key,color,hatch,by:{scen:[..]}}]
    var lines = spec.lines || [];

    // Extent across every bar, positives stacked up, negatives stacked down.
    var hi = 0, lo = 0;
    years.forEach(function (_, i) {
      scens.forEach(function (sc) {
        var up = 0, dn = 0;
        series.forEach(function (s) {
          var v = (s.by[sc] || [])[i] || 0;
          if (v >= 0) up += v; else dn += v;
        });
        hi = Math.max(hi, up); lo = Math.min(lo, dn);
      });
    });
    lines.forEach(function (l) {
      scens.forEach(function (sc) {
        (l.by[sc] || []).forEach(function (v) {
          hi = Math.max(hi, v || 0);
          lo = Math.min(lo, v || 0);
        });
      });
    });
    var ax = niceAxis(lo, hi);
    var y = gridY(f, ax, spec.unit);

    var slot = f.iw / years.length;
    var pad = slot * 0.16;
    var bw = (slot - 2 * pad) / scens.length;
    var fills = {};
    series.forEach(function (s) {
      fills[s.key] = s.hatch ? hatch(f.defs, s.color) : s.color;
    });

    years.forEach(function (yr, i) {
      var x0 = m.l + i * slot + pad;
      scens.forEach(function (sc, si) {
        var bx = x0 + si * bw, up = 0, dn = 0;
        var rows = [], tot = 0;
        series.forEach(function (s) {
          var v = (s.by[sc] || [])[i] || 0;
          if (Math.abs(v) < 1e-9) return;
          var top, bot;
          if (v >= 0) { bot = up; up += v; top = up; }
          else { top = dn; dn += v; bot = dn; }
          var yt = y(top), yb = y(bot);
          var rect = el("rect", {
            x: r1(bx + .5), y: r1(Math.min(yt, yb)), width: r1(bw - 1),
            height: r1(Math.max(.8, Math.abs(yb - yt))),
            fill: fills[s.key], stroke: s.hatch ? s.color : "none",
            "stroke-width": s.hatch ? .7 : 0
          }, f.svg);
          rows.push({ k: s.key, v: v, c: s.color });
          if (!s.excludeFromTotal) tot += v;
          hover(rect, tipRows(yr + " · " + label(sc), rows.slice(-1), spec.unit));
        });
        // A wide invisible target so the whole bar answers the mouse.
        var full = el("rect", {
          x: r1(bx), y: m.t, width: r1(bw), height: f.ih,
          fill: "transparent"
        }, f.svg);
        hover(full, tipRows(yr + " · " + label(sc), rows, spec.unit,
                            spec.totalLabel || "Total", tot));

        if (si === 0 && scens.length > 1) {
          el("line", { x1: r1(bx + bw * scens.length), y1: m.t,
                       x2: r1(bx + bw * scens.length), y2: m.t + f.ih,
                       stroke: "#fff", "stroke-width": 0 }, f.svg);
        }
      });

      // Year label, centred on the group of bars by construction.
      if (years.length <= 8 || i % 2 === 0 || i === years.length - 1) {
        txt(f.svg, m.l + i * slot + slot / 2, m.t + f.ih + 14 * FS, yr,
            { size: 9.5, anchor: "middle", fill: "#55627a", weight: 600 });
      }
      // Scenario initials under each bar, so the pair is readable without a key.
      if (scens.length > 1 && years.length <= 20) {
        scens.forEach(function (sc, si) {
          txt(f.svg, x0 + si * bw + bw / 2, m.t + f.ih + 25 * FS,
              label(sc).slice(0, 1), { size: 7.5, anchor: "middle", fill: "#aab3c2" });
        });
      }
    });

    lines.forEach(function (l) {
      scens.forEach(function (sc, si) {
        if (l.scen && l.scen !== sc) return;
        var vals = l.by[sc]; if (!vals) return;
        var d = vals.map(function (v, i) {
          return (i ? "L" : "M") + r1(m.l + i * slot + slot / 2) + " " + r1(y(v || 0));
        }).join(" ");
        el("path", { d: d, fill: "none", stroke: l.color,
                     "stroke-width": 1.9 * FS,
                     "stroke-dasharray": (l.scen || !si) ? null : "4 3",
                     "stroke-linejoin": "round" }, f.svg);
      });
    });

    var items = series.map(function (s) {
      return { name: s.name || s.key, color: s.color, hatch: s.hatch };
    }).concat(lines.map(function (l) {
      return { name: l.name, color: l.color, shape: "line" };
    }));
    if (scens.length > 1) {
      items.push({ name: label(scens[0]) + " | " + label(scens[1]),
                   color: "#c8d0dc" });
    }
    legend(host, items, spec.note);
  }

  function tipRows(title, rows, unit, totLabel, tot) {
    var h = "<b>" + title + "</b>";
    if (!rows.length) return h;
    h += "<hr>";
    rows.slice().reverse().forEach(function (r) {
      h += '<span class="k">' + r.k + "</span> " + fmt(r.v) + " " + unit + "<br>";
    });
    if (totLabel !== undefined) {
      h += '<hr><span class="k">' + totLabel + "</span> <b>" + fmt(tot) +
           " " + unit + "</b>";
    }
    return h;
  }

  var NETIMPORT = "#12356e";          // navy, next to the dark red of demand

  /* ========================================================= dispatch chart
   * Stacked hourly dispatch over the representative year, matching the
   * convention of epm-data-explorer: generation stacked up, exports and
   * storage charging stacked down, demand as a dark line, marginal cost on a
   * right-hand axis.
   */
  function dispatchChart(host, spec) {
    var scen = spec.scenario, scope = spec.scope, yr = spec.year;
    var pack = RD.dispatch[scope][scen];
    var axis = pack.axis, rows = pack.years[yr] || {};
    var price = RD.price[scope][scen];
    var n = axis.length;
    if (!n) { host.innerHTML = '<p class="lede">no dispatch stored</p>'; return; }

    // Difference view: this scenario minus the reference, fuel by fuel and slot
    // by slot.  A fuel that did not move anywhere in the year is dropped, so the
    // stack carries the changes and nothing else.
    var minus = spec.minus || null, pref = null;
    if (minus) {
      var rows2 = (RD.dispatch[scope][minus] || { years: {} }).years[yr] || {};
      var keys = {}, dr = {};
      Object.keys(rows).concat(Object.keys(rows2)).forEach(function (k) { keys[k] = 1; });
      Object.keys(keys).forEach(function (k) {
        var a = rows[k] || [], b = rows2[k] || [], v = [], nz = false;
        for (var j = 0; j < n; j++) {
          v.push((a[j] || 0) - (b[j] || 0));
          if (Math.abs(v[j]) > 0.05) nz = true;
        }
        if (nz) dr[k] = v;
      });
      rows = dr;
      pref = RD.price[scope][minus] || {};
    }

    var b0 = boxOf(spec, 980, 330), W = b0.W, H = b0.H;
    var m = { l: r1(46 * FS), r: r1(44 * FS), t: 20, b: r1(46 * FS) };
    var f = frame(host, W, H, m, b0.side);

    var order = RD.fuel_order.concat(["Imports"]);
    var down = ["Exports", "Storage Charge"];
    // Trade is not own generation; hatching keeps it apart from the fuels it
    // sits against in the stack.  One pattern per key, built once.
    var HATCHED = { Imports: 1, Exports: 1 };
    // On a scope whose zones trade with each other, the extraction already
    // collapsed the two series into the net position: say so in the labels and
    // drop the net line, which would only trace the top of the band.
    var NETTED = (RD.dispatch_netted || []).indexOf(scope) >= 0;
    var LBL = NETTED
      ? { Imports: "Net imports", Exports: "Net exports" } : {};
    var lbl = function (k) { return LBL[k] || k; };
    var fillOf = {};
    function fillFor(k) {
      if (!(k in fillOf)) {
        fillOf[k] = HATCHED[k] ? hatch(f.defs, colorOf(k)) : colorOf(k);
      }
      return fillOf[k];
    }
    var up = order.filter(function (k) { return rows[k]; });
    var dn = down.filter(function (k) { return rows[k]; });
    if (rows["Unmet demand"]) up.push("Unmet demand");

    var bars = up.concat(dn);
    var hi = 0, lo = 0;
    for (var i = 0; i < n; i++) {
      var a = 0, b = 0;
      if (minus) {
        // A difference has no fixed side: the same fuel is up in one hour and
        // down in the next, so the sign of the value decides where it stacks.
        bars.forEach(function (k) {
          var v = rows[k][i] || 0;
          if (v >= 0) a += v; else b += v;
        });
      } else {
        up.forEach(function (k) { a += rows[k][i] || 0; });
        dn.forEach(function (k) { b += rows[k][i] || 0; });
        a = Math.max(a, (rows.Demand || [])[i] || 0);
      }
      hi = Math.max(hi, a);
      lo = Math.min(lo, b);
    }
    var ax = niceAxis(lo, hi);
    var y = gridY(f, ax, minus ? "Δ MW" : "MW");
    var x = function (i) { return m.l + f.iw * i / n; };
    var bw = f.iw / n;

    function band(keys, sign) {
      var base = new Array(n).fill(0);
      keys.forEach(function (k) {
        var v = rows[k], top = base.slice();
        for (var i = 0; i < n; i++) top[i] = base[i] + (v[i] || 0);
        var d = "M" + r1(x(0)) + " " + r1(y(base[0]));
        for (i = 0; i < n; i++) {
          d += " L" + r1(x(i)) + " " + r1(y(top[i])) +
               " L" + r1(x(i + 1)) + " " + r1(y(top[i]));
        }
        for (i = n - 1; i >= 0; i--) {
          d += " L" + r1(x(i + 1)) + " " + r1(y(base[i])) +
               " L" + r1(x(i)) + " " + r1(y(base[i]));
        }
        el("path", { d: d + " Z", fill: fillFor(k),
                     "fill-opacity": HATCHED[k] ? 1 : .72,
                     stroke: colorOf(k),
                     "stroke-width": HATCHED[k] ? .8 : .5 }, f.svg);
        base = top;
      });
      return base;
    }
    /* Same stack, split around zero: one path per fuel holding one box per
       slot, since a ribbon cannot follow a base that jumps sides. */
    function stackedBars(keys) {
      var upb = new Array(n).fill(0), dnb = new Array(n).fill(0);
      keys.forEach(function (k) {
        var v = rows[k], d = "";
        for (var i = 0; i < n; i++) {
          var val = v[i] || 0;
          if (Math.abs(val) < 0.05) continue;
          var b0 = val > 0 ? upb[i] : dnb[i], t = b0 + val;
          if (val > 0) upb[i] = t; else dnb[i] = t;
          d += "M" + r1(x(i)) + " " + r1(y(Math.max(b0, t))) +
               "H" + r1(x(i + 1)) + "V" + r1(y(Math.min(b0, t))) +
               "H" + r1(x(i)) + "Z";
        }
        if (d) el("path", { d: d, fill: fillFor(k),
                            "fill-opacity": HATCHED[k] ? 1 : .82,
                            stroke: HATCHED[k] ? colorOf(k) : "none",
                            "stroke-width": HATCHED[k] ? .8 : 0 }, f.svg);
      });
    }

    if (minus) {
      stackedBars(bars);
    } else {
      band(up, 1);
      band(dn, -1);
    }

    if (rows.Demand && !minus) {
      var d = "";
      for (i = 0; i < n; i++) {
        d += (i ? " L" : "M") + r1(x(i)) + " " + r1(y(rows.Demand[i])) +
             " L" + r1(x(i + 1)) + " " + r1(y(rows.Demand[i]));
      }
      el("path", { d: d, fill: "none", stroke: "#8B0000",
                   "stroke-width": 1.5 * FS }, f.svg);
    }

    // Net imports: Imports + Exports.  Exports and Storage Charge come out of
    // the model already negative, so the two series add rather than subtract.
    // Above zero the zone is a net buyer in that slot, below it a net seller.
    // Drawn last so it reads over the stack.
    var net = null;
    if (!minus && !NETTED && (rows.Imports || rows.Exports)) {
      net = new Array(n);
      for (i = 0; i < n; i++) {
        net[i] = ((rows.Imports || [])[i] || 0) + ((rows.Exports || [])[i] || 0);
      }
      var nd = "";
      for (i = 0; i < n; i++) {
        nd += (i ? " L" : "M") + r1(x(i)) + " " + r1(y(net[i])) +
              " L" + r1(x(i + 1)) + " " + r1(y(net[i]));
      }
      el("path", { d: nd, fill: "none", stroke: NETIMPORT,
                   "stroke-width": 1.5 * FS }, f.svg);
    }

    // Marginal cost, right axis.
    var pv = axis.map(function (s) {
      var k = yr + "|" + s.q + "|" + s.d + "|" + s.t, v = price[k];
      if (!minus) return v;
      var w = pref[k];
      return (v == null || w == null) ? null : v - w;
    });
    var pnum = pv.filter(function (v) { return v != null; });
    var pax = niceAxis(minus ? Math.min.apply(null, pnum.concat([0])) : 0,
                       Math.max.apply(null, pnum.concat([minus ? 0 : 1])));
    var py = function (v) {
      return m.t + f.ih * (pax.hi - v) / (pax.hi - pax.lo || 1);
    };
    var pd = "";
    pv.forEach(function (v, i) {
      if (v == null) return;
      pd += (pd ? " L" : "M") + r1(x(i) + bw / 2) + " " + r1(py(v));
    });
    el("path", { d: pd, fill: "none", stroke: "#c0682a",
                 "stroke-width": 1.1 * FS,
                 "stroke-dasharray": r1(3 * FS) + " " + r1(2 * FS),
                 "stroke-opacity": .85 }, f.svg);
    pax.ticks.forEach(function (v) {
      txt(f.svg, m.l + f.iw + 6, py(v) + 3.2 * FS, fmt(v, 0),
          { size: 8.5, fill: "#c0682a" });
    });
    txt(f.svg, m.l + f.iw + 6, m.t - 9, minus ? "Δ$/MWh" : "$/MWh",
        { size: 9, fill: "#c0682a", weight: 600 });

    // Season / day-type axis: seasons named on top, day types below with the
    // share of the year each one stands for.
    var seasons = [], days = [];
    axis.forEach(function (s, i) {
      var ls = seasons[seasons.length - 1];
      if (!ls || ls.q !== s.q) seasons.push({ q: s.q, a: i, b: i + 1 });
      else ls.b = i + 1;
      var ld = days[days.length - 1];
      if (!ld || ld.q !== s.q || ld.d !== s.d) days.push({ q: s.q, d: s.d, a: i, b: i + 1 });
      else ld.b = i + 1;
    });
    days.forEach(function (g, gi) {
      if (gi) {
        el("line", { x1: r1(x(g.a)), y1: m.t, x2: r1(x(g.a)), y2: m.t + f.ih,
                     stroke: "#c8d0dc", "stroke-width": .7,
                     "stroke-dasharray": "3 3", "stroke-opacity": .8 }, f.svg);
      }
      var hrs = 0;
      for (var t = 1; t <= 24; t++) {
        hrs += RD.hours[g.q + "|" + g.d + "|t" + (t < 10 ? "0" + t : t)] || 0;
      }
      // Half-width panels leave a day type under 20 px of axis: the share of
      // the year is the first thing to go, then the label itself.
      var gw = x(g.b) - x(g.a);
      if (gw >= 11 * FS) {
        txt(f.svg, x((g.a + g.b) / 2), m.t + f.ih + 13 * FS,
            gw >= 30 * FS ? g.d + " " + Math.round(hrs / 87.6) + "%" : g.d,
            { size: 7.5, anchor: "middle", fill: "#9aa4b2" });
      }
    });
    seasons.forEach(function (g, gi) {
      if (gi) {
        el("line", { x1: r1(x(g.a)), y1: m.t - 4, x2: r1(x(g.a)),
                     y2: m.t + f.ih + 18 * FS,
                     stroke: "#8a94a6", "stroke-width": 1.2 }, f.svg);
      }
      txt(f.svg, x((g.a + g.b) / 2), m.t + f.ih + 30 * FS, g.q,
          { size: 10, anchor: "middle", fill: "#1a2333", weight: 700 });
    });

    // One transparent strip drives the crosshair and the read-out.
    var cross = el("line", { x1: 0, y1: m.t, x2: 0, y2: m.t + f.ih,
                             stroke: "#1a2333", "stroke-width": .8,
                             "stroke-opacity": 0 }, f.svg);
    var grab = el("rect", { x: m.l, y: m.t, width: f.iw, height: f.ih,
                            fill: "transparent" }, f.svg);
    grab.addEventListener("mousemove", function (ev) {
      var box = f.svg.getBoundingClientRect();
      var rel = (ev.clientX - box.left) / box.width * W;
      var i = Math.max(0, Math.min(n - 1, Math.floor((rel - m.l) / bw)));
      cross.setAttribute("x1", r1(x(i) + bw / 2));
      cross.setAttribute("x2", r1(x(i) + bw / 2));
      cross.setAttribute("stroke-opacity", .35);
      var s = axis[i];
      var sg = function (v) { return (minus && v > 0 ? "+" : "") + fmt(v, 0); };
      var h = "<b>" + yr + " · " + s.q + " " + s.d + " " + s.t +
              (minus ? " · Δ vs " + label(minus) : "") + "</b><hr>";
      bars.slice().reverse().forEach(function (k) {
        var v = rows[k][i];
        if (Math.abs(v) > 0.05) {
          h += '<span class="k">' + lbl(k) + "</span> " + sg(v) + " MW<br>";
        }
      });
      if (rows.Demand && !minus) h += '<hr><span class="k">Demand</span> <b>' +
                            fmt(rows.Demand[i], 0) + " MW</b>";
      if (net) h += '<br><span class="k">Net imports</span> <b>' +
                    (net[i] > 0 ? "+" : "") + fmt(net[i], 0) + " MW</b>";
      if (pv[i] != null) h += '<br><span class="k">' +
                              (minus ? "Δ marginal cost" : "Marginal cost") +
                              "</span> <b>" + (minus && pv[i] > 0 ? "+" : "") +
                              fmt(pv[i], 1) + " $/MWh</b>";
      tipOn(h, ev);
    });
    grab.addEventListener("mouseleave", function () {
      cross.setAttribute("stroke-opacity", 0); tipOff();
    });

    legend(host, bars.map(function (k) {
      return { name: lbl(k), color: colorOf(k), hatch: !!HATCHED[k] };
    }).concat(minus
      ? [{ name: "Δ marginal cost (right)", color: "#c0682a", shape: "line" }]
      : [{ name: "Demand", color: "#8B0000", shape: "line" }].concat(
          NETTED ? []
                 : [{ name: "Net imports", color: NETIMPORT, shape: "line" }],
          [{ name: "Marginal cost ($/MWh, right)", color: "#c0682a",
             shape: "line" }])),
      null);
  }

  /* ============================================================== map chart
   * Zones drawn from the projected geojson, with a flow arrow per corridor.
   * Arrow thickness is net energy, colour is the utilisation of the link.
   */
  function mapChart(host, spec) {
    var yr = spec.year, scen = spec.scenario, focus = spec.focus || null;
    var iy = RD.years.indexOf(yr);
    var b0 = boxOf(spec, 620, 430), W = b0.W, H = b0.H;
    var m = { l: 8, r: 8, t: 8, b: 8 };

    // Corridors first: what is actually drawn decides how tight the frame can be.
    var cor = RD.corridors[scen];
    var flows = [];
    Object.keys(cor).forEach(function (k) {
      var c = cor[k];
      var ca = RD.geo.centroids[c.a], cb = RD.geo.centroids[c.b];
      if (!ca || !cb) return;
      if (focus && RD.zcmap[c.a] !== focus && RD.zcmap[c.b] !== focus) return;
      var net = (c.fwd[iy] || 0) - (c.rev[iy] || 0);
      var gross = (c.fwd[iy] || 0) + (c.rev[iy] || 0);
      // A pair with neither capacity nor energy is not a link, it is a hole in
      // the data; drawing it puts a dotted line across half the map.
      if (gross <= 0 && (c.ntc[iy] || 0) <= 0) return;
      flows.push({ k: k, c: c, ca: ca, cb: cb, net: net, gross: gross });
    });

    // Frame what is actually drawn — the focus country and its counterparties,
    // or every modelled zone — instead of the padded region-wide box.
    var box = RD.geo.box.slice();
    var xs = [], ys = [];
    Object.keys(RD.geo.zones).forEach(function (z) {
      if (focus && RD.zcmap[z] !== focus) return;
      RD.geo.zones[z].forEach(function (r) {
        r.forEach(function (p) { xs.push(p[0]); ys.push(p[1]); });
      });
    });
    flows.forEach(function (o) {
      xs.push(o.ca[0], o.cb[0]); ys.push(o.ca[1], o.cb[1]);
    });
    if (xs.length) {
      var bb = [Math.min.apply(null, xs), Math.min.apply(null, ys),
                Math.max.apply(null, xs), Math.max.apply(null, ys)];
      var px = Math.max(.8, (bb[2] - bb[0]) * .12),
          py = Math.max(.6, (bb[3] - bb[1]) * .12);
      box = [bb[0] - px, bb[1] - py, bb[2] + px, bb[3] + py];
    }

    var midLat = (box[1] + box[3]) / 2;
    var kx = Math.cos(midLat * Math.PI / 180);
    var spanX = (box[2] - box[0]) * kx, spanY = box[3] - box[1];
    // The region is a wide strip; giving it a square panel would be four fifths
    // empty, so the height follows the ground it has to cover.
    H = Math.max(170, Math.min(H * 1.3,
                               (W - m.l - m.r) * spanY / spanX + m.t + m.b));
    var f = frame(host, W, H, m, b0.side);
    var sc = Math.min(f.iw / spanX, f.ih / spanY);
    var ox = m.l + (f.iw - spanX * sc) / 2, oy = m.t + (f.ih - spanY * sc) / 2;
    var PX = function (lon) { return ox + (lon - box[0]) * kx * sc; };
    var PY = function (lat) { return oy + (box[3] - lat) * sc; };
    var inBox = function (c) {
      return c && c[0] >= box[0] && c[0] <= box[2] && c[1] >= box[1] && c[1] <= box[3];
    };

    function path(rings) {
      return rings.map(function (r) {
        return r.map(function (p, i) {
          return (i ? "L" : "M") + r1(PX(p[0])) + " " + r1(PY(p[1]));
        }).join(" ") + " Z";
      }).join(" ");
    }

    el("rect", { x: m.l, y: m.t, width: f.iw, height: f.ih, fill: "#f4f7fb",
                 stroke: "#e3e8f0" }, f.svg);

    // A zoomed frame cuts through neighbours, so all geometry is clipped to it.
    var clipId = "mclip" + (++HATCH_ID);
    var cp = el("clipPath", { id: clipId }, f.defs);
    el("rect", { x: m.l, y: m.t, width: f.iw, height: f.ih }, cp);
    var G = el("g", { "clip-path": "url(#" + clipId + ")" }, f.svg);

    Object.keys(RD.geo.ext).forEach(function (z) {
      el("path", { d: path(RD.geo.ext[z]), fill: "#e8ecf3", stroke: "#d3dae5",
                   "stroke-width": .7 }, G);
    });
    var zones = Object.keys(RD.geo.zones);
    zones.forEach(function (z, i) {
      var c = RD.zcmap[z];
      var on = !focus || c === focus;
      var p = el("path", {
        d: path(RD.geo.zones[z]),
        fill: on ? MAPBAND[i % MAPBAND.length] : "#dfe5ee",
        "fill-opacity": on ? .34 : .5,
        stroke: on ? "#1a2333" : "#c8d0dc", "stroke-width": on ? .8 : .5
      }, G);
      hover(p, "<b>" + z + "</b><br><span class='k'>" + c + "</span>");
    });

    // Arrows.  fwd is a -> b, rev is b -> a; the net decides which way it points.
    var gmax = Math.max.apply(null, flows.map(function (o) { return o.gross; }).concat([0.1]));

    // On the regional map the nine Turkish internal links would drown the
    // cross-border ones, so they are drawn first and thinner.
    flows.sort(function (p, q) {
      var f = function (o) { return RD.zcmap[o.c.a] === RD.zcmap[o.c.b] ? 0 : 1; };
      return f(p) - f(q);
    });

    var idleSeen = false;
    flows.forEach(function (o) {
      var from = o.net >= 0 ? o.ca : o.cb, to = o.net >= 0 ? o.cb : o.ca;
      var x1 = PX(from[0]), y1 = PY(from[1]), x2 = PX(to[0]), y2 = PY(to[1]);
      var util = o.c.util[iy] || 0;
      var w = o.gross > 0 ? 1.2 + 6 * Math.sqrt(o.gross / gmax) : 1;
      var domestic = !focus && RD.zcmap[o.c.a] === RD.zcmap[o.c.b];
      if (domestic) w *= .55;
      var col = o.gross <= 0 ? "#b7c0cc"
              : util >= .85 ? "#c0392b" : util >= .6 ? "#c0682a" : "#1b6ca8";
      if (o.gross <= 0) idleSeen = true;
      // Nudge the line off the centroids so both ends stay visible.
      var dx = x2 - x1, dy = y2 - y1, L = Math.sqrt(dx * dx + dy * dy) || 1;
      var ux = dx / L, uy = dy / L;
      var t = Math.min(16, L * .18);
      var sx = x1 + ux * t, sy = y1 + uy * t;      // tail
      var ex = x2 - ux * t, ey = y2 - uy * t;      // tip
      // The head is drawn, not markered: an SVG marker scales with the stroke,
      // so a 10 px flow line swallows its own arrowhead.
      var hs = o.gross > 0 ? Math.max(8, w * 2.1) : 0;
      var hw = hs * .48;
      var bx = ex - ux * hs, by = ey - uy * hs;
      var g = el("g", { "clip-path": "url(#" + clipId + ")" }, f.svg);
      el("line", { x1: r1(sx), y1: r1(sy),
                   x2: r1(hs ? bx + ux * hs * .35 : ex),
                   y2: r1(hs ? by + uy * hs * .35 : ey),
                   stroke: col, "stroke-width": r1(w),
                   "stroke-opacity": domestic ? .5 : .85,
                   "stroke-linecap": "round",
                   "stroke-dasharray": o.gross <= 0 ? "3 3" : null }, g);
      if (hs) {
        el("polygon", { points: [r1(ex) + "," + r1(ey),
                                 r1(bx - uy * hw) + "," + r1(by + ux * hw),
                                 r1(bx + uy * hw) + "," + r1(by - ux * hw)].join(" "),
                        fill: col, "fill-opacity": domestic ? .55 : .95 }, g);
      }
      el("line", { x1: r1(x1), y1: r1(y1), x2: r1(x2), y2: r1(y2),
                   stroke: "transparent", "stroke-width": Math.max(12, w + 8) }, g);
      var an = o.net >= 0 ? o.c.a : o.c.b, bn = o.net >= 0 ? o.c.b : o.c.a;
      var meta = RD.corridor_meta[o.k];
      var h = "<b>" + an + " &#8594; " + bn + "</b> &middot; " + yr + "<hr>" +
              '<span class="k">Net</span> ' + fmt(Math.abs(o.net)) + " TWh<br>" +
              '<span class="k">Gross</span> ' + fmt(o.gross) + " TWh<br>" +
              '<span class="k">Capacity</span> ' + fmt(o.c.ntc[iy], 0) + " MW<br>" +
              '<span class="k">Utilisation</span> ' + fmt(util * 100, 0) + " %";
      if (meta) {
        h += '<hr><span class="k">' + meta.project + "</span>";
        meta.lines.slice(0, 4).forEach(function (l) {
          h += "<br>" + l.from + "&#8211;" + l.to + (l.kv ? " " + l.kv + " kV" : "") +
               ' <span class="k">(' + l.status + ")</span>";
        });
      }
      hover(g, h);
    });

    // Names, drawn last so nothing covers them.  A regional map carries nine
    // Turkish zones and would be unreadable zone by zone, so it is labelled by
    // country; the zone name stays in the tooltip either way.
    if (focus) {
      zones.forEach(function (z) {
        var c = RD.geo.centroids[z];
        if (!inBox(c)) return;
        // The map is centred on the country the tab is about, so its own name
        // adds nothing; only the neighbours need naming.
        if (RD.zcmap[z] === focus) return;
        txt(f.svg, PX(c[0]), PY(c[1]) + 3, z,
            { size: 8.5, anchor: "middle", halo: 2.4,
              fill: "#77839a", weight: 500 });
      });
    } else {
      var byC = {};
      zones.forEach(function (z) {
        var c = RD.geo.centroids[z];
        if (!inBox(c)) return;
        var k = RD.zcmap[z] || z;
        (byC[k] = byC[k] || []).push(c);
      });
      Object.keys(byC).forEach(function (k) {
        var pts = byC[k];
        var mx = 0, my = 0;
        pts.forEach(function (c) { mx += c[0]; my += c[1]; });
        txt(f.svg, PX(mx / pts.length), PY(my / pts.length) + 3, k,
            { size: 10, anchor: "middle", halo: 3, fill: "#16202f", weight: 700 });
      });
    }
    Object.keys(RD.geo.ext).forEach(function (z) {
      var c = RD.geo.centroids[z];
      if (inBox(c)) txt(f.svg, PX(c[0]), PY(c[1]), z,
                        { size: 9, anchor: "middle", halo: 2.4, fill: "#8d97a6" });
    });
    // A partner that exists as a pseudo zone (the Iran swap) has an anchor but
    // no polygon, so nothing has named it yet: label the point the arrow lands.
    var named = {};
    flows.forEach(function (o) {
      [o.c.a, o.c.b].forEach(function (z) {
        if (RD.geo.zones[z] || RD.geo.ext[z] || named[z]) return;
        var c = RD.geo.centroids[z];
        if (!inBox(c)) return;
        named[z] = 1;
        txt(f.svg, PX(c[0]), PY(c[1]) + 3, z.replace(/_/g, " "),
            { size: 9, anchor: "middle", halo: 2.4, fill: "#8d97a6" });
      });
    });

    var lg = [{ name: "< 60 % used", color: "#1b6ca8" },
              { name: "60&#8211;85 %", color: "#c0682a" },
              { name: "&#8805; 85 % (congested)", color: "#c0392b" }];
    if (idleSeen) lg.push({ name: "capacity, no flow", color: "#b7c0cc" });
    legend(host, lg, "arrow = net direction, width = gross energy");
  }

  /* ========================================================= corridor chart
   * One bar group per corridor, corridors gathered under the project that
   * builds them, bars showing how the link evolves across the horizon.
   */
  function corridorChart(host, spec) {
    var scen = spec.scenario, cor = RD.corridors[scen];
    var shown = spec.years || ["2025", "2030", "2035", "2040"];
    var idx = shown.map(function (y) { return RD.years.indexOf(y); });

    var keys = Object.keys(cor).filter(function (k) {
      var c = cor[k];
      if (spec.focus && RD.zcmap[c.a] !== spec.focus && RD.zcmap[c.b] !== spec.focus) {
        return false;
      }
      return c.ntc.some(function (v) { return v > 0; }) ||
             c.fwd.some(function (v) { return v > 0.01; }) ||
             c.rev.some(function (v) { return v > 0.01; });
    });
    if (!keys.length) { host.innerHTML = '<p class="lede">no corridor</p>'; return; }

    var groups = {};
    keys.forEach(function (k) {
      var p = (RD.corridor_meta[k] || {}).project || "Existing network";
      (groups[p] = groups[p] || []).push(k);
    });
    var gnames = Object.keys(groups).sort(function (a, b) {
      if (a === "Existing network") return -1;
      if (b === "Existing network") return 1;
      return a.localeCompare(b);
    });

    var nCor = keys.length;
    var b0 = boxOf(spec, 980, 330), W = b0.W, H = b0.H;
    var m = { l: 48, r: 12, t: 30, b: 96 };
    var f = frame(host, W, H, m, b0.side);

    var hi = 0;
    keys.forEach(function (k) {
      idx.forEach(function (i) { hi = Math.max(hi, cor[k].ntc[i] || 0); });
    });
    var ax = niceAxis(0, hi);
    var y = gridY(f, ax, "MW");

    // With three corridors a full-width canvas would give 300 px bars, so the
    // slot is capped and the whole cluster centred instead.
    var slot = Math.min(f.iw / nCor, 26 * shown.length + 34);
    var bw = Math.min((slot * .64) / shown.length, 30);
    var X0 = m.l + Math.max(0, (f.iw - slot * nCor) / 2);
    var flowFill = hatch(f.defs, "#16202f");
    var col = 0;

    gnames.forEach(function (gn, gi) {
      var members = groups[gn];
      var gx0 = X0 + col * slot;
      members.forEach(function (k) {
        var c = cor[k];
        var x0 = X0 + col * slot + (slot - bw * shown.length) / 2;
        shown.forEach(function (yr, j) {
          var i = idx[j], v = c.ntc[i] || 0;
          var bx = x0 + j * bw;
          var rect = el("rect", {
            x: r1(bx + .5), y: r1(y(v)), width: r1(bw - 1),
            height: r1(Math.max(.8, y(0) - y(v))),
            fill: c.external ? "#7f8fa6" : MAPBAND[gi % MAPBAND.length],
            "fill-opacity": .85
          }, f.svg);
          // How much of that capacity actually carried energy.
          var util = Math.min(1, c.util[i] || 0);
          if (v > 0 && util > 0) {
            el("rect", { x: r1(bx + .5), y: r1(y(v)), width: r1(bw - 1),
                         height: r1(Math.max(.8, (y(0) - y(v)) * util)),
                         fill: flowFill, "fill-opacity": .5,
                         "pointer-events": "none" }, f.svg);
          }
          var meta = RD.corridor_meta[k];
          var h = "<b>" + c.a + " ↔ " + c.b + "</b> · " + yr + "<hr>" +
                  '<span class="k">Capacity</span> ' + fmt(v, 0) + " MW<br>" +
                  '<span class="k">' + c.a + " → " + c.b + "</span> " +
                  fmt(c.fwd[i]) + " TWh<br>" +
                  '<span class="k">' + c.b + " → " + c.a + "</span> " +
                  fmt(c.rev[i]) + " TWh<br>" +
                  '<span class="k">Utilisation</span> ' + fmt(util * 100, 0) + " %";
          if (meta) {
            h += '<hr><span class="k">' + meta.project + "</span>";
            meta.lines.slice(0, 5).forEach(function (l) {
              h += "<br>" + l.from + "–" + l.to + (l.kv ? " " + l.kv + " kV" : "") +
                   ' <span class="k">(' + l.status +
                   (l.entry ? ", " + l.entry : "") + ")</span>";
            });
          }
          hover(rect, h);
        });
        var cx = X0 + col * slot + slot / 2;
        txt(f.svg, cx, m.t + f.ih + 12, shortPair(c),
            { size: 9, anchor: "end", fill: "#55627a", rotate: -35 });
        col++;
      });
      // Project band above the corridors it owns.
      var gx1 = X0 + col * slot;
      el("line", { x1: r1(gx0 + 3), y1: m.t - 12, x2: r1(gx1 - 3), y2: m.t - 12,
                   stroke: gn === "Existing network" ? "#c8d0dc"
                                                     : MAPBAND[gi % MAPBAND.length],
                   "stroke-width": 2.5, "stroke-linecap": "round" }, f.svg);
      var cap = txt(f.svg, (gx0 + gx1) / 2, m.t - 17,
                    fitLabel(gn, gx1 - gx0 - 8, 8.5),
                    { size: 8.5, anchor: "middle", fill: "#55627a", weight: 600 });
      el("title", {}, cap).textContent = gn;   // full name on hover
      if (gi < gnames.length - 1) {
        el("line", { x1: r1(gx1), y1: m.t - 8, x2: r1(gx1), y2: m.t + f.ih + 4,
                     stroke: "#e3e8f0", "stroke-width": 1 }, f.svg);
      }
    });

    legend(host, [{ name: "loaded share of the year", color: "#16202f",
                    hatch: true },
                  { name: "external corridor", color: "#7f8fa6" }],
           shown.join(" · "));
  }

  /* Squeeze a label into the pixels it has: an acronym in brackets first,
     then a plain trim.  Character width is estimated, which is close enough
     for a band caption. */
  function fitLabel(s, px, size) {
    var wOf = function (t) { return t.length * size * .53; };
    if (wOf(s) <= px) return s;
    var m = /\(([^)]+)\)\s*$/.exec(s);
    if (m && wOf(m[1]) <= px) return m[1];
    var n = Math.max(3, Math.floor(px / (size * .53)) - 1);
    return s.slice(0, n) + "…";
  }

  function shortPair(c) {
    var s = function (z) { return z.length > 11 ? z.slice(0, 10) + "." : z; };
    return s(c.a) + "–" + s(c.b);
  }

  /* =============================================================== NDP plot
   * Model against the published national plan, on the four plan milestones.
   */
  function ndpChart(host, spec) {
    var plan = RD.plans.capacity_gw[spec.scope];
    if (!plan) { host.innerHTML = '<p class="lede">no published plan</p>'; return; }
    var pyears = RD.plans.years.map(String);
    var a = RD.annual[spec.scope][spec.scenario];

    // The plan reports Hydro and Solar; the model splits them further.
    var MERGE = { Reservoir: "Hydro", ROR: "Hydro", PSH: "Hydro", PV: "Solar",
                  "Onshore Wind": "Wind", "Offshore Wind": "Wind" };
    var model = {};
    Object.keys(a.cap).forEach(function (fu) {
      var k = MERGE[fu] || fu;
      var t = model[k] = model[k] || new Array(pyears.length).fill(0);
      pyears.forEach(function (y, j) {
        var i = RD.years.indexOf(y);
        if (i >= 0) t[j] += a.cap[fu][i] || 0;
      });
    });
    var cats = [];
    ["Nuclear", "Coal", "Gas", "Hydro", "Wind", "Solar", "Battery", "Biomass",
     "Geothermal"].forEach(function (k) {
      if (plan[k] || model[k]) cats.push(k);
    });

    var b0 = boxOf(spec, 620, 300), W = b0.W, H = b0.H;
    var m = { l: 44, r: 12, t: 22, b: 52 };
    var f = frame(host, W, H, m, b0.side);
    var hi = 0;
    pyears.forEach(function (_, j) {
      var p = 0, mo = 0;
      cats.forEach(function (k) {
        p += (plan[k] || [])[j] || 0; mo += (model[k] || [])[j] || 0;
      });
      hi = Math.max(hi, p, mo);
    });
    var ax = niceAxis(0, hi), y = gridY(f, ax, "GW");
    var slot = f.iw / pyears.length, pad = slot * .18, bw = (slot - 2 * pad) / 2;

    pyears.forEach(function (yr, j) {
      [["Plan", plan], ["Model", model]].forEach(function (pair, si) {
        var bx = m.l + j * slot + pad + si * bw, base = 0, rows = [], tot = 0;
        var any = cats.some(function (k) { return (pair[1][k] || [])[j]; });
        cats.forEach(function (k) {
          var v = (pair[1][k] || [])[j] || 0;
          if (v <= 0) return;
          var top = base + v;
          var rect = el("rect", { x: r1(bx + .5), y: r1(y(top)),
                                  width: r1(bw - 1), height: r1(y(base) - y(top)),
                                  fill: colorOf(k),
                                  "fill-opacity": si ? 1 : .55,
                                  stroke: si ? "none" : colorOf(k),
                                  "stroke-width": si ? 0 : .8,
                                  "stroke-dasharray": si ? null : "2 1.5" }, f.svg);
          rows.push({ k: k, v: v }); tot += v; base = top;
          hover(rect, "<b>" + yr + " · " + pair[0] + "</b><hr>" +
                      '<span class="k">' + k + "</span> " + fmt(v) + " GW");
        });
        if (!any) {
          txt(f.svg, bx + bw / 2, y(0) - 5, "n/a",
              { size: 7.5, anchor: "middle", fill: "#b7c0cc" });
        } else {
          txt(f.svg, bx + bw / 2, y(tot) - 4, fmt(tot, 1),
              { size: 8, anchor: "middle", fill: "#55627a", weight: 600 });
        }
        txt(f.svg, bx + bw / 2, m.t + f.ih + 24, pair[0],
            { size: 7.5, anchor: "middle", fill: "#aab3c2" });
      });
      txt(f.svg, m.l + j * slot + slot / 2, m.t + f.ih + 14, yr,
          { size: 9.5, anchor: "middle", fill: "#55627a", weight: 600 });
    });
    legend(host, cats.map(function (k) { return { name: k, color: colorOf(k) }; })
                     .concat([{ name: "left = plan (outlined), right = model",
                                color: "#c8d0dc" }]));
  }

  /* ---------------------------------------------------------------- boot */

  function seriesFor(scope, kind, scens) {
    var out = [], seen = {};
    scens.forEach(function (sc) {
      Object.keys(RD.annual[scope][sc][kind]).forEach(function (k) { seen[k] = 1; });
    });
    RD.fuel_order.forEach(function (k) {
      if (!seen[k]) return;
      var by = {};
      scens.forEach(function (sc) {
        by[sc] = RD.annual[scope][sc][kind][k] || new Array(RD.years.length).fill(0);
      });
      out.push({ key: k, color: colorOf(k), by: by });
    });
    return out;
  }

  function tradeSeries(scope, scens, sign) {
    var names = {}, ext = {};
    scens.forEach(function (sc) {
      Object.keys(RD.annual[scope][sc].trade).forEach(function (p) { names[p] = 1; });
      (RD.annual[scope][sc].ext_partners || []).forEach(function (p) { ext[p] = 1; });
    });
    return Object.keys(names).sort().map(function (p, i) {
      var by = {};
      scens.forEach(function (sc) {
        var t = RD.annual[scope][sc].trade[p];
        by[sc] = (t ? t[sign > 0 ? "imp" : "exp"] : new Array(RD.years.length).fill(0))
          .map(function (v) { return sign * v; });
      });
      return { key: (sign > 0 ? "Imports · " : "Exports · ") + p,
               name: p, color: MAPBAND[i % MAPBAND.length], hatch: true,
               external: !!ext[p], by: by };
    });
  }

  /* Net position per scenario: imports minus exports over every partner, the
     external ones (Russia, Romania, ...) included.  Above the axis the scope is
     a net buyer that year, below it a net seller. */
  // The bars run through blues, teals and sands, so the net line is kept off
  // that range entirely: near-black first, then crimson.
  var NETCOLOR = ["#101820", "#b3123f", "#7a3ba8", "#0f7a3d", "#c0682a"];

  function netLines(scope, scens) {
    return scens.map(function (sc, si) {
      var tr = RD.annual[scope][sc].trade, by = {};
      by[sc] = RD.years.map(function (_, i) {
        var v = 0;
        Object.keys(tr).forEach(function (p) {
          v += (tr[p].imp[i] || 0) - (tr[p].exp[i] || 0);
        });
        return v;
      });
      return { name: "Net · " + label(sc), color: NETCOLOR[si % NETCOLOR.length],
               scen: sc, by: by };
    });
  }

  /* Re-bases a stack spec on one scenario: every series becomes scenario minus
     reference, and what did not move drops out.  Bars split around zero on
     their own, so the same chart draws it. */
  function rebase(o, minus) {
    var scens = o.scenarios.filter(function (sc) { return sc !== minus; });
    function shift(list) {
      return (list || []).map(function (s) {
        var by = {}, c = {};
        scens.forEach(function (sc) {
          var a = s.by[sc] || [], b = s.by[minus] || [];
          by[sc] = RD.years.map(function (_, i) { return (a[i] || 0) - (b[i] || 0); });
        });
        for (var k in s) c[k] = s[k];
        c.by = by;
        return c;
      }).filter(function (s) {
        return scens.some(function (sc) {
          return s.by[sc].some(function (v) { return Math.abs(v) > 1e-6; });
        });
      });
    }
    o.series = shift(o.series);
    o.lines = shift((o.lines || []).filter(function (l) { return !l.scen; }));
    o.scenarios = scens;
    o.unit = "Δ " + o.unit;
    o.note = null;
    return o;
  }

  function draw(host) {
    var spec = JSON.parse(host.getAttribute("data-spec"));
    FS = document.body.classList.contains("deck") ? DECK : 1;
    var scens = spec.scenarios || RD.scenarios;
    if (spec.type === "stack") {
      var s = seriesFor(spec.scope, spec.kind, scens);
      if (spec.kind === "cap") {
        var by = {};
        scens.forEach(function (sc) {
          by[sc] = RD.years.map(function (_, i) {
            var tot = 0, cor = RD.corridors[sc];
            Object.keys(cor).forEach(function (k) {
              var c = cor[k];
              var inA = RD.zcmap[c.a] === spec.scope, inB = RD.zcmap[c.b] === spec.scope;
              if (spec.scope === "Region" ? (c.external) : (inA !== inB)) {
                tot += (c.ntc[i] || 0) / 1000;
              }
            });
            return tot;
          });
        });
        s.push({ key: "Interconnection capacity", color: "#7f8fa6", hatch: true,
                 excludeFromTotal: true, by: by });
      } else {
        s = s.concat(tradeSeries(spec.scope, scens, 1))
             .concat(tradeSeries(spec.scope, scens, -1));
      }
      var lines = [];
      if (spec.kind === "gen") {
        var dby = {};
        scens.forEach(function (sc) { dby[sc] = RD.annual[spec.scope][sc].demand; });
        lines.push({ name: "Demand", color: "#8B0000", by: dby });
      }
      var o = { scope: spec.scope, scenarios: scens, series: s,
                lines: lines, w: spec.w, h: spec.h, legendBelow: spec.legendBelow,
                unit: spec.kind === "cap" ? "GW" : "TWh",
                note: null };
      stackChart(host, spec.minus ? rebase(o, spec.minus) : o);
    } else if (spec.type === "trade") {
      var t = {
        scope: spec.scope, scenarios: scens, unit: "TWh",
        w: spec.w, h: spec.h, legendBelow: spec.legendBelow,
        series: tradeSeries(spec.scope, scens, 1)
                  .concat(tradeSeries(spec.scope, scens, -1))
                  .map(function (o) { o.hatch = o.external; return o; }),
        lines: netLines(spec.scope, scens),
        totalLabel: "Net"
      };
      stackChart(host, spec.minus ? rebase(t, spec.minus) : t);
    } else if (spec.type === "dispatch") {
      dispatchChart(host, spec);
    } else if (spec.type === "map") {
      mapChart(host, spec);
    } else if (spec.type === "corridor") {
      corridorChart(host, spec);
    } else if (spec.type === "ndp") {
      ndpChart(host, spec);
    }
  }

  function redrawAll() {
    [].forEach.call(document.querySelectorAll("[data-spec]"), function (h) {
      try {
        draw(h);
      } catch (e) {
        h.innerHTML = '<p class="lede">chart failed: ' + e.message + "</p>";
        if (window.console) console.error(h.getAttribute("data-spec"), e);
      }
    });
  }

  function boot() {
    // Deck mode: bigger type on a smaller plot, for charts that get pasted
    // two-per-slide and end up scaled to half their painted width.
    var deck = document.getElementById("deck");
    if (deck) {
      deck.addEventListener("click", function () {
        var on = document.body.classList.toggle("deck");
        deck.classList.toggle("on", on);
        redrawAll();
      });
    }

    // Tabs.
    var tabs = [].slice.call(document.querySelectorAll("nav.tabs button"));
    tabs.forEach(function (b) {
      b.addEventListener("click", function () {
        tabs.forEach(function (o) { o.classList.remove("on"); });
        b.classList.add("on");
        [].forEach.call(document.querySelectorAll("section.tab"), function (s) {
          s.classList.toggle("on", s.id === b.getAttribute("data-tab"));
        });
        window.scrollTo(0, 0);
      });
    });

    // Pickers rewrite the sibling chart's spec and redraw it.
    [].forEach.call(document.querySelectorAll("[data-picker]"), function (p) {
      var field = p.getAttribute("data-picker");
      var target = document.getElementById(p.getAttribute("data-target"));
      [].forEach.call(p.querySelectorAll("button"), function (b) {
        b.addEventListener("click", function () {
          [].forEach.call(p.querySelectorAll("button"), function (o) {
            o.classList.remove("on");
          });
          b.classList.add("on");
          var spec = JSON.parse(target.getAttribute("data-spec"));
          spec[field] = b.getAttribute("data-value");
          target.setAttribute("data-spec", JSON.stringify(spec));
          draw(target);
        });
      });
    });

    redrawAll();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
