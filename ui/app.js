(() => {
  const $ = (s) => document.querySelector(s);
  const cfg = window.PATHSNAP_CONFIG || {};
  const NS = "http://www.w3.org/2000/svg";
  const store = {
    get(k){ try { return localStorage.getItem(k); } catch { return null; } },
    set(k,v){ try { localStorage.setItem(k,v); } catch {} }
  };
  let apiBase = (new URLSearchParams(location.search).get("api") || store.get("ps_api") || cfg.API_BASE || "http://localhost:8000").replace(/\/+$/, "");
  let vb = null, home = null, busy = false;

  const svg = $("#map"), tripList = $("#tripList"), statusEl = $("#status"), dot = $("#dot");
  $("#apiUrl").value = apiBase;

  function setStatus(text, kind){
    statusEl.textContent = text;
    statusEl.className = "status" + (kind === "err" ? " err" : "");
    dot.className = "dot" + (kind === "ok" ? " ok" : kind === "err" ? " err" : "");
  }

  async function call(path, options = {}){
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), cfg.TIMEOUT_MS || 120000);
    const t0 = performance.now();
    try {
      const res = await fetch(apiBase + path, {
        ...options, signal: ctrl.signal, headers: { "Content-Type": "application/json" }
      });
      if (!res.ok) {
        let detail = "";
        try { detail = (await res.json()).detail || ""; } catch {}
        throw new Error("HTTP " + res.status + (detail ? ": " + detail : ""));
      }
      return { data: await res.json(), ms: performance.now() - t0 };
    } finally { clearTimeout(timer); }
  }

  function explain(err){
    if (err.name === "AbortError") return "The API took too long to answer. Check that the model is loaded on Laptop 1.";
    if (err instanceof TypeError) return "Cannot reach " + apiBase + ". Check that the API is running, both laptops are on the same network, the port is open in the firewall, and CORS is enabled on the API.";
    return err.message;
  }

  async function loadTrips(){
    setStatus("Connecting to " + apiBase + " …");
    tripList.replaceChildren();
    try {
      const { data } = await call("/trips");
      const raw = Array.isArray(data) ? data : (data.trips || []);
      const ids = raw.map(t => (t && typeof t === "object") ? (t.trip_id ?? t.id) : t);
      if (!ids.length) { setStatus("Connected, but the API returned no trips.", "err"); return; }
      ids.forEach(id => {
        const b = document.createElement("button");
        b.type = "button"; b.className = "trip"; b.textContent = "Trip " + id;
        b.setAttribute("aria-pressed", "false");
        b.addEventListener("click", () => runTrip(id, b));
        tripList.appendChild(b);
      });
      setStatus("Connected. " + ids.length + " trips available.", "ok");
    } catch (e) { setStatus(explain(e), "err"); }
  }

  async function runTrip(id, btn){
    if (busy) return;
    busy = true;
    tripList.querySelectorAll("button").forEach(b => { b.setAttribute("aria-pressed", String(b === btn)); b.disabled = true; });
    setStatus("Running prediction for trip " + id + " …");
    try {
      const { data, ms } = await call("/predict", { method: "POST", body: JSON.stringify({ trip_id: id }) });
      render(data, ms);
      setStatus("Trip " + id + " done in " + Math.round(ms) + " ms.", "ok");
    } catch (e) { setStatus(explain(e), "err"); }
    finally { busy = false; tripList.querySelectorAll("button").forEach(b => b.disabled = false); }
  }

  const toPt = (p) => Array.isArray(p) ? { lat: p[0], lon: p[1] } : { lat: p.lat, lon: p.lon };

  function render(d, ms){
    const gps = (d.gps || []).map(toPt);
    const pr = d.predicted_route;
    const arcs = (Array.isArray(pr) ? pr : (pr && pr.arcs) || []).map(a => (a.coords || []).map(toPt));
    const all = gps.concat(...arcs);
    if (!all.length) { $("#empty").textContent = "The API returned no coordinates for this trip."; $("#empty").style.display = "grid"; return; }
    $("#empty").style.display = "none";

    const lat0 = all.reduce((s, p) => s + p.lat, 0) / all.length;
    const k = Math.cos(lat0 * Math.PI / 180);
    const xy = (p) => [p.lon * k, -p.lat];
    const xs = all.map(p => xy(p)[0]), ys = all.map(p => xy(p)[1]);
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const w = Math.max(x1 - x0, 1e-6), h = Math.max(y1 - y0, 1e-6), pad = Math.max(w, h) * 0.08;
    home = { x: x0 - pad, y: y0 - pad, w: w + 2 * pad, h: h + 2 * pad };
    vb = { ...home }; applyView();

    const f = (v) => v.toFixed(7);
    const dots = (pts) => pts.map(p => { const [x, y] = xy(p); return "M" + f(x) + " " + f(y) + "h0"; }).join("");
    const lines = arcs.map(c => c.map((p, i) => { const [x, y] = xy(p); return (i ? "L" : "M") + f(x) + " " + f(y); }).join("")).join("");

    svg.replaceChildren();
    const mk = (parent, attrs) => { const el = document.createElementNS(NS, "path"); for (const a in attrs) el.setAttribute(a, attrs[a]); parent.appendChild(el); return el; };
    const style = { fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round", "vector-effect": "non-scaling-stroke" };
    const gRoute = document.createElementNS(NS, "g"), gGps = document.createElementNS(NS, "g");
    gRoute.id = "gRoute"; gGps.id = "gGps";
    svg.append(gRoute, gGps);
    mk(gRoute, { ...style, d: lines, stroke: "#E07B00", "stroke-width": 4, opacity: .9 });
    mk(gGps, { ...style, d: dots(gps), stroke: "#2F6FDE", "stroke-width": 5 });
    if (gps.length) {
      mk(svg, { ...style, d: dots([gps[0]]), stroke: "#fff", "stroke-width": 16 });
      mk(svg, { ...style, d: dots([gps[0]]), stroke: "#1B8A3B", "stroke-width": 12 });
      mk(svg, { ...style, d: dots([gps[gps.length - 1]]), stroke: "#fff", "stroke-width": 16 });
      mk(svg, { ...style, d: dots([gps[gps.length - 1]]), stroke: "#14212B", "stroke-width": 12 });
    }
    syncLayers();

    const items = [["GPS points", gps.length], ["Predicted segments", arcs.length], ["Round trip (ms)", Math.round(ms)]];
    if (d.latency_ms != null) items.push(["Server time (ms)", Math.round(d.latency_ms)]);
    for (const [key, val] of Object.entries(d.metrics || {})) if (typeof val !== "object") items.unshift([key.replace(/_/g, " "), val]);
    $("#metrics").replaceChildren(...items.map(([label, val]) => {
      const box = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = label;
      dd.textContent = (typeof val === "number" && !Number.isInteger(val)) ? val.toFixed(4) : String(val);
      box.append(dt, dd); return box;
    }));
  }

  function applyView(){ if (vb) svg.setAttribute("viewBox", [vb.x, vb.y, vb.w, vb.h].join(" ")); }
  function syncLayers(){
    const g = $("#gGps"), r = $("#gRoute");
    if (g) g.style.display = $("#lyGps").checked ? "" : "none";
    if (r) r.style.display = $("#lyRoute").checked ? "" : "none";
  }

  // zoom (wheel) and pan (drag)
  function geom(){
    const r = svg.getBoundingClientRect(), upp = Math.max(vb.w / r.width, vb.h / r.height);
    return { r, upp, ox: (r.width - vb.w / upp) / 2, oy: (r.height - vb.h / upp) / 2 };
  }
  svg.addEventListener("wheel", (e) => {
    if (!vb) return; e.preventDefault();
    const g = geom(), px = e.clientX - g.r.left, py = e.clientY - g.r.top;
    const ux = vb.x + (px - g.ox) * g.upp, uy = vb.y + (py - g.oy) * g.upp;
    const f = e.deltaY < 0 ? 0.85 : 1 / 0.85;
    vb.w *= f; vb.h *= f;
    vb.x = ux - (px - g.ox) * g.upp * f; vb.y = uy - (py - g.oy) * g.upp * f;
    applyView();
  }, { passive: false });
  let drag = null;
  svg.addEventListener("pointerdown", (e) => { if (vb) { drag = { x: e.clientX, y: e.clientY }; svg.setPointerCapture(e.pointerId); } });
  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const g = geom();
    vb.x -= (e.clientX - drag.x) * g.upp; vb.y -= (e.clientY - drag.y) * g.upp;
    drag = { x: e.clientX, y: e.clientY }; applyView();
  });
  svg.addEventListener("pointerup", () => { drag = null; });

  $("#lyGps").addEventListener("change", syncLayers);
  $("#lyRoute").addEventListener("change", syncLayers);
  $("#resetView").addEventListener("click", () => { if (home) { vb = { ...home }; applyView(); } });
  $("#apiForm").addEventListener("submit", (e) => {
    e.preventDefault();
    apiBase = $("#apiUrl").value.trim().replace(/\/+$/, "");
    store.set("ps_api", apiBase); loadTrips();
  });

  loadTrips();
})();
