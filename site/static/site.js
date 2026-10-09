// BucketWeights: theme toggle, the Teams table of contents, sortable tables, player filters.
(function () {
  const root = document.documentElement;

  // Paper / Blacktop
  const btn = document.getElementById("themeBtn");
  const isDark = () => (root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches);
  const label = () => { if (btn) btn.textContent = isDark() ? "Paper" : "Blacktop"; };
  label();
  if (btn) btn.addEventListener("click", () => {
    root.dataset.theme = isDark() ? "light" : "dark";
    try { localStorage.setItem("bw-theme", root.dataset.theme); } catch (e) {}
    label();
  });

  // 30 Teams
  const dlg = document.getElementById("teamsDlg");
  const open = document.getElementById("teamsBtn");
  if (dlg && open) {
    open.addEventListener("click", () => dlg.showModal());
    document.getElementById("teamsClose").addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  }

  // Sortable tables: click a heading; numbers use data-v when present
  const value = (td) => {
    const raw = td.dataset.v !== undefined ? td.dataset.v : td.textContent.trim().replace("−", "-").replace(/[^0-9.\-]/g, "");
    const n = parseFloat(raw);
    return isNaN(n) ? td.textContent.trim().toLowerCase() : n;
  };
  document.querySelectorAll("table.sortable").forEach((table) => {
    const heads = [...table.querySelectorAll("thead th")];
    heads.forEach((th, i) => {
      const b = th.querySelector("button");
      if (!b) return;
      b.addEventListener("click", () => {
        const asc = th.getAttribute("aria-sort") === "descending";
        heads.forEach((h) => h.removeAttribute("aria-sort"));
        th.setAttribute("aria-sort", asc ? "ascending" : "descending");
        const body = table.tBodies[0];
        const rows = [...body.rows];
        rows.sort((a, c) => {
          const x = value(a.cells[i]), y = value(c.cells[i]);
          const r = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
          return asc ? r : -r;
        });
        rows.forEach((r) => body.appendChild(r));
      });
    });
  });

  // Players page filters
  const pt = document.getElementById("playerTable");
  if (pt) {
    const team = document.getElementById("teamFilter"), name = document.getElementById("nameFilter"), reg = document.getElementById("regulars");
    const apply = () => {
      const t = team.value, q = name.value.trim().toLowerCase(), r = reg.checked;
      [...pt.tBodies[0].rows].forEach((row) => {
        const ok = (!t || row.dataset.team === t) && (!q || row.dataset.name.includes(q)) && (!r || parseFloat(row.dataset.fga) >= 8);
        row.hidden = !ok;
      });
    };
    [team, name, reg].forEach((el) => el.addEventListener("input", apply));
    apply();
  }
})();
