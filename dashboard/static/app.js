// RoboLab eval dashboard frontend.
// Single-page: sidebar tree (runs → tasks → episodes) + main pane that swaps
// between benchmark/run/task/episode views.

const state = {
  runs: [],            // RunMeta[]
  tasksByRun: {},      // run_id → TaskSummary[]
  episodesByKey: {},   // `${run_id}::${task}` → EpisodeRow[]
  selection: { view: 'overview' },
  expanded: { runs: new Set(), tasks: new Set() }, // task key = run_id::task
  compareSet: new Set(), // run_ids selected for cross-run comparison (Ctrl/Cmd-click)

  // Top-level route: 'home' | 'scenes' | 'tasks' | 'results'. Results and
  // Tasks each show their own sidebar body; Home / Scenes hide the
  // sidebar entirely. Default to 'home' so a fresh load lands on the
  // landing page rather than the Results sidebar.
  route: 'home',
  // Lazy-loaded catalog data — fetched only when the user navigates to the
  // corresponding tab for the first time.
  catalog: { tasks: {}, scenes: null, taskFolders: null },
  // Active task folders (multi-select). A Set of folder paths — preset names
  // ("benchmark", "general/easy") OR free-form relative / absolute paths.
  // Loaded lazily from localStorage on first render, seeded with the
  // constants.py default if storage is empty.
  tasksFolders: null,
  // Custom (non-preset) folders the user has added at any point. Persisted
  // so deactivating one doesn't make it disappear — it just becomes a grey
  // entry the user can re-activate later. Separate from `tasksFolders`
  // because preset folders are auto-discovered, while customs need to be
  // remembered across sessions.
  tasksKnownCustomFolders: null,
  // Per-folder validation result keyed by folder path. Populated by the
  // Tasks page on each render via /api/tasks/validate, used to drive chip
  // status indicators (counts, warnings, errors).
  tasksFolderStatus: {},
  // Task list sort + filter UI state — persisted across re-renders.
  tasksSort: { key: 'task_name', dir: 'asc' },
  // Multi-select filter: each kind holds a Set of selected bucket values. A
  // task passes when EVERY kind with a non-empty set has a matching value
  // (i.e. AND across kinds, OR within a kind).
  tasksFilter: { difficulty: new Set(), attribute: new Set() },
  // Per-table sort state, lazy-initialised. Key is whatever identifier the
  // table passes to sortableTh().
  sort: {},
};

// ---- helpers ----------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const fmtPct = (r) => `${(r * 100).toFixed(1)}%`;
const fmtSec = (s) => s ? `${s.toFixed(1)}s` : '—';
const fmtScore = (s) => (s == null || !Number.isFinite(s)) ? '—' : (s * 100).toFixed(1);

// Inline SVG dancing-robot mark (Ribble). Wrapped in a .ribble-host span so
// JS hover wiring has a reliable rectangular hit-area (the bare SVG is
// stroke-only; cursoring through empty interior wouldn't fire mouse events).
// el() doesn't know about the SVG namespace so we parse via a template
// element and pluck the first child node.
function ribbleIcon(size = 22) {
  const tmpl = document.createElement('template');
  tmpl.innerHTML = `<span class="ribble-host"><svg viewBox="0 0 24 24" class="ribble-icon" width="${size}" height="${size}" aria-hidden="true">
    <line x1="12" y1="2" x2="12" y2="5" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" />
    <circle cx="12" cy="2.2" r="1.05" fill="currentColor" />
    <rect x="6" y="5" width="12" height="9" rx="2.2" fill="none" stroke="currentColor" stroke-width="1.4" />
    <circle cx="9.3" cy="9.2" r="1.05" fill="currentColor" class="ribble-eyes-default" />
    <circle cx="14.7" cy="9.2" r="1.05" fill="currentColor" class="ribble-eyes-default" />
    <path d="M 8.4 9.7 Q 9.3 8.4 10.2 9.7" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" fill="none" class="ribble-eyes-happy" />
    <path d="M 13.8 9.7 Q 14.7 8.4 15.6 9.7" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" fill="none" class="ribble-eyes-happy" />
    <line x1="9" y1="12" x2="15" y2="12" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" class="ribble-mouth" />
    <path d="M 9 11.6 Q 12 13.6 15 11.6" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" fill="none" class="ribble-smile" />
    <rect x="7.5" y="15" width="9" height="6.5" rx="1.4" fill="none" stroke="currentColor" stroke-width="1.4" />
    <line x1="5" y1="17" x2="7.5" y2="17.5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" class="ribble-arm-l" />
    <line x1="19" y1="17" x2="16.5" y2="17.5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" class="ribble-arm-r" />
  </svg></span>`;
  const host = tmpl.content.firstElementChild;
  const svg = host.querySelector('.ribble-icon');
  wireRibble(svg, host);
  return host;
}

// ---- Ribble hover wiring ---------------------------------------------------
// JS path instead of CSS :hover: more reliable across browsers, and trivial
// to verify in DevTools (watch the `.ribble-happy` class flip on mouseenter).
function wireRibble(svg, trigger) {
  trigger = trigger || svg;
  if (trigger._ribbleWired) return;
  trigger._ribbleWired = true;
  trigger.addEventListener('mouseenter', () => {
    svg.classList.add('ribble-happy');
  });
  trigger.addEventListener('mouseleave', () => {
    svg.classList.remove('ribble-happy');
  });
}

// ---- generic sortable-table helpers --------------------------------------
//
// Usage:
//   const onChange = () => rerenderThisView();
//   th: sortableTh('overviewTasks', 'sr', 'SR%', { numeric: true, alignRight: true, onChange })
//   row data: sortRows(rows, 'overviewTasks', { task: r => r.task, sr: { get: r => r.rate, numeric: true }, ... })
//
// Default sort key + direction can be seeded with seedSort('overviewTasks', 'task', 'asc').

function _sortState(id) {
  if (!state.sort[id]) state.sort[id] = { key: null, dir: 'asc' };
  return state.sort[id];
}

function seedSort(id, key, dir = 'asc') {
  const s = _sortState(id);
  if (s.key == null) { s.key = key; s.dir = dir; }
}

function sortableTh(tableId, columnKey, label, opts = {}) {
  const ss = _sortState(tableId);
  const active = ss.key === columnKey;
  const arrow = active ? (ss.dir === 'asc' ? '▲' : '▼') : '';
  return el('th', {
    class: `px-3 py-2 sort-th ${active ? 'active' : ''} ${opts.alignRight ? 'text-right' : 'text-left'}`,
    style: opts.style || {},
    title: opts.title || '',
    onclick: () => {
      if (ss.key === columnKey) {
        ss.dir = ss.dir === 'asc' ? 'desc' : 'asc';
      } else {
        ss.key = columnKey;
        ss.dir = opts.defaultDir || 'asc';
      }
      if (opts.onChange) opts.onChange();
    },
  }, label, arrow ? el('span', { class: 'arrow' }, ` ${arrow}`) : null);
}

function sortRows(rows, tableId, accessors) {
  const ss = _sortState(tableId);
  if (!ss.key) return rows;
  const acc = accessors[ss.key];
  if (!acc) return rows;
  const get = acc.get || acc;
  const numeric = acc.numeric;
  const dir = ss.dir === 'desc' ? -1 : 1;
  return [...rows].sort((a, b) => {
    let av = get(a);
    let bv = get(b);
    if (numeric) {
      av = Number(av); bv = Number(bv);
      const am = !Number.isFinite(av);
      const bm = !Number.isFinite(bv);
      if (am && bm) return 0;
      if (am) return 1;             // missing always last
      if (bm) return -1;
      return (av - bv) * dir;
    }
    const as = av == null ? '' : String(av).toLowerCase();
    const bs = bv == null ? '' : String(bv).toLowerCase();
    if (!as && !bs) return 0;
    if (!as) return 1;
    if (!bs) return -1;
    return as < bs ? -1 * dir : 1 * dir;
  });
}
// Mean and across-task std of `timing.policy_inference_avg_ms`.
//
// The recorder writes this number per-task (every episode in a task shares the
// same value) so the meaningful variance is the spread across tasks, not the
// (zero) spread across episodes within one task. We dedupe by task first.
function inferenceStatsMs(eps) {
  const byTask = new Map();
  for (const e of eps) {
    const v = e.timing && e.timing.policy_inference_avg_ms;
    if (typeof v !== 'number' || !Number.isFinite(v)) continue;
    if (!byTask.has(e.task)) byTask.set(e.task, v);
  }
  const vals = [...byTask.values()];
  if (!vals.length) return null;
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const std = vals.length > 1
    ? Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (vals.length - 1))
    : null;
  return { mean, std, n: vals.length };
}
function inferenceMeanMs(eps) { const s = inferenceStatsMs(eps); return s ? s.mean : null; }
// Short display for a source dir: last segment + parent for context, e.g.
// "/path/to/external_drive/robolab_output" → "external_drive/robolab_output".
function _shortSource(path) {
  if (!path) return '';
  const parts = path.replace(/\/$/, '').split('/').filter(Boolean);
  return parts.slice(-2).join('/') || path;
}

// 95% Wilson CI for a binomial proportion k/n. Numerically close to the
// Beta(k+1, n-k+1) interval used by robolab.core.logging.results, no
// dependencies. Returns [lo, hi] in [0, 1].
function wilsonCI(k, n) {
  if (!n) return [0, 1];
  const z = 1.96;
  const p = k / n;
  const denom = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denom;
  const margin = (z * Math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

// 95% normal CI for a sample mean given mean, sample std, and sample size.
function normalCI(mean, std, n) {
  if (!Number.isFinite(mean) || !Number.isFinite(std) || !n || n < 2) return null;
  const z = 1.96;
  const half = z * std / Math.sqrt(n);
  return [mean - half, mean + half];
}

// Render an SR cell with CI: "29.7% [24.5-35.4] ±5.4".
// CI bounds prefer the server-supplied (Beta) values when present (`opts.lcb`,
// `opts.ucb`); otherwise fall back to client-side Wilson for breakdowns
// computed in JS.
function fmtSRCell(s, n, opts = {}) {
  const sr = n ? s / n : 0;
  const txt = `${(sr * 100).toFixed(1)}%`;
  if (!n) return el('span', {}, txt);
  const lo = opts.lcb != null ? opts.lcb : wilsonCI(s, n)[0];
  const hi = opts.ucb != null ? opts.ucb : wilsonCI(s, n)[1];
  const half = (hi - lo) / 2;
  // Compact display: ±half only. Full range goes in the title tooltip.
  const title = `${s}/${n} · 95% CI [${(lo * 100).toFixed(1)}–${(hi * 100).toFixed(1)}]`;
  return el('span', { class: 'val-with-ci', title },
    el('span', { class: 'val', style: opts.best ? { color: 'var(--success)', fontWeight: 600 } : {} }, txt),
    el('span', { class: 'ci' }, `±${(half * 100).toFixed(1)}`));
}

// Render a Score cell with Student-t CI. Prefers server-supplied bounds
// (`opts.lcb`, `opts.ucb`); falls back to a normal CI computed from (mean,
// std, n) when only raw stats are available.
function fmtScoreCell(mean, std, n, opts = {}) {
  if (mean == null || !Number.isFinite(mean)) return el('span', { style: { color: 'var(--text-2)' } }, '—');
  // Scores live in [0, 1] internally but display as 0–100 for readability.
  const valEl = el('span', { class: 'val', style: opts.best ? { color: 'var(--success)', fontWeight: 600 } : {} },
    (mean * 100).toFixed(1));
  let lo = opts.lcb, hi = opts.ucb;
  if (lo == null || hi == null) {
    const ci = normalCI(mean, std, n);
    if (ci) { lo = ci[0]; hi = ci[1]; }
  }
  if (lo == null || hi == null || !Number.isFinite(lo) || !Number.isFinite(hi)) {
    return el('span', { class: 'val-with-ci', title: n ? `n=${n}` : '' }, valEl);
  }
  const half = (hi - lo) / 2;
  const title = `n=${n} · 95% CI [${(lo * 100).toFixed(1)}–${(hi * 100).toFixed(1)}]`;
  return el('span', { class: 'val-with-ci', title }, valEl,
    el('span', { class: 'ci' }, `±${(half * 100).toFixed(1)}`));
}

// Default speed for every <video> rendered by the dashboard. playbackRate is a
// JS property (not an HTML attribute) and some browsers reset it on certain
// transitions, so we set it eagerly AND re-apply on loadedmetadata + play.
const DEFAULT_VIDEO_RATE = 2.0;
function applyDefaultPlaybackRate(video) {
  video.playbackRate = DEFAULT_VIDEO_RATE;
  const reapply = () => { video.playbackRate = DEFAULT_VIDEO_RATE; };
  video.addEventListener('loadedmetadata', reapply);
  video.addEventListener('play', reapply);
}

// Track a video's load state on its parent host. Adds `.is-loading` until
// the first frame is ready, swaps to `.is-failed` on error. The CSS pseudo
// element on `.video-host` displays the actual "Loading…" / "video failed"
// overlay. Mark `host` with `video-host` class via the caller.
function attachVideoLoading(video, host) {
  host.classList.add('is-loading');
  const done = () => {
    host.classList.remove('is-loading');
    host.classList.remove('is-failed');
  };
  const fail = () => {
    host.classList.remove('is-loading');
    host.classList.add('is-failed');
  };
  video.addEventListener('loadeddata', done, { once: true });
  video.addEventListener('canplay', done, { once: true });
  video.addEventListener('error', fail, { once: true });
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'style' && typeof v === 'object') {
      // CSS custom properties (--foo) must go through setProperty; direct
      // assignment via Object.assign silently no-ops.
      for (const [sk, sv] of Object.entries(v)) {
        if (sk.startsWith('--')) node.style.setProperty(sk, sv);
        else node.style[sk] = sv;
      }
    }
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return node;
}

function badge(success) {
  return el('span', { class: `chip ${success ? 'success' : 'fail'}` }, success ? 'success' : 'fail');
}

function chip(text) { return el('span', { class: 'chip' }, text); }

function bar(rate) {
  return el('div', { class: 'bar', style: { '--rate': `${(rate * 100).toFixed(1)}%` } });
}

// Combined SR + Score bar — solid green to SR, hatched stripe to Score, grey
// remainder to 100%. Each region carries its own title so hovering surfaces
// the underlying number. Score is clamped >= SR (the data invariant: score
// counts every successful episode plus partial-completion credit on failures).
function srScoreBar(sr, score, opts = {}) {
  if (sr == null || !Number.isFinite(sr)) sr = 0;
  if (score == null || !Number.isFinite(score)) score = sr;
  const srClamped = Math.max(0, Math.min(1, sr));
  const scoreClamped = Math.max(srClamped, Math.min(1, score));
  const srTitle = opts.srTitle || `SR ${fmtPct(srClamped)}`;
  const scoreTitle = opts.scoreTitle || `Score ${fmtScore(scoreClamped)}`;
  return el('div', { class: 'sr-score-bar', style: opts.style || {} },
    el('div', { class: 'stripe', title: scoreTitle, style: { width: `${(scoreClamped * 100).toFixed(1)}%` } }),
    el('div', { class: 'fill',   title: srTitle,    style: { width: `${(srClamped * 100).toFixed(1)}%` } }));
}

// ---- source registry (top of sidebar) --------------------------------------

async function loadSources() {
  const data = await fetchJSON('/api/sources');
  return data.sources || [];
}

async function renderSources() {
  const host = $('#sources');
  host.innerHTML = '';
  let sources = [];
  try { sources = await loadSources(); }
  catch (e) {
    host.appendChild(el('div', { class: 'text-xs', style: { color: 'var(--fail)' } }, `sources: ${e.message}`));
    return;
  }
  if (!sources.length) {
    host.appendChild(el('div', { class: 'text-xs', style: { color: 'var(--text-2)' } }, 'No result sources. Enter an output directory below.'));
    return;
  }
  for (const path of sources) {
    const row = el('div', { class: 'flex items-center gap-1 text-xs group' },
      el('span', { class: 'flex-1 truncate font-mono', title: path, style: { color: 'var(--text-1)' } }, path),
      el('button', {
        class: 'leading-none',
        title: 'remove',
        style: {
          color: 'var(--text-2)',
          fontSize: '18px',
          fontWeight: '600',
          width: '20px',
          height: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '4px',
          transition: 'background 80ms, color 80ms',
        },
        onmouseover: (e) => { e.currentTarget.style.color = 'var(--text-0)'; e.currentTarget.style.background = 'var(--bg-2)'; },
        onmouseout: (e) => { e.currentTarget.style.color = 'var(--text-2)'; e.currentTarget.style.background = ''; },
        onclick: async () => {
          try {
            await fetch(`/api/sources?path=${encodeURIComponent(path)}`, { method: 'DELETE' });
            await refreshAll();
          } catch (e) {
            alert(`failed to remove: ${e.message}`);
          }
        },
      }, '×'));
    host.appendChild(row);
  }
}

async function addSourceFromInput() {
  const input = $('#source-input');
  const path = (input.value || '').trim();
  if (!path) return;
  try {
    const r = await fetch('/api/sources', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      alert(`add failed: ${j.detail || r.status}`);
      return;
    }
    input.value = '';
    await refreshAll();
  } catch (e) {
    alert(`add failed: ${e.message}`);
  }
}

async function refreshAll() {
  // clear caches that depend on the source list
  state.tasksByRun = {};
  state.episodesByKey = {};
  state.compareSet.clear();
  await renderSources();
  state.runs = await fetchJSON('/api/runs');
  renderSidebar();
  renderCompareBar();
  if (state.selection.view === 'overview') selectOverview();
}

// ---- sidebar ----------------------------------------------------------------

async function ensureTasks(runId) {
  if (state.tasksByRun[runId]) return state.tasksByRun[runId];
  const data = await fetchJSON(`/api/runs/${encodeURIComponent(runId)}/summary`);
  state.tasksByRun[runId] = data.tasks;
  return data.tasks;
}

async function ensureEpisodes(runId, task) {
  const key = `${runId}::${task}`;
  if (state.episodesByKey[key]) return state.episodesByKey[key];
  const data = await fetchJSON(
    `/api/runs/${encodeURIComponent(runId)}/tasks/${encodeURIComponent(task)}/episodes`
  );
  state.episodesByKey[key] = data;
  return data;
}

function renderSidebar() {
  const root = $('#sidebar');
  root.innerHTML = '';
  if (!state.runs.length) {
    root.appendChild(el('div', { class: 'p-3 text-slate-500' }, 'No experiments found.'));
    return;
  }
  // Only label per-experiment source when there's more than one distinct
  // source dir — otherwise the label is redundant (matches the global one).
  const sourceSet = new Set(state.runs.map((r) => r.source).filter(Boolean));
  const showSource = sourceSet.size > 1;
  for (const run of state.runs) {
    const isExpanded = state.expanded.runs.has(run.run_id);
    const isSelected = state.selection.view === 'run' && state.selection.run_id === run.run_id;
    const isCompared = state.compareSet.has(run.run_id);
    const row = el('div', {
      class: `tree-row flex items-center gap-2 px-3 py-1.5 ${isSelected ? 'selected' : ''} ${isCompared ? 'compared' : ''}`,
      onclick: async (ev) => {
        // Ctrl/Cmd-click: toggle inclusion in compare set instead of navigating.
        if (ev.ctrlKey || ev.metaKey) {
          ev.preventDefault();
          if (state.compareSet.has(run.run_id)) state.compareSet.delete(run.run_id);
          else state.compareSet.add(run.run_id);
          renderSidebar();
          renderCompareBar();
          return;
        }
        if (state.expanded.runs.has(run.run_id)) state.expanded.runs.delete(run.run_id);
        else { state.expanded.runs.add(run.run_id); await ensureTasks(run.run_id); }
        selectRun(run.run_id);
        renderSidebar();
      },
    },
      el('input', {
        type: 'checkbox',
        checked: isCompared ? '' : false,
        class: 'shrink-0',
        title: 'include in comparison',
        onclick: (ev) => {
          ev.stopPropagation();
          if (ev.target.checked) state.compareSet.add(run.run_id);
          else state.compareSet.delete(run.run_id);
          renderSidebar();
          renderCompareBar();
        },
      }),
      el('span', { class: 'w-3', style: { color: 'var(--text-2)' } }, isExpanded ? '▾' : '▸'),
      el('div', { class: 'flex-1 min-w-0' },
        el('div', { class: 'font-medium truncate', title: run.run_id }, run.run_id),
        el('div', { class: 'text-xs', style: { color: 'var(--text-2)' } },
          run.policy ? `${run.policy} · ` : '',
          `${run.num_success}/${run.num_episodes} (${fmtPct(run.success_rate)})`),
        showSource && run.source
          ? el('div', {
              class: 'text-xs truncate font-mono',
              title: run.source,
              style: { color: 'var(--text-2)', opacity: 0.7, fontSize: '10px' },
            }, _shortSource(run.source))
          : null));
    root.appendChild(row);

    if (isExpanded) {
      const tasks = state.tasksByRun[run.run_id] || [];
      for (const t of tasks) {
        const taskKey = `${run.run_id}::${t.task}`;
        const taskExpanded = state.expanded.tasks.has(taskKey);
        const taskSelected = state.selection.view === 'task'
          && state.selection.run_id === run.run_id && state.selection.task === t.task;
        const trow = el('div', {
          class: `tree-row flex items-center gap-2 pl-8 pr-3 py-1 ${taskSelected ? 'selected' : ''}`,
          onclick: async (ev) => {
            ev.stopPropagation();
            if (state.expanded.tasks.has(taskKey)) state.expanded.tasks.delete(taskKey);
            else { state.expanded.tasks.add(taskKey); await ensureEpisodes(run.run_id, t.task); }
            selectTask(run.run_id, t.task);
            renderSidebar();
          },
        },
          el('span', { class: 'w-3 text-slate-400 text-xs' }, taskExpanded ? '▾' : '▸'),
          el('div', { class: 'flex-1 min-w-0' },
            el('div', { class: 'truncate text-sm', title: t.task }, t.task),
            el('div', { class: 'text-xs text-slate-400' },
              `${t.num_success}/${t.num_episodes} (${fmtPct(t.success_rate)})`)));
        root.appendChild(trow);

        if (taskExpanded) {
          const eps = state.episodesByKey[taskKey] || [];
          for (const ep of eps) {
            const epSelected = state.selection.view === 'episode'
              && state.selection.run_id === run.run_id
              && state.selection.task === t.task
              && state.selection.env_id === ep.env_id
              && state.selection.run_index === ep.run_index;
            const erow = el('div', {
              class: `tree-row flex items-center gap-2 pl-14 pr-3 py-1 text-xs ${epSelected ? 'selected' : ''}`,
              onclick: (ev) => {
                ev.stopPropagation();
                selectEpisode(run.run_id, t.task, ep.env_id, ep.run_index);
                renderSidebar();
              },
            },
              el('span', { class: `chip ${ep.success ? 'success' : 'fail'}`, style: { padding: '0 0.4rem' } },
                ep.success ? '✓' : '✗'),
              el('span', {}, `run ${ep.run_index} · env ${ep.env_id}`),
              el('span', { class: 'ml-auto text-slate-400' }, fmtSec(ep.duration)));
            root.appendChild(erow);
          }
        }
      }
    }
  }
}

// ---- compare bar / report ---------------------------------------------------

function renderCompareBar() {
  const bar = $('#compare-bar');
  bar.innerHTML = '';
  const n = state.compareSet.size;
  if (n < 1) return;

  const box = el('div', { class: 'compare-bar' },
    el('span', {}, n === 1
      ? '1 experiment selected — Ctrl/Cmd-click more to compare'
      : `${n} experiments selected`),
    el('div', { class: 'flex-1' }),
    el('button', {
      onclick: () => { selectAnalyze(); },
      disabled: n < 1 ? '' : false,
      style: n < 1 ? { opacity: 0.5, cursor: 'not-allowed' } : {},
      title: 'Pool episodes from selected experiments and break down by attribute / instruction type',
    }, 'Combine →'),
    el('button', {
      onclick: () => { selectCompare(); },
      disabled: n < 2 ? '' : false,
      style: n < 2 ? { opacity: 0.5, cursor: 'not-allowed' } : {},
      title: 'Side-by-side per-task success-rate matrix across selected experiments',
    }, 'Compare →'),
    el('button', {
      class: 'ghost',
      onclick: () => { state.compareSet.clear(); renderSidebar(); renderCompareBar(); },
    }, 'Clear'));
  bar.appendChild(box);
}

function selectCompare() {
  if (state.compareSet.size < 2) return;
  state.selection = { view: 'compare' };
  renderCompareReport();
}

function selectAnalyze() {
  if (state.compareSet.size < 1) return;
  state.selection = { view: 'analyze' };
  renderAnalyzeReport();
}

// Drill-down from a per-task matrix cell: same task across N experiments,
// with each experiment's videos listed for side-by-side comparison.
function selectTaskCompare(task, runIds, from) {
  if (!runIds || !runIds.length) return;
  state.selection = { view: 'taskcompare', task, run_ids: [...runIds], from };
  renderTaskCompareReport();
}

async function renderTaskCompareReport() {
  const { task, run_ids, from } = state.selection;
  const backLabel = from === 'combine' ? 'Combine' : 'Compare';
  const backSelect = from === 'combine' ? selectAnalyze : selectCompare;
  setBreadcrumb(
    el('a', { class: 'hover:underline cursor-pointer', onclick: selectOverview }, 'Overview'),
    el('a', { class: 'hover:underline cursor-pointer', onclick: backSelect }, `${backLabel} ${run_ids.length} experiments`),
    task,
  );
  const pane = $('#pane');
  pane.innerHTML = '';
  pane.appendChild(el('div', { class: 'mb-4' },
    el('h2', { class: 'text-xl font-semibold mb-1 truncate', title: task }, task),
    el('p', { class: 'text-sm', style: { color: 'var(--text-2)' } },
      `Side-by-side across ${run_ids.length} experiments`)));

  // Fetch episodes per experiment (cached).
  const epsByRun = {};
  const loadingMsg = el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'Loading episodes…');
  pane.appendChild(loadingMsg);
  try {
    await Promise.all(run_ids.map(async (id) => {
      const k = `${id}::${task}`;
      if (state.episodesByKey[k]) { epsByRun[id] = state.episodesByKey[k]; return; }
      try {
        const eps = await fetchJSON(`/api/runs/${encodeURIComponent(id)}/tasks/${encodeURIComponent(task)}/episodes`);
        state.episodesByKey[k] = eps;
        epsByRun[id] = eps;
      } catch (e) {
        epsByRun[id] = [];   // task missing from that experiment — render empty section
      }
    }));
  } catch (e) {
    loadingMsg.textContent = `Error loading: ${e.message}`;
    loadingMsg.style.color = 'var(--fail)';
    return;
  }
  loadingMsg.remove();

  for (const id of run_ids) {
    const eps = epsByRun[id] || [];
    const run = state.runs.find((r) => r.run_id === id);
    const succ = eps.filter((e) => e.success).length;
    const n = eps.length;
    const scores = eps.map((e) => e.score).filter((s) => s != null);
    const mean = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
    const std = scores.length > 1
      ? Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / (scores.length - 1))
      : null;

    const sectionHeader = el('div', {
      class: 'flex items-center gap-3 mb-3 pb-3 flex-wrap',
      style: { borderBottom: '1px solid var(--border)' },
    },
      el('div', { class: 'font-semibold truncate', title: id, style: { maxWidth: '420px' } }, id),
      run && run.policy ? el('span', { class: 'chip', style: { fontSize: '11px' } }, run.policy) : null,
      el('div', { class: 'flex-1' }),
      el('div', { class: 'flex items-center gap-4' },
        el('div', { class: 'metric-block text-sm' },
          el('div', { class: 'text-xs', style: { color: 'var(--text-2)' } }, 'SR'),
          fmtSRCell(succ, n)),
        el('div', { class: 'metric-block text-sm' },
          el('div', { class: 'text-xs', style: { color: 'var(--text-2)' } }, 'Score'),
          fmtScoreCell(mean, std, scores.length))));

    const grid = el('div', { class: 'grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-3' });
    if (!eps.length) {
      grid.appendChild(el('div', { class: 'col-span-full text-sm', style: { color: 'var(--text-2)' } },
        'No episodes for this task in this experiment.'));
    } else {
      for (const ep of eps) grid.appendChild(buildEpisodeCard(id, task, ep));
    }

    pane.appendChild(el('div', { class: 'panel p-4 mb-5' }, sectionHeader, grid));
  }
}

async function renderAnalyzeReport() {
  const runIds = [...state.compareSet];
  setBreadcrumb(
    el('a', { class: 'hover:underline cursor-pointer', onclick: selectOverview }, 'Overview'),
    `Combine ${runIds.length} experiment${runIds.length === 1 ? '' : 's'}`,
  );
  const pane = $('#pane');
  pane.innerHTML = '';
  pane.appendChild(el('div', { class: 'mb-4' },
    el('h2', { class: 'text-xl font-semibold mb-1' },
      `Combining ${runIds.length} experiment${runIds.length === 1 ? '' : 's'}`),
    el('div', { class: 'flex flex-wrap gap-2 mt-2' },
      ...runIds.map((id) => {
        const run = state.runs.find((r) => r.run_id === id);
        const label = run ? `${id}${run.policy ? ` · ${run.policy}` : ''}` : id;
        return el('span', { class: 'chip accent', title: id, style: { fontFamily: 'ui-monospace, monospace' } }, label);
      }))));

  const loadingMsg = el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'Loading episodes from all experiments…');
  pane.appendChild(loadingMsg);

  // Fetch summaries + episodes upfront (same as Compare). We keep both shapes:
  //   - perRun  : [{id, tasks: TaskSummary[]}] → drives the per-task matrix
  //   - epsByRun: {id: EpisodeRow[]}            → drives the per-experiment / attribute matrices
  //   - allEps  : pooled list for the top stat cards
  let perRun;
  const epsByRun = {};
  let allEps = [];
  try {
    perRun = await Promise.all(runIds.map(async (id) => {
      const data = await fetchJSON(`/api/runs/${encodeURIComponent(id)}/summary`);
      state.tasksByRun[id] = data.tasks;
      return { id, tasks: data.tasks };
    }));
    for (const { id, tasks } of perRun) {
      const lists = await Promise.all(tasks.map(async (t) => {
        const k = `${id}::${t.task}`;
        if (state.episodesByKey[k]) return state.episodesByKey[k];
        const eps = await fetchJSON(`/api/runs/${encodeURIComponent(id)}/tasks/${encodeURIComponent(t.task)}/episodes`);
        state.episodesByKey[k] = eps;
        return eps;
      }));
      epsByRun[id] = lists.flat();
      for (const ep of epsByRun[id]) allEps.push({ ...ep, run_id: id });
    }
  } catch (e) {
    loadingMsg.textContent = `Error loading: ${e.message}`;
    loadingMsg.style.color = 'var(--fail)';
    return;
  }
  loadingMsg.remove();

  if (!allEps.length) {
    pane.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'No episodes in the selected experiments.'));
    return;
  }

  // Top: 3 pooled stat cards. Below uses the same matrix style as Compare
  // but with a single "Combined" column instead of per-experiment columns.
  const total = allEps.length;
  const succ = allEps.filter((e) => e.success).length;
  const scores = allEps.map((e) => e.score).filter((s) => s != null);
  const meanScore = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
  const srBracket = total
    ? (() => { const [lo, hi] = wilsonCI(succ, total); return `[${(lo*100).toFixed(1)}–${(hi*100).toFixed(1)}] ±${((hi-lo)/2*100).toFixed(1)}`; })()
    : null;
  let scoreBracket = null;
  if (scores.length > 1) {
    const m = meanScore;
    const std = Math.sqrt(scores.reduce((a, b) => a + (b - m) ** 2, 0) / (scores.length - 1));
    const [lo, hi] = normalCI(m, std, scores.length);
    scoreBracket = `[${(lo*100).toFixed(1)}–${(hi*100).toFixed(1)}] ±${((hi-lo)/2*100).toFixed(1)}`;
  }
  pane.appendChild(el('div', { class: 'grid grid-cols-2 md:grid-cols-3 gap-3 mb-6' },
    statCard('Episodes', total.toLocaleString()),
    statCard('SR%', fmtPct(total ? succ / total : 0), srBracket || `${succ}/${total}`),
    statCard('Score',
      fmtScore(meanScore),
      scoreBracket || (scores.length ? `mean over ${scores.length}/${total} episodes` : 'not recorded'))));

  // Collapse N experiments into one "Combined" pseudo-experiment so the matrix
  // helpers can be reused as-is. Each unique task gets a combined TaskSummary
  // pooling all selected experiments; episodes are pooled into one big list.
  const combinedId = 'Combined';
  const combinedTasks = buildCombinedTaskSummaries(perRun);
  const combinedPerRun = [{ id: combinedId, tasks: combinedTasks }];
  const combinedEpsByRun = { [combinedId]: allEps };
  renderCrossExperimentMatrices(pane, [combinedId], combinedPerRun, combinedEpsByRun);
}

// Pool per-experiment TaskSummary[] into one combined TaskSummary[] keyed by
// task name. Score mean uses an episode-weighted average; score std uses the
// law of total variance (within + between).
function buildCombinedTaskSummaries(perRun) {
  const byTask = new Map();
  for (const { tasks } of perRun) {
    for (const t of tasks) {
      let b = byTask.get(t.task);
      if (!b) {
        b = { task: t.task, num_episodes: 0, num_success: 0,
              score_means: [], score_n: 0,
              dur_sum: 0, instruction: t.instruction };
        byTask.set(t.task, b);
      }
      b.num_episodes += t.num_episodes;
      b.num_success += t.num_success;
      b.dur_sum += (t.mean_duration || 0) * t.num_episodes;
      if (t.mean_score != null && t.score_n) {
        b.score_means.push({ mean: t.mean_score, n: t.score_n, std: t.score_std });
        b.score_n += t.score_n;
      }
      b.instruction ||= t.instruction;
    }
  }
  const out = [];
  for (const b of byTask.values()) {
    const n = b.num_episodes;
    let mean_score = null, score_std = null;
    if (b.score_n) {
      const total = b.score_means.reduce((a, x) => a + x.n, 0);
      mean_score = b.score_means.reduce((a, x) => a + x.mean * x.n, 0) / total;
      // pooled variance via law of total variance
      const within = b.score_means.reduce((a, x) => a + x.n * ((x.std || 0) ** 2), 0) / total;
      const between = b.score_means.reduce((a, x) => a + x.n * (x.mean - mean_score) ** 2, 0) / total;
      score_std = total > 1 ? Math.sqrt(within + between) : null;
    }
    out.push({
      task: b.task, num_episodes: n, num_success: b.num_success,
      success_rate: n ? b.num_success / n : 0,
      mean_score, score_std, score_n: b.score_n,
      mean_duration: n ? b.dur_sum / n : 0,
      instruction: b.instruction,
    });
  }
  return out.sort((a, b) => a.task.localeCompare(b.task));
}

// Shared analytics block: stat cards + trajectory metrics + breakdowns.
// Reused by the multi-run Analyze report and the single-run view.
function renderEpisodeAnalytics(host, eps, opts = {}) {
  const showPolicy = opts.showPolicy !== false;
  const showTaskBreakdown = opts.showTaskBreakdown !== false;

  const total = eps.length;
  const succ = eps.filter((e) => e.success).length;
  const scores = eps.map((e) => e.score).filter((s) => s != null);
  const meanScore = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;

  const srBracket = total
    ? (() => { const [lo, hi] = wilsonCI(succ, total); return `[${(lo*100).toFixed(1)}–${(hi*100).toFixed(1)}] ±${((hi-lo)/2*100).toFixed(1)}`; })()
    : null;
  let scoreBracket = null;
  if (scores.length > 1) {
    const m = meanScore;
    const std = Math.sqrt(scores.reduce((a, b) => a + (b - m) ** 2, 0) / (scores.length - 1));
    const [lo, hi] = normalCI(m, std, scores.length);
    scoreBracket = `[${(lo*100).toFixed(1)}–${(hi*100).toFixed(1)}] ±${((hi-lo)/2*100).toFixed(1)}`;
  }
  host.appendChild(el('div', { class: 'grid grid-cols-2 md:grid-cols-3 gap-3 mb-6' },
    statCard('Episodes', total.toLocaleString()),
    statCard('SR%', fmtPct(total ? succ / total : 0), srBracket || `${succ}/${total}`),
    statCard('Score',
      fmtScore(meanScore),
      scoreBracket || (scores.length ? `mean over ${scores.length}/${total} episodes` : 'not recorded'))));

  host.appendChild(el('h3', { class: 'text-sm font-semibold uppercase tracking-wider mb-2',
                              style: { color: 'var(--text-2)' } }, 'Trajectory metrics'));
  host.appendChild(buildMetricsPanel(eps));

  const breakdowns = el('div', { class: 'grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6' });
  breakdowns.appendChild(buildBreakdown('By attribute', bucketByMulti(eps, (e) => e.attributes || [])));
  breakdowns.appendChild(buildBreakdown('By instruction type', bucketBy(eps, (e) => e.instruction_type || '(none)')));
  if (showPolicy) {
    breakdowns.appendChild(buildBreakdown('By policy', bucketBy(eps, (e) => {
      const r = state.runs.find((rr) => rr.run_id === e.run_id);
      return (r && r.policy) || '(unknown)';
    })));
  }
  if (showTaskBreakdown) {
    const byTask = bucketBy(eps, (e) => e.task);
    const topTasks = Object.entries(byTask)
      .sort((a, b) => b[1].n - a[1].n)
      .slice(0, 20)
      .reduce((o, [k, v]) => { o[k] = v; return o; }, {});
    breakdowns.appendChild(buildBreakdown('By task (top 20 by count)', topTasks));
  }
  host.appendChild(breakdowns);
}

function statCard(label, value, sub) {
  return el('div', { class: 'panel p-3' },
    el('div', { class: 'text-xs uppercase tracking-wider mb-1', style: { color: 'var(--text-2)' } }, label),
    el('div', { class: 'text-2xl font-semibold' }, value),
    sub ? el('div', { class: 'text-xs', style: { color: 'var(--text-2)' } }, sub) : null);
}

// Metric set sourced from robolab.core.logging.results trajectory metrics.
// Order here drives display order. Tuples: [field, label, formatter]
const TRAJECTORY_METRICS = [
  ['ee_path_length',    'EE path length',        (v) => `${v.toFixed(1)} m`],
  ['ee_speed_mean',     'EE speed (mean)',       (v) => `${(v * 100).toFixed(1)} cm/s`],
  ['ee_sparc',          'EE SPARC',              (v) => v.toFixed(1)],
  ['ee_isj',            'EE jerk (ISJ)',         (v) => v.toFixed(1)],
];

function metricStats(eps, field) {
  const vals = [];
  for (const e of eps) {
    const v = e.metrics && e.metrics[field];
    if (typeof v === 'number' && Number.isFinite(v)) vals.push(v);
  }
  if (!vals.length) return null;
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
  const std = Math.sqrt(variance);
  return { mean, std, n: vals.length };
}

function buildMetricsPanel(eps) {
  // Split by success/fail for side-by-side comparison (interpretive context
  // for SPARC etc. is meaningless without it — failed eps often look "smoother"
  // because they spend more time stationary).
  const succEps = eps.filter((e) => e.success);
  const failEps = eps.filter((e) => !e.success);

  const rows = TRAJECTORY_METRICS.map(([field, label, fmt]) => {
    const all = metricStats(eps, field);
    if (!all) return null;
    const succ = metricStats(succEps, field);
    const fail = metricStats(failEps, field);
    const cell = (s) => s
      ? el('span', { title: `n=${s.n}` },
          fmt(s.mean),
          el('span', { class: 'text-xs ml-1', style: { color: 'var(--text-2)' } },
            ` ± ${fmt(s.std).replace(/\s*(cm\/s|m\/s|m)$/, '')}`))
      : el('span', { style: { color: 'var(--text-2)' } }, '—');
    return el('tr', {},
      el('td', { class: 'label-col', title: label }, label),
      el('td', { class: 'cell-num' }, cell(all)),
      el('td', { class: 'cell-num', style: { color: 'var(--success)' } }, cell(succ)),
      el('td', { class: 'cell-num', style: { color: 'var(--fail)' } }, cell(fail)));
  }).filter(Boolean);

  if (!rows.length) {
    return el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } },
      'No trajectory metrics recorded for these episodes.');
  }

  return el('table', { class: 'matrix panel w-full overflow-hidden mb-6' },
    el('thead', {}, el('tr', {},
      el('th', { class: 'label-col' }, 'Metric'),
      el('th', { class: 'cell-num' }, 'All'),
      el('th', { class: 'cell-num', style: { color: 'var(--success)' } }, 'Success'),
      el('th', { class: 'cell-num', style: { color: 'var(--fail)' } }, 'Fail'))),
    el('tbody', {}, ...rows));
}

function _addToBucket(buckets, key, ep) {
  if (!buckets[key]) buckets[key] = { n: 0, s: 0, scoreSum: 0, scoreSqSum: 0, scoreN: 0 };
  const b = buckets[key];
  b.n += 1;
  if (ep.success) b.s += 1;
  if (typeof ep.score === 'number' && Number.isFinite(ep.score)) {
    b.scoreSum += ep.score;
    b.scoreSqSum += ep.score * ep.score;
    b.scoreN += 1;
  }
}

function bucketBy(eps, key) {
  const out = {};
  for (const e of eps) _addToBucket(out, key(e), e);
  return out;
}

function bucketByMulti(eps, keys) {
  const out = {};
  for (const e of eps) {
    const ks = keys(e);
    if (!ks.length) { _addToBucket(out, '(none)', e); continue; }
    for (const k of ks) _addToBucket(out, k, e);
  }
  return out;
}

function buildBreakdown(title, buckets) {
  const entries = Object.entries(buckets).sort((a, b) => b[1].n - a[1].n);
  const card = el('div', { class: 'panel overflow-hidden' },
    el('div', { class: 'px-3 py-2 text-xs uppercase tracking-wider',
                style: { color: 'var(--text-2)', borderBottom: '1px solid var(--border)' } }, title),
    el('table', { class: 'matrix' },
      el('thead', {}, el('tr', {},
        el('th', { class: 'label-col' }, ''),
        el('th', { class: 'cell-num', style: { color: 'var(--text-2)', fontWeight: 400 } }, 'N'),
        el('th', { style: { color: 'var(--text-2)', fontWeight: 400 } }, ''),
        el('th', { class: 'cell-num', style: { color: 'var(--text-2)', fontWeight: 400 } }, 'SR%'),
        el('th', { class: 'cell-num', style: { color: 'var(--text-2)', fontWeight: 400 } }, 'Score'))),
      el('tbody', {},
        ...entries.map(([k, v]) => {
          const score = v.scoreN ? v.scoreSum / v.scoreN : null;
          const scoreStd = v.scoreN > 1 ? Math.sqrt(v.scoreSqSum / v.scoreN - score * score) : null;
          const sr = v.n ? v.s / v.n : 0;
          return el('tr', {},
            el('td', { class: 'label-col', title: k }, k),
            el('td', { class: 'cell-num', style: { color: 'var(--text-2)' } }, `${v.s}/${v.n}`),
            el('td', { style: { width: '120px', padding: '0 0.5rem' } },
              srScoreBar(sr, score)),
            el('td', { class: 'cell-num' }, fmtSRCell(v.s, v.n)),
            el('td', { class: 'cell-num' }, fmtScoreCell(score, scoreStd, v.scoreN)));
        }))));
  return card;
}

async function renderCompareReport() {
  const runIds = [...state.compareSet];
  setBreadcrumb(
    el('a', { class: 'hover:underline cursor-pointer', onclick: selectOverview }, 'Overview'),
    `Compare ${runIds.length} experiments`,
  );
  const pane = $('#pane');
  pane.innerHTML = '';
  pane.appendChild(el('div', { class: 'mb-4' },
    el('h2', { class: 'text-xl font-semibold mb-1' }, `Comparing ${runIds.length} experiments`),
    el('div', { class: 'flex flex-wrap gap-2 mt-2' },
      ...runIds.map((id) => {
        const run = state.runs.find((r) => r.run_id === id);
        const label = run ? `${id}${run.policy ? ` · ${run.policy}` : ''}` : id;
        return el('span', { class: 'chip accent', title: id, style: { fontFamily: 'ui-monospace, monospace' } }, label);
      }))));

  const loadingMsg = el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'Loading task data + episodes…');
  pane.appendChild(loadingMsg);

  // Fetch each run's task summary, then every task's episode list. We need
  // episode-level data for the per-experiment / attribute / instruction-type
  // sections that render BEFORE the per-task matrix, so do both up front.
  let perRun;
  const epsByRun = {};
  try {
    perRun = await Promise.all(runIds.map(async (id) => {
      const data = await fetchJSON(`/api/runs/${encodeURIComponent(id)}/summary`);
      state.tasksByRun[id] = data.tasks;
      return { id, tasks: data.tasks };
    }));
    for (const { id, tasks } of perRun) {
      const lists = await Promise.all(tasks.map(async (t) => {
        const k = `${id}::${t.task}`;
        if (state.episodesByKey[k]) return state.episodesByKey[k];
        const eps = await fetchJSON(`/api/runs/${encodeURIComponent(id)}/tasks/${encodeURIComponent(t.task)}/episodes`);
        state.episodesByKey[k] = eps;
        return eps;
      }));
      epsByRun[id] = lists.flat();
    }
  } catch (e) {
    loadingMsg.textContent = `Error loading: ${e.message}`;
    loadingMsg.style.color = 'var(--fail)';
    return;
  }
  loadingMsg.remove();

  renderCrossExperimentMatrices(pane, runIds, perRun, epsByRun);
}

// Shared cross-experiment view: Per-experiment metrics → Attribute comparison
// → Instruction-type comparison → Per-task success rate matrix.
// Used by both Compare and Combine — they only differ in the top of the page.
function renderCrossExperimentMatrices(pane, runIds, perRun, epsByRun) {
  // Use a content container so sort handlers can clear and re-render in place
  // without duplicating into the surrounding pane (which holds the title/chips
  // populated by renderCompareReport / renderAnalyzeReport above us).
  let content = pane.querySelector(':scope > .cross-exp-content');
  if (content) content.innerHTML = '';
  else { content = el('div', { class: 'cross-exp-content' }); pane.appendChild(content); }

  // Build union of all task names across selected runs.
  const allTasks = new Set();
  for (const { tasks } of perRun) for (const t of tasks) allTasks.add(t.task);
  const taskNames = [...allTasks].sort();

  // task → run_id → TaskSummary | undefined
  const grid = {};
  for (const { id, tasks } of perRun) {
    for (const t of tasks) {
      (grid[t.task] ||= {})[id] = t;
    }
  }

  // Per-run totals.
  const totals = Object.fromEntries(perRun.map(({ id, tasks }) => {
    const ne = tasks.reduce((a, t) => a + t.num_episodes, 0);
    const ns = tasks.reduce((a, t) => a + t.num_success, 0);
    return [id, { n: ne, s: ns, rate: ne ? ns / ne : 0 }];
  }));

  // Matrix table — each run gets a (SR%, Score) cell pair.
  const compareTableId = `compareTask::${runIds.join(',')}`;
  seedSort(compareTableId, 'task', 'asc');
  const onCmpSort = () => renderCrossExperimentMatrices(pane, runIds, perRun, epsByRun);
  // Build accessors for sort: by task name, by each experiment's SR/score.
  const cmpAccessors = { task: (task) => task };
  for (const id of runIds) {
    cmpAccessors[`sr::${id}`] = { get: (task) => { const t = grid[task][id]; return t ? t.success_rate : null; }, numeric: true };
    cmpAccessors[`score::${id}`] = { get: (task) => { const t = grid[task][id]; return t ? t.mean_score : null; }, numeric: true };
  }

  const table = el('table', { class: 'matrix panel overflow-hidden' });
  const headTop = el('tr', {},
    el('th', { rowspan: 2, class: 'label-col sort-th',
               style: _sortState(compareTableId).key === 'task' ? { color: 'var(--text-0)' } : {},
               onclick: () => {
                 const ss = _sortState(compareTableId);
                 if (ss.key === 'task') ss.dir = ss.dir === 'asc' ? 'desc' : 'asc';
                 else { ss.key = 'task'; ss.dir = 'asc'; }
                 onCmpSort();
               },
             }, el('span', { class: 'label-cell' }, 'Task',
                  _sortState(compareTableId).key === 'task'
                    ? el('span', { class: 'arrow' }, ` ${_sortState(compareTableId).dir === 'asc' ? '▲' : '▼'}`)
                    : null)),
    ...runIds.map((id) => el('th', { colspan: 2, class: 'pair-start', title: id, style: { textAlign: 'center' } }, id)),
  );
  const headSub = el('tr', {},
    ...runIds.flatMap((id) => [
      sortableTh(compareTableId, `sr::${id}`, 'SR%', { numeric: true, alignRight: true, defaultDir: 'desc', onChange: onCmpSort,
                                                       style: { color: 'var(--text-2)', fontWeight: 400 } }),
      sortableTh(compareTableId, `score::${id}`, 'Score', { numeric: true, alignRight: true, defaultDir: 'desc', onChange: onCmpSort,
                                                            style: { color: 'var(--text-2)', fontWeight: 400 } }),
    ]),
  );
  // Tag the sub-headers with pair-start / pair-end visually so the dividers still draw.
  let nth = 0;
  for (const th of headSub.querySelectorAll('th')) {
    th.classList.add(nth % 2 === 0 ? 'pair-start' : 'pair-end');
    nth += 1;
  }
  table.appendChild(el('thead', {}, headTop, headSub));

  // ---- OVERALL row (pinned to the top of the body) ----
  // Per-experiment overall score = episodes-weighted mean of task-level mean_scores.
  const runOverallScore = (id) => {
    let sum = 0, n = 0;
    for (const t of perRun.find((p) => p.id === id).tasks) {
      if (t.mean_score == null) continue;
      sum += t.mean_score * t.num_episodes;
      n += t.num_episodes;
    }
    return n ? sum / n : null;
  };
  const tot = el('tr', { style: { borderBottom: '2px solid var(--border-strong)' } },
    el('td', { class: 'label-col', style: { fontWeight: 600 } },
       el('span', { class: 'label-cell' }, 'OVERALL')));
  const rates = runIds.map((id) => totals[id].rate);
  const maxOverall = Math.max(...rates);
  const minOverall = Math.min(...rates);
  for (let i = 0; i < runIds.length; i++) {
    const id = runIds[i];
    const t = totals[id];
    const isBest = t.rate === maxOverall && rates.length > 1 && maxOverall > minOverall;
    tot.appendChild(el('td', { class: `cell-num pair-start ${isBest ? 'best' : ''}`, style: { fontWeight: 600 } },
      fmtSRCell(t.s, t.n, { best: isBest })));
    tot.appendChild(el('td', { class: 'cell-num pair-end', style: { fontWeight: 600 } },
      fmtScoreCell(runOverallScore(id), null, null)));
  }
  const body = el('tbody', {});
  body.appendChild(tot);

  // ---- per-task rows ----
  // Click on task name → per-task drill-down across the originally-selected
  // experiments. In Combine, runIds is the synthetic ['Combined']; use the
  // real compareSet so we get one section per actual experiment.
  const drillRunIds = [...state.compareSet];
  const drillFrom = state.selection && state.selection.view === 'analyze' ? 'combine' : 'compare';
  const sortedTaskNames = sortRows(taskNames, compareTableId, cmpAccessors);
  for (const task of sortedTaskNames) {
    const row = el('tr', {});
    const taskCell = el('td', {
      class: 'label-col task-link',
      title: `${task} — click to compare across experiments`,
      onclick: () => selectTaskCompare(task, drillRunIds, drillFrom),
    }, el('span', { class: 'label-cell' }, task));
    row.appendChild(taskCell);
    const taskRates = [];
    for (const id of runIds) {
      const t = grid[task][id];
      taskRates.push(t ? t.success_rate : null);
    }
    const validRates = taskRates.filter((r) => r !== null);
    const maxR = validRates.length ? Math.max(...validRates) : null;
    const minR = validRates.length ? Math.min(...validRates) : null;
    for (let i = 0; i < runIds.length; i++) {
      const id = runIds[i];
      const t = grid[task][id];
      if (!t) {
        row.appendChild(el('td', { class: 'cell-num pair-start', style: { color: 'var(--text-2)' } }, '—'));
        row.appendChild(el('td', { class: 'cell-num pair-end', style: { color: 'var(--text-2)' } }, '—'));
        continue;
      }
      const isBestSR = maxR !== null && t.success_rate === maxR && validRates.length > 1 && maxR > minR;
      row.appendChild(el('td', { class: `cell-num pair-start ${isBestSR ? 'best' : ''}` },
        fmtSRCell(t.num_success, t.num_episodes, { best: isBestSR, lcb: t.sr_lcb, ucb: t.sr_ucb })));
      row.appendChild(el('td', { class: 'cell-num pair-end' },
        fmtScoreCell(t.mean_score, t.score_std, t.score_n, { lcb: t.score_lcb, ucb: t.score_ucb })));
    }
    body.appendChild(row);
  }

  table.appendChild(body);

  // ---- render order: aggregates first, per-task matrix last ----
  content.appendChild(el('h3', { class: 'text-sm font-semibold uppercase tracking-wider mb-2',
                              style: { color: 'var(--text-2)' } }, 'Per-experiment metrics'));
  content.appendChild(buildPerRunMetricsMatrix(runIds, epsByRun));

  // `attributes` is a flat list mixing task-level capability attributes with
  // difficulty tokens. Split the bucket into two sections so they're easier
  // to scan.
  const DIFFICULTY_TOKENS = new Set(['simple', 'moderate', 'complex']);
  const isDifficulty = (a) => DIFFICULTY_TOKENS.has(a);

  content.appendChild(el('h3', { class: 'text-sm font-semibold uppercase tracking-wider mt-6 mb-2',
                              style: { color: 'var(--text-2)' } }, 'Difficulty comparison'));
  content.appendChild(buildCrossRunBreakdown(runIds, epsByRun,
    (e) => (e.attributes || []).filter(isDifficulty),
    `diffComp::${runIds.join(',')}`));

  content.appendChild(el('h3', { class: 'text-sm font-semibold uppercase tracking-wider mt-6 mb-2',
                              style: { color: 'var(--text-2)' } }, 'Attribute comparison'));
  content.appendChild(buildCrossRunBreakdown(runIds, epsByRun,
    (e) => (e.attributes || []).filter((a) => !isDifficulty(a)),
    `attrComp::${runIds.join(',')}`));

  content.appendChild(el('h3', { class: 'text-sm font-semibold uppercase tracking-wider mt-6 mb-2',
                              style: { color: 'var(--text-2)' } }, 'Instruction-type comparison'));
  content.appendChild(buildCrossRunBreakdown(runIds, epsByRun, (e) => [e.instruction_type || '(none)'],
                                             `instrComp::${runIds.join(',')}`));

  content.appendChild(el('h3', { class: 'text-sm font-semibold uppercase tracking-wider mt-6 mb-2',
                              style: { color: 'var(--text-2)' } }, 'Per-task success rate'));
  content.appendChild(table);
}

// Transposed cross-run breakdown: rows = experiments, columns = bucket pairs
// (SR%, Score). keyExtractor(ep) → list of bucket keys to credit this episode
// to. `tableId` is required so multiple breakdowns on the same page have
// separate sort state (e.g. attribute comparison vs instruction-type
// comparison). Best SR per bucket is highlighted across rows.
function buildCrossRunBreakdown(runIds, epsByRun, keyExtractor, tableId) {
  const perRun = {};
  for (const id of runIds) {
    perRun[id] = {};
    for (const ep of (epsByRun[id] || [])) {
      const keys = keyExtractor(ep);
      const list = keys.length ? keys : ['(none)'];
      for (const k of list) {
        if (!perRun[id][k]) perRun[id][k] = { n: 0, s: 0, scoreSum: 0, scoreSqSum: 0, scoreN: 0 };
        const b = perRun[id][k];
        b.n += 1;
        if (ep.success) b.s += 1;
        if (typeof ep.score === 'number' && Number.isFinite(ep.score)) {
          b.scoreSum += ep.score;
          b.scoreSqSum += ep.score * ep.score;
          b.scoreN += 1;
        }
      }
    }
  }
  // Column order: by total episode count across runs (desc) — most-populated
  // buckets first.
  const counts = {};
  for (const id of runIds) {
    for (const [k, v] of Object.entries(perRun[id])) {
      counts[k] = (counts[k] || 0) + v.n;
    }
  }
  const allKeys = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  if (!allKeys.length) {
    return el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'no buckets');
  }

  const cellSR = (id, k) => {
    const b = perRun[id][k];
    return b && b.n ? b.s / b.n : null;
  };
  const cellScore = (id, k) => {
    const b = perRun[id][k];
    return b && b.scoreN ? b.scoreSum / b.scoreN : null;
  };

  // Per-bucket best SR across rows (only when ≥2 runs have data and the values
  // actually differ).
  const bestSR = {};
  for (const k of allKeys) {
    const vals = runIds.map((id) => cellSR(id, k)).filter((v) => v != null);
    if (vals.length < 2) { bestSR[k] = null; continue; }
    const mx = Math.max(...vals);
    const mn = Math.min(...vals);
    bestSR[k] = mx > mn ? mx : null;
  }

  seedSort(tableId, 'id', 'asc');
  const onSort = () => {
    const old = document.querySelector(`[data-table-id="${tableId}"]`);
    if (!old || !old.parentNode) return;
    old.parentNode.replaceChild(buildCrossRunBreakdown(runIds, epsByRun, keyExtractor, tableId), old);
  };

  const accessors = { id: (id) => id };
  for (const k of allKeys) {
    accessors[`sr::${k}`]    = { get: (id) => cellSR(id, k),    numeric: true };
    accessors[`score::${k}`] = { get: (id) => cellScore(id, k), numeric: true };
  }
  const sortedRuns = sortRows(runIds, tableId, accessors);

  // Manual Experiment header with rowspan=2 to span both header rows (mirrors
  // the per-task matrix Task header). sortableTh doesn't take rowspan so we
  // build this one directly.
  const ss = _sortState(tableId);
  const expActive = ss.key === 'id';
  const expArrow = expActive ? (ss.dir === 'asc' ? '▲' : '▼') : '';
  const expTh = el('th', {
    rowspan: 2, class: `label-col sort-th ${expActive ? 'active' : ''}`,
    onclick: () => {
      if (ss.key === 'id') ss.dir = ss.dir === 'asc' ? 'desc' : 'asc';
      else { ss.key = 'id'; ss.dir = 'asc'; }
      onSort();
    },
  }, el('span', { class: 'label-cell' }, 'Experiment',
       expArrow ? el('span', { class: 'arrow' }, ` ${expArrow}`) : null));

  const tbl = el('table', { class: 'matrix panel overflow-hidden', 'data-table-id': tableId });
  const headTop = el('tr', {}, expTh,
    ...allKeys.map((k) => el('th', { colspan: 2, class: 'pair-start', title: k, style: { textAlign: 'center' } }, k)),
  );
  const headSub = el('tr', {},
    ...allKeys.flatMap((k) => [
      sortableTh(tableId, `sr::${k}`, 'SR%', { numeric: true, alignRight: true, defaultDir: 'desc', onChange: onSort,
                                              style: { color: 'var(--text-2)', fontWeight: 400 } }),
      sortableTh(tableId, `score::${k}`, 'Score', { numeric: true, alignRight: true, defaultDir: 'desc', onChange: onSort,
                                                   style: { color: 'var(--text-2)', fontWeight: 400 } }),
    ]),
  );
  // Pair-start/pair-end visuals on the sub-header so the column-pair dividers
  // still draw (matches the per-task matrix).
  let nth = 0;
  for (const th of headSub.querySelectorAll('th')) {
    th.classList.add(nth % 2 === 0 ? 'pair-start' : 'pair-end');
    nth += 1;
  }
  tbl.appendChild(el('thead', {}, headTop, headSub));

  const body = el('tbody', {});
  for (const id of sortedRuns) {
    const row = el('tr', {},
      el('td', { class: 'label-col', title: id, style: { fontFamily: 'ui-monospace, monospace' } },
        el('span', { class: 'label-cell' }, id)));
    for (const k of allKeys) {
      const b = perRun[id][k];
      if (!b) {
        row.appendChild(el('td', { class: 'cell-num pair-start', style: { color: 'var(--text-2)' } }, '—'));
        row.appendChild(el('td', { class: 'cell-num pair-end', style: { color: 'var(--text-2)' } }, '—'));
        continue;
      }
      const sr = b.s / b.n;
      const score = b.scoreN ? b.scoreSum / b.scoreN : null;
      const scoreStd = b.scoreN > 1 ? Math.sqrt(b.scoreSqSum / b.scoreN - score * score) : null;
      const isBest = bestSR[k] != null && sr === bestSR[k];
      row.appendChild(el('td', { class: `cell-num pair-start ${isBest ? 'best' : ''}` },
        fmtSRCell(b.s, b.n, { best: isBest })));
      row.appendChild(el('td', { class: 'cell-num pair-end' },
        fmtScoreCell(score, scoreStd, b.scoreN)));
    }
    body.appendChild(row);
  }
  tbl.appendChild(body);
  return tbl;
}

// Transposed per-run metrics matrix: rows = runs, columns = metrics. Higher-
// is-better metrics (SR%, Mean score) bold the column max; lower-is-better
// (Inference ms) bolds the column min; trajectory metrics have no canonical
// direction so values are shown without highlight. Click any metric header to
// sort the rows by that column.
function buildPerRunMetricsMatrix(runIds, epsByRun) {
  const tableId = `perRunMetrics::${runIds.join(',')}`;

  // Pre-compute per-run stats so sort + best-of detection are cheap.
  const stats = {};
  for (const id of runIds) {
    const eps = epsByRun[id] || [];
    const s = eps.filter((e) => e.success).length;
    const sr = eps.length ? s / eps.length : null;
    const ss = eps.map((e) => e.score).filter((v) => typeof v === 'number' && Number.isFinite(v));
    let score = null, scoreStd = null;
    if (ss.length) {
      score = ss.reduce((a, b) => a + b, 0) / ss.length;
      scoreStd = ss.length > 1
        ? Math.sqrt(ss.reduce((a, b) => a + (b - score) ** 2, 0) / (ss.length - 1))
        : null;
    }
    const inf = inferenceStatsMs(eps);
    const traj = {};
    for (const [field] of TRAJECTORY_METRICS) traj[field] = metricStats(eps, field);
    const numTasks = new Set(eps.map((e) => e.task)).size;
    stats[id] = { id, n: eps.length, s, sr, score, scoreStd, scoreN: ss.length, inf, traj, numTasks };
  }

  // Only show inference / trajectory columns when at least one run has data.
  const showInf = runIds.some((id) => stats[id].inf);
  const visibleTraj = TRAJECTORY_METRICS.filter(([f]) => runIds.some((id) => stats[id].traj[f]));

  // Column descriptors. dir: 'higher' | 'lower' | null (no best highlight).
  const cols = [
    { key: 'episodes', label: 'Episodes', dir: null,     get: (s) => s.n },
    { key: 'success',  label: 'Success',  dir: null,     get: (s) => s.s },
    { key: 'numTasks', label: '# tasks',  dir: null,     get: (s) => s.numTasks },
    { key: 'sr',       label: 'SR%',      dir: 'higher', get: (s) => s.sr },
    { key: 'score',    label: 'Score (100)', dir: 'higher', get: (s) => s.score },
  ];
  if (showInf) cols.push({ key: 'inference', label: 'Inference (ms)', dir: 'lower', get: (s) => s.inf ? s.inf.mean : null });
  for (const [field, label] of visibleTraj) {
    cols.push({ key: `traj::${field}`, label, dir: null, get: (s) => s.traj[field] ? s.traj[field].mean : null, traj: field });
  }

  // Per-column best (only for directional metrics, and only when values differ).
  const bestByKey = {};
  for (const c of cols) {
    if (!c.dir) { bestByKey[c.key] = null; continue; }
    const vals = runIds.map((id) => c.get(stats[id])).filter((v) => v != null && Number.isFinite(v));
    if (vals.length < 2) { bestByKey[c.key] = null; continue; }
    const mx = Math.max(...vals);
    const mn = Math.min(...vals);
    if (mx === mn) { bestByKey[c.key] = null; continue; }
    bestByKey[c.key] = c.dir === 'higher' ? mx : mn;
  }

  seedSort(tableId, 'sr', 'desc');
  const onSort = () => {
    const old = document.querySelector(`[data-table-id="${tableId}"]`);
    if (!old || !old.parentNode) return;
    old.parentNode.replaceChild(buildPerRunMetricsMatrix(runIds, epsByRun), old);
  };

  const accessors = { id: (s) => s.id };
  for (const c of cols) accessors[c.key] = { get: c.get, numeric: true };
  const rows = sortRows(runIds.map((id) => stats[id]), tableId, accessors);

  // Manual Experiment header — sortable, label-col styling.
  const ss = _sortState(tableId);
  const expActive = ss.key === 'id';
  const expArrow = expActive ? (ss.dir === 'asc' ? '▲' : '▼') : '';
  const expTh = el('th', {
    class: `label-col sort-th ${expActive ? 'active' : ''}`,
    onclick: () => {
      if (ss.key === 'id') ss.dir = ss.dir === 'asc' ? 'desc' : 'asc';
      else { ss.key = 'id'; ss.dir = 'asc'; }
      onSort();
    },
  }, el('span', { class: 'label-cell' }, 'Experiment',
       expArrow ? el('span', { class: 'arrow' }, ` ${expArrow}`) : null));

  const tableEl = el('table', { class: 'matrix panel overflow-hidden', 'data-table-id': tableId });
  tableEl.appendChild(el('thead', {}, el('tr', {}, expTh,
    ...cols.map((c) => sortableTh(tableId, c.key, c.label, {
      numeric: true, alignRight: true,
      defaultDir: c.dir === 'lower' ? 'asc' : 'desc',
      title: c.label,
      onChange: onSort,
    })),
  )));
  const body = el('tbody', {});
  for (const s of rows) {
    const tr = el('tr', {});
    tr.appendChild(el('td', { class: 'label-col', title: s.id, style: { fontFamily: 'ui-monospace, monospace' } },
      el('span', { class: 'label-cell' }, s.id)));
    // Episodes / Success / # tasks — plain integer counts.
    tr.appendChild(el('td', { class: 'cell-num' }, s.n.toLocaleString()));
    tr.appendChild(el('td', { class: 'cell-num' }, s.s.toLocaleString()));
    tr.appendChild(el('td', { class: 'cell-num' }, s.numTasks.toLocaleString()));
    // SR cell.
    if (s.n) {
      const isBest = bestByKey.sr != null && s.sr === bestByKey.sr;
      tr.appendChild(el('td', { class: `cell-num ${isBest ? 'best' : ''}` }, fmtSRCell(s.s, s.n, { best: isBest })));
    } else {
      tr.appendChild(el('td', { class: 'cell-num', style: { color: 'var(--text-2)' } }, '—'));
    }
    // Score cell.
    if (s.score != null) {
      const isBest = bestByKey.score != null && s.score === bestByKey.score;
      tr.appendChild(el('td', { class: `cell-num ${isBest ? 'best' : ''}` },
        fmtScoreCell(s.score, s.scoreStd, s.scoreN, { best: isBest })));
    } else {
      tr.appendChild(el('td', { class: 'cell-num', style: { color: 'var(--text-2)' } }, '—'));
    }
    // Inference cell (only when column is shown). Value on top, ±std below.
    if (showInf) {
      if (s.inf) {
        const isBest = bestByKey.inference != null && s.inf.mean === bestByKey.inference;
        tr.appendChild(el('td', { class: `cell-num ${isBest ? 'best' : ''}`, title: `${s.inf.n} tasks` },
          el('span', { class: 'val-with-ci' },
            el('span', { class: 'val', style: isBest ? { color: 'var(--success)', fontWeight: 600 } : {} },
              s.inf.mean.toFixed(1)),
            s.inf.std != null ? el('span', { class: 'ci' }, `±${s.inf.std.toFixed(1)}`) : null)));
      } else {
        tr.appendChild(el('td', { class: 'cell-num', style: { color: 'var(--text-2)' } }, '—'));
      }
    }
    // Trajectory cells — value on top, ±std below (matches the score/inference cells).
    for (const [field, , fmt] of visibleTraj) {
      const c = s.traj[field];
      if (!c) {
        tr.appendChild(el('td', { class: 'cell-num', style: { color: 'var(--text-2)' } }, '—'));
      } else {
        tr.appendChild(el('td', { class: 'cell-num', title: `n=${c.n}` },
          el('span', { class: 'val-with-ci' },
            el('span', { class: 'val' }, fmt(c.mean)),
            el('span', { class: 'ci' },
              `±${fmt(c.std).replace(/\s*(cm\/s|m\/s|m)$/, '')}`))));
      }
    }
    body.appendChild(tr);
  }
  tableEl.appendChild(body);
  return tableEl;
}

// ---- selections / main pane -------------------------------------------------

function setBreadcrumb(...parts) {
  const root = $('#breadcrumb');
  root.innerHTML = '';
  parts.forEach((p, i) => {
    if (i > 0) root.appendChild(el('span', { class: 'mx-1 text-slate-600' }, '/'));
    root.appendChild(typeof p === 'string' ? el('span', {}, p) : p);
  });
}

function selectOverview() {
  state.selection = { view: 'overview' };
  renderOverview();
}

async function selectRun(runId) {
  state.selection = { view: 'run', run_id: runId };
  await ensureTasks(runId);
  renderRun(runId);
}

async function selectTask(runId, task) {
  state.selection = { view: 'task', run_id: runId, task };
  await ensureEpisodes(runId, task);
  renderTask(runId, task);
}

async function selectEpisode(runId, task, envId, runIndex) {
  state.selection = { view: 'episode', run_id: runId, task, env_id: envId, run_index: runIndex };
  await ensureEpisodes(runId, task);
  renderEpisode(runId, task, envId, runIndex);
}

// ---- overview view ----------------------------------------------------------

async function renderOverview() {
  setBreadcrumb('Results overview');
  const pane = $('#pane');
  pane.innerHTML = '';
  pane.appendChild(el('div', { class: 'text-slate-400' }, 'Loading overview…'));
  let data;
  try { data = await fetchJSON('/api/overview'); }
  catch (e) { pane.innerHTML = ''; pane.appendChild(el('div', { class: 'text-red-400' }, String(e))); return; }
  pane.innerHTML = '';

  // policy chips
  const head = el('div', { class: 'mb-6' },
    el('h2', { class: 'text-xl font-semibold mb-1' }, 'Results overview'),
    el('p', { class: 'text-slate-400 text-sm mb-4' },
      `${data.num_runs} experiments · ${data.tasks.length} unique tasks across ${data.policies.length} policies`));
  pane.appendChild(head);

  if (data.policies.length) {
    const section = el('div', { class: 'mb-8' },
      el('h3', { class: 'text-sm font-semibold uppercase tracking-wide text-slate-400 mb-2' }, 'Policies'),
      el('div', { class: 'grid grid-cols-2 md:grid-cols-4 gap-3' },
        ...data.policies.map((p) =>
          el('div', { class: 'panel p-3' },
            el('div', { class: 'font-medium' }, p.policy),
            el('div', { class: 'text-xs text-slate-400' }, `${p.tasks} tasks · ${p.n} episodes`),
            el('div', { class: 'mt-1.5' }, srScoreBar(p.rate, p.score)),
            // SR and Score side by side. CI line wraps under each value via
            // .val-with-ci (inline-flex column); using items-start so the
            // labels align with the top of the value block.
            el('div', { class: 'grid grid-cols-2 gap-3 mt-2 items-start text-sm' },
              el('div', {},
                el('div', { class: 'text-xs mb-0.5', style: { color: 'var(--text-2)' } }, 'SR'),
                el('div', { class: 'metric-block' },
                  fmtSRCell(p.s, p.n, { lcb: p.sr_lcb, ucb: p.sr_ucb }))),
              el('div', {},
                el('div', { class: 'text-xs mb-0.5', style: { color: 'var(--text-2)' } }, 'Score'),
                el('div', { class: 'metric-block' },
                  fmtScoreCell(p.score, p.score_std, p.score_n, { lcb: p.score_lcb, ucb: p.score_ucb }))))))));
    pane.appendChild(section);
  }

  // per-task table — sortable
  seedSort('overviewTasks', 'task', 'asc');
  const onChange = () => renderOverview();
  const sortedTasks = sortRows(data.tasks, 'overviewTasks', {
    task: (r) => r.task,
    n: { get: (r) => r.n, numeric: true },
    s: { get: (r) => r.s, numeric: true },
    sr: { get: (r) => r.rate, numeric: true },
    score: { get: (r) => r.score, numeric: true },
    nruns: { get: (r) => r.runs.length, numeric: true },
  });
  const table = el('table', { class: 'panel w-full text-sm overflow-hidden' },
    el('thead', { class: 'text-xs uppercase tracking-wider' },
      el('tr', {},
        sortableTh('overviewTasks', 'task', 'Task', { onChange }),
        sortableTh('overviewTasks', 'n', 'Episodes', { numeric: true, alignRight: true, onChange }),
        sortableTh('overviewTasks', 's', 'Success', { numeric: true, alignRight: true, onChange }),
        sortableTh('overviewTasks', 'sr', 'SR%', { numeric: true, alignRight: true, onChange, style: { width: '12rem' } }),
        sortableTh('overviewTasks', 'score', 'Score', { numeric: true, alignRight: true, onChange }),
        sortableTh('overviewTasks', 'nruns', 'Number of experiments', { numeric: true, alignRight: true, onChange }))),
    el('tbody', {},
      ...sortedTasks.map((t) =>
        el('tr', { class: 'tbl-row' },
          el('td', { class: 'px-3 py-1.5 font-medium text-left' }, t.task),
          el('td', { class: 'px-3 py-1.5 text-right tabular-nums' }, String(t.n)),
          el('td', { class: 'px-3 py-1.5 text-right tabular-nums' }, String(t.s)),
          el('td', { class: 'px-3 py-1.5' },
            el('div', { class: 'flex items-center gap-2' },
              el('div', { style: { flex: '1' } }, srScoreBar(t.rate, t.score)),
              el('div', { style: { minWidth: '120px', textAlign: 'right' } }, fmtSRCell(t.s, t.n, { lcb: t.sr_lcb, ucb: t.sr_ucb })))),
          el('td', { class: 'px-3 py-1.5 text-right' }, fmtScoreCell(t.score, t.score_std, t.score_n, { lcb: t.score_lcb, ucb: t.score_ucb })),
          el('td', { class: 'px-3 py-1.5 text-xs text-right tabular-nums', style: { color: 'var(--text-2)' } }, String(t.runs.length))))));
  pane.appendChild(table);
}

// ---- run view ---------------------------------------------------------------

async function renderRun(runId) {
  setBreadcrumb(el('a', { class: 'hover:underline cursor-pointer', onclick: selectOverview }, 'Overview'), runId);
  const tasks = state.tasksByRun[runId] || [];
  const totalEps = tasks.reduce((a, t) => a + t.num_episodes, 0);
  const totalSucc = tasks.reduce((a, t) => a + t.num_success, 0);

  const pane = $('#pane');
  pane.innerHTML = '';
  const headerLine = el('p', { class: 'text-sm', style: { color: 'var(--text-2)' } },
    `${tasks.length} tasks · ${totalSucc}/${totalEps} (${fmtPct(totalEps ? totalSucc / totalEps : 0)})`);
  pane.appendChild(el('div', { class: 'mb-6' },
    el('h2', { class: 'text-xl font-semibold mb-1 truncate', title: runId }, runId),
    headerLine));

  // Inline analytics: fetch every task's episodes (cached) and render the
  // shared analytics block above the task list. Renders progressively so
  // the task table can show as soon as analytics are ready.
  const analyticsHost = el('div', { class: 'mb-6' });
  analyticsHost.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } },
    `Loading analytics for ${tasks.length} tasks…`));
  pane.appendChild(analyticsHost);
  const tableSentinel = el('div');
  pane.appendChild(tableSentinel);

  try {
    const lists = await Promise.all(tasks.map(async (t) => {
      const k = `${runId}::${t.task}`;
      if (state.episodesByKey[k]) return state.episodesByKey[k];
      const eps = await fetchJSON(
        `/api/runs/${encodeURIComponent(runId)}/tasks/${encodeURIComponent(t.task)}/episodes`);
      state.episodesByKey[k] = eps;
      return eps;
    }));
    const allEps = lists.flat().map((e) => ({ ...e, run_id: runId }));
    analyticsHost.innerHTML = '';
    // Append avg policy-inference time to the run header (if recorded).
    const infStats = inferenceStatsMs(allEps);
    if (infStats) {
      headerLine.appendChild(document.createTextNode(' · '));
      const txt = infStats.std != null
        ? `avg inference ${infStats.mean.toFixed(1)} ± ${infStats.std.toFixed(1)} ms`
        : `avg inference ${infStats.mean.toFixed(1)} ms`;
      headerLine.appendChild(el('span', {
        title: `policy_inference_avg_ms averaged across ${infStats.n} tasks (per-task value × n_tasks)`,
      }, txt));
    }
    if (allEps.length) {
      // skip the per-policy breakdown — single-run view, only one policy.
      renderEpisodeAnalytics(analyticsHost, allEps, { showPolicy: false, showTaskBreakdown: false });
    }
  } catch (e) {
    analyticsHost.innerHTML = '';
    analyticsHost.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--fail)' } },
      `analytics load failed: ${e.message}`));
  }

  seedSort('runTasks', 'task', 'asc');
  const onRunSortChange = () => renderRun(runId);
  const sortedRunTasks = sortRows(tasks, 'runTasks', {
    task: (t) => t.task,
    // Episodes column shows "<succ>/<total>". Sorting by total alone is a
    // no-op when every task ran the same N — sort by successes, breaking
    // ties on total so the visible numbers still order sensibly.
    n: { get: (t) => (t.num_success || 0) * 100000 + (t.num_episodes || 0), numeric: true },
    sr: { get: (t) => t.success_rate, numeric: true },
    score: { get: (t) => t.mean_score, numeric: true },
    dur: { get: (t) => t.mean_duration, numeric: true },
    instruction: (t) => t.instruction,
  });
  const table = el('table', { class: 'panel w-full text-sm overflow-hidden' },
    el('thead', { class: 'text-xs uppercase tracking-wider' },
      el('tr', {},
        sortableTh('runTasks', 'task', 'Task', { onChange: onRunSortChange }),
        sortableTh('runTasks', 'n', 'Episodes', { numeric: true, alignRight: true, onChange: onRunSortChange }),
        sortableTh('runTasks', 'sr', 'SR%', { numeric: true, alignRight: true, onChange: onRunSortChange }),
        sortableTh('runTasks', 'score', 'Score', { numeric: true, alignRight: true, onChange: onRunSortChange }),
        sortableTh('runTasks', 'dur', 'Avg dur', { numeric: true, alignRight: true, onChange: onRunSortChange,
                                                    title: 'Mean episode wall-clock duration' }),
        sortableTh('runTasks', 'instruction', 'Instruction', { onChange: onRunSortChange }))),
    el('tbody', {},
      ...sortedRunTasks.map((t) =>
        el('tr', { class: 'tbl-row cursor-pointer tbl-row-clickable',
                   onclick: () => { selectTask(runId, t.task); renderSidebar(); } },
          el('td', { class: 'px-3 py-1.5 font-medium text-left' }, t.task),
          el('td', { class: 'px-3 py-1.5 text-right tabular-nums' }, `${t.num_success}/${t.num_episodes}`),
          el('td', { class: 'px-3 py-1.5' },
            el('div', { class: 'flex items-center gap-2' },
              el('div', { style: { flex: '1' } }, srScoreBar(t.success_rate, t.mean_score)),
              el('div', { style: { minWidth: '120px', textAlign: 'right' } }, fmtSRCell(t.num_success, t.num_episodes, { lcb: t.sr_lcb, ucb: t.sr_ucb })))),
          el('td', { class: 'px-3 py-1.5 text-right' }, fmtScoreCell(t.mean_score, t.score_std, t.score_n, { lcb: t.score_lcb, ucb: t.score_ucb })),
          el('td', { class: 'px-3 py-1.5 text-right tabular-nums' }, fmtSec(t.mean_duration)),
          el('td', { class: 'px-3 py-1.5 text-left truncate max-w-md', style: { color: 'var(--text-2)' }, title: t.instruction || '' },
             t.instruction || '—')))));
  tableSentinel.appendChild(el('h3', { class: 'text-sm font-semibold uppercase tracking-wider mb-2',
                                       style: { color: 'var(--text-2)' } }, 'Per-task results'));
  tableSentinel.appendChild(table);
}

// ---- task view --------------------------------------------------------------

function renderTask(runId, task) {
  setBreadcrumb(
    el('a', { class: 'hover:underline cursor-pointer', onclick: selectOverview }, 'Overview'),
    el('a', { class: 'hover:underline cursor-pointer', onclick: () => selectRun(runId) }, runId),
    task,
  );
  const eps = state.episodesByKey[`${runId}::${task}`] || [];
  const n = eps.length;
  const s = eps.filter((e) => e.success).length;
  const instr = eps.find((e) => e.instruction)?.instruction || '';
  const scores = eps.map((e) => e.score).filter((v) => v != null);
  const meanScore = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;

  const pane = $('#pane');
  pane.innerHTML = '';
  pane.appendChild(el('div', { class: 'mb-4' },
    el('h2', { class: 'text-xl font-semibold mb-1 truncate', title: task }, task),
    el('p', { class: 'text-slate-400 text-sm mb-2' }, instr || '—'),
    el('div', { class: 'flex flex-wrap gap-2 text-sm' },
      chip(`${s}/${n} succeeded`),
      chip(`SR ${fmtPct(n ? s / n : 0)}`),
      chip(`Score ${fmtScore(meanScore)}`))));

  // episode grid: each card auto-plays the viewport mp4 (or the first other
  // video) as a preview. Falls back to thumb png, then to a "no media" tile.
  const grid = el('div', { class: 'grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-3' });
  for (const ep of eps) grid.appendChild(buildEpisodeCard(runId, task, ep));
  if (!eps.length) {
    grid.appendChild(el('div', { class: 'text-slate-500 col-span-full' }, 'No episodes.'));
  }
  pane.appendChild(grid);
}

// Reusable episode card: autoplaying viewport preview + run/env caption +
// success badge. Click → episode view. Used by the task view and the
// per-task drill-down across experiments.
function buildEpisodeCard(runId, task, ep) {
  const vids = ep.videos || [];
  const previewCam = vids.find((v) => v.name === 'viewport') || vids[0] || null;
  const previewUrl = previewCam
    ? `/api/runs/${encodeURIComponent(runId)}/tasks/${encodeURIComponent(task)}/episodes/${ep.env_id}/run/${ep.run_index}/video?name=${encodeURIComponent(previewCam.name)}`
    : null;
  const thumbUrl = ep.last_frame_path
    ? `/api/runs/${encodeURIComponent(runId)}/tasks/${encodeURIComponent(task)}/episodes/${ep.env_id}/run/${ep.run_index}/thumb`
    : null;

  // Card top: video (with loading overlay) when available, else thumbnail
  // image, else a "no media" tile. The video sits inside a `.video-host`
  // wrapper so the CSS-driven "Loading…" overlay can show until the first
  // frame is ready.
  let previewBlock;
  if (previewUrl) {
    const wrap = el('div', { class: 'video-host w-full aspect-video' });
    const video = el('video', {
      src: previewUrl,
      autoplay: '', muted: '', loop: '', playsinline: '',
      preload: 'metadata',
      class: 'w-full h-full object-cover ep-thumb',
    });
    applyDefaultPlaybackRate(video);
    attachVideoLoading(video, wrap);
    wrap.appendChild(video);
    previewBlock = wrap;
  } else if (thumbUrl) {
    previewBlock = el('img', { src: thumbUrl, class: 'w-full aspect-video object-cover ep-thumb' });
  } else {
    previewBlock = el('div', { class: 'w-full aspect-video flex items-center justify-center text-xs cam-empty' }, 'no media');
  }

  return el('div', {
    class: 'panel overflow-hidden cursor-pointer ep-card',
    onclick: () => { selectEpisode(runId, task, ep.env_id, ep.run_index); renderSidebar(); },
  },
    previewBlock,
    el('div', { class: 'p-2 flex items-center justify-between' },
      el('div', { class: 'text-xs' }, `run ${ep.run_index} · env ${ep.env_id}`),
      badge(ep.success)));
}

// ---- episode view -----------------------------------------------------------

async function renderEpisode(runId, task, envId, runIndex) {
  setBreadcrumb(
    el('a', { class: 'hover:underline cursor-pointer', onclick: selectOverview }, 'Overview'),
    el('a', { class: 'hover:underline cursor-pointer', onclick: () => selectRun(runId) }, runId),
    el('a', { class: 'hover:underline cursor-pointer', onclick: () => selectTask(runId, task) }, task),
    `run ${runIndex} · env ${envId}`,
  );

  const ep = (state.episodesByKey[`${runId}::${task}`] || [])
    .find((e) => e.env_id === envId && e.run_index === runIndex);
  if (!ep) {
    $('#pane').innerHTML = '<div style="color: var(--fail);">Episode not found.</div>';
    return;
  }
  const videos = ep.videos || [];

  const pane = $('#pane');
  pane.innerHTML = '';

  // ---- header: dataset/episode chip (lerobot puts these in a chip too) ----
  const epLabel = `${task} · run ${runIndex} · env ${envId}`;
  const epChip = el('div', { class: 'chip accent', title: epLabel, style: { fontFamily: 'ui-monospace, monospace' } },
    epLabel);
  const header = el('div', { class: 'flex items-start gap-3 flex-wrap mb-4' },
    epChip,
    badge(ep.success),
    chip(`${ep.episode_step} steps`),
    chip(fmtSec(ep.duration)),
    ep.instruction_type ? chip(`instr: ${ep.instruction_type}`) : null,
    ...(ep.attributes || []).map((a) => chip(a)));
  pane.appendChild(header);

  // ---- LANGUAGE INSTRUCTION block (placed above the viewport so it reads
  // before the user starts watching the video) ----
  pane.appendChild(el('div', { class: 'mb-5' },
    el('div', { class: 'lang-label mb-1' }, 'Language instruction'),
    el('div', { class: 'text-base', style: { color: 'var(--text-0)' } }, ep.instruction || '—')));

  // ---- camera tile grid (lerobot-style, autoplay+muted+loop, mono labels) ----
  // Collect the <video> elements so we can sync seek/playhead with events below.
  const camVideos = [];
  if (videos.length) {
    const grid = el('div', { class: 'cam-grid mb-5' });
    for (const v of videos) {
      const url = `/api/runs/${encodeURIComponent(runId)}/tasks/${encodeURIComponent(task)}/episodes/${envId}/run/${runIndex}/video?name=${encodeURIComponent(v.name)}`;
      const wrap = el('div', { class: 'cam-tile-video-wrap video-host' });
      const video = el('video', {
        src: url,
        autoplay: '', muted: '', loop: '', playsinline: '', controls: '',
      });
      applyDefaultPlaybackRate(video);
      attachVideoLoading(video, wrap);
      camVideos.push(video);
      wrap.appendChild(video);
      grid.appendChild(el('div', { class: 'cam-tile' },
        el('div', { class: 'cam-tile-label' },
          el('span', { class: 'dot' }),
          v.name),
        wrap));
    }
    pane.appendChild(grid);
  }

  // Stash cam videos so subordinate renderers (events strip, time-series plots)
  // can drive playhead sync without us threading the array through every call.
  window.__camVideosForSync = camVideos;

  // (Events strip is rendered inside the Episode tab — see renderTabStatistics.)

  // ---- tab bar ----
  const tabs = ['Episode', 'Metrics', 'Metadata'];
  // Carry forward old tab keys so a deep link doesn't blank the body.
  const legacy = { 'Statistics': 'Episode', 'Action Insights': 'Metrics', 'Meta': 'Metadata', 'Frames': 'Episode' };
  const activeTab = legacy[state.episodeTab] || state.episodeTab || 'Episode';
  const tabBar = el('div', {
    class: 'flex gap-1 mb-4', style: { borderBottom: '1px solid var(--border)' },
  });
  for (const t of tabs) {
    tabBar.appendChild(el('div', {
      class: `tab ${t === activeTab ? 'active' : ''}`,
      onclick: () => { state.episodeTab = t; renderEpisode(runId, task, envId, runIndex); },
    }, t));
  }
  pane.appendChild(tabBar);

  const body = el('div', { id: 'tab-body' });
  pane.appendChild(body);

  if (activeTab === 'Episode') {
    renderTabStatistics(body, runId, task, envId, runIndex, ep);
  } else if (activeTab === 'Metrics') {
    renderTabActionInsights(body, ep);
  } else if (activeTab === 'Metadata') {
    renderTabMeta(body, ep);
  }
}

// ---- events ---------------------------------------------------------------

async function loadAndRenderEvents(host, runId, task, envId, runIndex, camVideos, ep) {
  // Placeholder so the user gets feedback while the events log fetches —
  // matches the "Loading time-series…" affordance the plot host shows below.
  const loadingMsg = el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } },
    'Loading events log…');
  host.appendChild(loadingMsg);
  let data;
  try {
    data = await fetchJSON(
      `/api/runs/${encodeURIComponent(runId)}/tasks/${encodeURIComponent(task)}/episodes/${envId}/run/${runIndex}/events`
    );
  } catch (e) {
    loadingMsg.remove();
    if (/404/.test(e.message)) {
      // No log file at all — same empty-state as no events.
      host.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } },
        'No subtask tracking for this episode.'));
    } else {
      host.appendChild(el('div', { class: 'text-xs', style: { color: 'var(--text-2)' } },
        `events: ${e.message}`));
    }
    return;
  }
  loadingMsg.remove();
  const events = data.events || [];
  if (!events.length) {
    host.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } },
      'No subtask tracking for this episode.'));
    return;
  }

  // Without dt we can't anchor markers to a video timeline; just render the list.
  // maxTime = episode duration when we know it, so the strip spans the entire
  // episode (not just the range of recorded events). Floor at the last event
  // and at 1s so a degenerate empty episode still renders something.
  const dt = data.dt;
  const eventsLast = Math.max(0, ...events.map((e) => e.time_s || 0));
  let maxTime = dt ? Math.max(ep && ep.duration ? ep.duration : 0, eventsLast, 1) : null;
  // If a camera video loads with a richer duration, use that instead — its
  // currentTime is what the playhead is anchored to, so the math should agree.
  if (dt && camVideos && camVideos[0]) {
    const driver = camVideos[0];
    const sync = () => {
      if (driver.duration && Number.isFinite(driver.duration) && driver.duration > maxTime) {
        maxTime = driver.duration;
        // Reposition existing markers under the new span (only matters when
        // video.duration exceeds the recorded duration).
        for (const { ev, node } of markers) {
          node.style.left = `${(ev.time_s / maxTime) * 100}%`;
        }
      }
    };
    driver.addEventListener('loadedmetadata', sync, { once: true });
  }

  host.appendChild(el('div', { class: 'lang-label mb-1' }, `Events (${events.length})`));

  // strip
  let strip = null;
  let playhead = null;
  let markers = [];
  if (dt) {
    strip = el('div', { class: 'events-strip mb-2' });
    strip.appendChild(el('div', { class: 'events-strip-track' }));
    for (const ev of events) {
      const x = (ev.time_s / maxTime) * 100;
      const m = el('div', {
        class: `events-strip-marker ${ev.severity}`,
        style: { left: `${x}%` },
        title: `${(ev.time_s || 0).toFixed(2)}s · ${ev.name}\n${ev.info || ''}`,
        onclick: (e) => { e.stopPropagation(); seekAll(camVideos, ev.time_s); },
      });
      strip.appendChild(m);
      markers.push({ ev, node: m });
    }
    playhead = el('div', { class: 'events-strip-playhead', style: { left: '0%' } });
    strip.appendChild(playhead);
    strip.addEventListener('click', (e) => {
      const rect = strip.getBoundingClientRect();
      const frac = (e.clientX - rect.left) / rect.width;
      seekAll(camVideos, frac * maxTime);
    });
    host.appendChild(strip);
  }

  // score progress bar — tracks cumulative subtask score as the video plays.
  // Only meaningful when at least one event carries a score AND we have a
  // timebase to drive playback sync.
  let scoreBarFill = null;
  let scoreBarText = null;
  const hasAnyScore = events.some((e) => e.score != null);
  if (dt && hasAnyScore) {
    const wrap = el('div', { class: 'flex items-center gap-2 mb-2' });
    wrap.appendChild(el('span', { class: 'lang-label', style: { margin: 0 } }, 'Score'));
    const barOuter = el('div', {
      class: 'flex-1',
      style: {
        height: '6px',
        borderRadius: '3px',
        background: 'var(--border)',
        overflow: 'hidden',
      },
    });
    scoreBarFill = el('div', {
      style: {
        height: '100%',
        width: '0%',
        background: 'var(--success)',
        transition: 'width 120ms linear',
      },
    });
    barOuter.appendChild(scoreBarFill);
    scoreBarText = el('span', {
      class: 'font-mono',
      style: {
        fontSize: '11px',
        color: 'var(--text-1)',
        minWidth: '34px',
        textAlign: 'right',
        fontVariantNumeric: 'tabular-nums',
      },
    }, '0.00');
    wrap.appendChild(barOuter);
    wrap.appendChild(scoreBarText);
    host.appendChild(wrap);
  }

  // list
  const listEl = el('div', { class: 'events-list' });
  const rowEls = events.map((ev) =>
    el('div', {
      class: `events-row ${ev.severity}`,
      onclick: () => { if (ev.time_s != null) seekAll(camVideos, ev.time_s); },
    },
      el('span', { class: 'ev-time' }, ev.time_s != null ? `${ev.time_s.toFixed(2)}s` : `step ${ev.step}`),
      el('span', { class: 'ev-info', title: ev.info || '' },
        el('span', { class: 'ev-name' }, ev.name),
        ev.info ? ' · ' + ev.info : ''),
      el('span', { class: 'ev-time' }, ev.score != null ? `s=${Number(ev.score).toFixed(2)}` : '')));
  rowEls.forEach((r) => listEl.appendChild(r));
  host.appendChild(listEl);

  // Sync: highlight the most-recent passed event as the video plays.
  if (camVideos.length && dt) {
    const driver = camVideos[0];
    let activeIdx = -1;
    let lastScore = -1;  // -1 (not 0/1) so the first paint always fires
    const onTime = () => {
      const t = driver.currentTime;
      // playhead
      if (playhead) playhead.style.left = `${Math.min(100, (t / maxTime) * 100)}%`;
      // find latest event with time_s <= t
      let idx = -1;
      for (let i = 0; i < events.length; i++) {
        if ((events[i].time_s ?? Infinity) <= t) idx = i;
        else break;
      }
      if (idx !== activeIdx) {
        if (activeIdx >= 0) {
          rowEls[activeIdx].classList.remove('active');
          markers[activeIdx] && markers[activeIdx].node.classList.remove('active');
        }
        if (idx >= 0) {
          rowEls[idx].classList.add('active');
          markers[idx] && markers[idx].node.classList.add('active');
          // keep the active row in view (only when user isn't scrolling)
          rowEls[idx].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
        activeIdx = idx;
      }
      // Score bar: cumulative score = latest defined score with time_s <= t.
      // Walk backwards from idx so we skip events that didn't carry a score.
      if (scoreBarFill) {
        let s = 0;
        for (let j = idx; j >= 0; j--) {
          if (events[j].score != null) { s = events[j].score; break; }
        }
        if (s !== lastScore) {
          const pct = Math.max(0, Math.min(1, s)) * 100;
          scoreBarFill.style.width = `${pct}%`;
          if (scoreBarText) scoreBarText.textContent = Number(s).toFixed(2);
          lastScore = s;
        }
      }
    };
    driver.addEventListener('timeupdate', onTime);
  }
}

function seekAll(videos, time_s) {
  if (time_s == null || !Number.isFinite(time_s)) return;
  for (const v of videos) {
    try {
      v.currentTime = Math.max(0, time_s);
      if (v.paused) v.play().catch(() => {});
    } catch { /* readyState too low — ignore, video will catch up */ }
  }
}

// ---- episode tabs ---------------------------------------------------------

async function renderTabStatistics(host, runId, task, envId, runIndex, ep) {
  host.innerHTML = '';

  // Events strip + list at the top of the Episode tab, so it lives next to
  // the time-series charts it's most naturally read against. The cam videos
  // already live above the tab bar, so the strip's click-to-seek and
  // playhead-sync still target the persistent <video> elements via the
  // window.__camVideosForSync stash.
  const camVideos = window.__camVideosForSync || [];
  const eventsHost = el('div', { class: 'mb-5' });
  host.appendChild(eventsHost);
  loadAndRenderEvents(eventsHost, runId, task, envId, runIndex, camVideos, ep);

  if (!ep.has_hdf5) {
    host.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } },
      'No data.hdf5 for this task — nothing to plot.'));
    return;
  }
  const plotsHost = el('div', {});
  plotsHost.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'Loading time-series…'));
  host.appendChild(plotsHost);
  try {
    const ts = await fetchJSON(
      `/api/runs/${encodeURIComponent(runId)}/tasks/${encodeURIComponent(task)}/episodes/${envId}/run/${runIndex}/timeseries`
    );
    plotsHost.innerHTML = '';
    renderPlots(plotsHost, ts);
  } catch (e) {
    plotsHost.innerHTML = '';
    plotsHost.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--fail)' } }, `Time-series error: ${e.message}`));
  }
}

function renderTabFrames(host, ep) {
  host.innerHTML = '';
  if (!ep.last_frame_path) {
    host.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } },
      'No frame thumbnails available for this episode.'));
    return;
  }
  host.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'Last frame:'));
  // Placeholder — only the last-frame png is currently available; future iteration
  // can sample N frames from the mp4 via ffmpeg.
}

function renderTabActionInsights(host, ep) {
  host.innerHTML = '';
  const rows = Object.entries(ep.metrics || {});
  if (!rows.length) {
    host.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } },
      'No action data for this episode.'));
    return;
  }
  const card = el('table', { class: 'panel w-full text-sm overflow-hidden' },
    el('tbody', {}, ...rows.map(([k, v]) =>
      el('tr', { style: { borderTop: '1px solid var(--border)' } },
        el('td', { class: 'px-3 py-1.5', style: { color: 'var(--text-1)' } }, k),
        el('td', { class: 'px-3 py-1.5 tabular-nums' }, typeof v === 'number' ? v.toFixed(4) : String(v))))));
  host.appendChild(card);
}

function renderTabMeta(host, ep) {
  host.innerHTML = '';
  const kv = [
    ['task', ep.task],
    ['policy', ep.policy || '—'],
    ['run_index', ep.run_index],
    ['env_id', ep.env_id],
    ['episode_step', ep.episode_step],
    ['duration', fmtSec(ep.duration)],
    ['success', ep.success ? 'true' : 'false'],
    ['instruction', ep.instruction || '—'],
    ['instruction_type', ep.instruction_type || '—'],
    ['attributes', (ep.attributes || []).join(', ') || '—'],
    ['has_hdf5', ep.has_hdf5 ? 'true' : 'false'],
    ['videos', (ep.videos || []).map((v) => v.name).join(', ') || 'none'],
  ];
  const card = el('table', { class: 'panel w-full text-sm overflow-hidden' },
    el('tbody', {}, ...kv.map(([k, v]) =>
      el('tr', { style: { borderTop: '1px solid var(--border)' } },
        el('td', { class: 'px-3 py-1.5', style: { color: 'var(--text-1)', width: '180px' } }, k),
        el('td', { class: 'px-3 py-1.5 font-mono text-xs' }, String(v))))));
  host.appendChild(card);
}

// Leribble-style: pair consecutive channels (e.g. panda_joint1+panda_joint2) into
// one small chart, lay them out in a 2-col grid. Small (1-d) and special (eef/quat)
// signals get their own chart.
function renderPlots(host, ts) {
  host.innerHTML = '';
  const names = Object.keys(ts.series || {});
  if (!names.length) {
    host.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } },
      'No time-series in this episode.'));
    return;
  }

  // Decide grouping per signal: arrays of {title, labels, x, ys}.
  // Special composite: if arm_joint_pos + gripper_pos are both present, emit
  // a synthetic full-width "state" chart that overlays all of them (mirrors
  // the actions chart shape, since action_dim ≈ joint_dim + gripper).
  const groups = [];
  const handled = new Set();
  if (ts.series['arm_joint_pos']) {
    const armS = ts.series['arm_joint_pos'];
    const gripS = ts.series['gripper_pos'];
    const x = armS.data.map((row) => row[0]);
    const ys = armS.labels.map((_, i) => armS.data.map((row) => row[i + 1]));
    const labels = armS.labels.map((_, i) => `arm_joint_pos[${i}]`);
    if (gripS && gripS.data.length === armS.data.length) {
      const gLabels = gripS.labels;
      for (let i = 0; i < gLabels.length; i++) {
        ys.push(gripS.data.map((row) => row[i + 1]));
        labels.push(gLabels.length === 1 ? 'gripper_pos' : `gripper_pos[${i}]`);
      }
      handled.add('gripper_pos');
    }
    groups.push({ title: 'state', labels, x, ys });
    handled.add('arm_joint_pos');
  }

  for (const name of names) {
    if (handled.has(name)) continue;
    const { labels, data } = ts.series[name];
    const x = data.map((row) => row[0]);
    const ys = labels.map((_, i) => data.map((row) => row[i + 1]));

    if (ys.length <= 1) {
      groups.push({ title: name, labels, x, ys });
      continue;
    }

    // 3-d signals (eef_pos, ee_pos): all in one chart (x/y/z)
    if (ys.length === 3 && (name.includes('eef_pos') || name.includes('ee_pos'))) {
      groups.push({ title: name, labels, x, ys });
      continue;
    }

    // Quaternions: 4 dims, one chart
    if (ys.length === 4 && name.toLowerCase().includes('quat')) {
      groups.push({ title: name, labels, x, ys });
      continue;
    }

    // Actions / states: all dims in one chart, full-width.
    if (name.toLowerCase().includes('action') || name.toLowerCase().includes('state')) {
      groups.push({ title: name, labels, x, ys });
      continue;
    }

    // Multi-d arm joints / other signals: pair up by 2s, like lerobot's joint1+joint2.
    for (let i = 0; i < ys.length; i += 2) {
      const labs = labels.slice(i, i + 2);
      groups.push({
        title: `${name}[${i}${labs.length === 2 ? `, ${i + 1}` : ''}]`,
        labels: labs,
        x,
        ys: ys.slice(i, i + 2),
      });
    }
  }

  const grid = el('div', { class: 'grid grid-cols-1 xl:grid-cols-2 gap-3' });
  host.appendChild(grid);

  // Subtle palette per series within a chart (matches lerobot pairing colors).
  const palette = ['#60a5fa', '#f472b6', '#34d399', '#fbbf24', '#a78bfa', '#fb923c', '#22d3ee'];

  const plotDivs = [];   // for video-driven playhead sync
  for (const g of groups) {
    // Actions + primary state (joint_position) are the most-inspected charts
    // and benefit from the extra horizontal real estate. Other state sub-blocks
    // (joint_velocity, root_pose per object, etc.) use the standard grid width.
    const _name = (g.title || '').toLowerCase();
    const isFullWidth = _name.includes('action')
      || _name === 'state'
      || _name.includes('joint_position');
    const plotDiv = el('div', { style: { height: isFullWidth ? '260px' : '170px' } });
    const card = el('div', {
      class: isFullWidth ? 'panel p-2 col-span-full' : 'panel p-2',
    },
      el('div', { class: 'lang-label mb-1', style: { paddingLeft: '4px' } }, g.title),
      plotDiv);
    grid.appendChild(card);

    const traces = g.labels.map((label, i) => ({
      x: g.x,
      y: g.ys[i],
      mode: 'lines',
      name: label,
      line: { width: 1.25, color: palette[i % palette.length] },
      hovertemplate: '%{x:.2f} → %{y:.3f}<extra>%{fullData.name}</extra>',
    }));

    Plotly.newPlot(plotDiv, traces, {
      margin: { t: 4, r: 8, b: 32, l: 36 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#a4a8b4', size: 10 },
      xaxis: {
        title: { text: ts.dt ? 'time (s)' : 'step', font: { size: 9 }, standoff: 4 },
        gridcolor: 'rgba(120,128,150,0.12)',
        zerolinecolor: 'rgba(120,128,150,0.3)',
        tickfont: { size: 9 },
        ticksuffix: ts.dt ? 's' : '',
      },
      yaxis: {
        gridcolor: 'rgba(120,128,150,0.12)',
        zerolinecolor: 'rgba(120,128,150,0.3)',
        tickfont: { size: 9 },
      },
      legend: {
        orientation: 'h', y: 1.12, x: 0,
        font: { color: '#a4a8b4', size: 9 },
        bgcolor: 'rgba(0,0,0,0)',
      },
      showlegend: g.labels.length > 1,
      // Range over which the playhead can live; relayouted below.
      shapes: [],
      autosize: true,
    }, { displayModeBar: false, responsive: true });

    // Plotly's `responsive: true` only catches *window* resizes. When other
    // async content (events list, breakdowns) lands on the page it can shift
    // the chart's column width without firing a window event, leaving the
    // canvas wider than its panel. A ResizeObserver on the parent card
    // forces Plotly.Plots.resize whenever the container actually changes.
    const ro = new ResizeObserver(() => Plotly.Plots.resize(plotDiv));
    ro.observe(card);

    plotDivs.push(plotDiv);
  }

  // Wire chart ↔ video sync if cam videos are available on this page.
  // Driver = first video; its currentTime drives a vertical "playhead" line
  // on every plot, and clicking on a chart seeks all videos to that x.
  const cams = (window.__camVideosForSync || []);
  if (cams.length && plotDivs.length && ts.dt) {
    const driver = cams[0];
    const makeShape = (t) => [{
      type: 'line', x0: t, x1: t, yref: 'paper', y0: 0, y1: 1,
      line: { color: 'rgba(255,255,255,0.9)', width: 1.5 },
    }];
    // Throttle to one update per animation frame; timeupdate can fire at
    // ~60Hz on some browsers and Plotly relayout isn't free.
    let pending = false;
    const updateShapes = () => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        const shapes = makeShape(driver.currentTime);
        for (const pd of plotDivs) {
          Plotly.relayout(pd, { shapes });
        }
      });
    };
    driver.addEventListener('timeupdate', updateShapes);
    driver.addEventListener('seeked', updateShapes);
    updateShapes();

    for (const pd of plotDivs) {
      pd.on('plotly_click', (ev) => {
        if (ev.points && ev.points.length) seekAll(cams, ev.points[0].x);
      });
      pd.style.cursor = 'crosshair';
    }
  }
}

// ---- boot -------------------------------------------------------------------

// ---- top-level router (Home / Scenes / Tasks / Results) -------------------

function setRoute(route) {
  state.route = route;
  // Toggle the sidebar pane and swap its inner content per route. Home /
  // Scenes hide the sidebar entirely; Results and Tasks each show their
  // own sidebar body.
  const sidebar = $('#sidebar-pane');
  const sbResults = $('#sidebar-content-results');
  const sbTasks = $('#sidebar-content-tasks');
  const usesSidebar = route === 'results' || route === 'tasks';
  if (sidebar) sidebar.style.display = usesSidebar ? '' : 'none';
  if (sbResults) sbResults.style.display = route === 'results' ? '' : 'none';
  if (sbTasks)   sbTasks.style.display   = route === 'tasks'   ? '' : 'none';
  // Highlight active top-nav tab.
  for (const btn of document.querySelectorAll('.topnav-tab')) {
    btn.classList.toggle('active', btn.dataset.route === route);
  }
  // Clear breadcrumb + compare bar between routes.
  $('#breadcrumb').innerHTML = '';
  $('#compare-bar').innerHTML = '';
  if (route === 'home') renderHome();
  else if (route === 'scenes') renderScenesIndex();
  else if (route === 'tasks') renderTasksIndex();
  else { renderCompareBar(); selectOverview(); }
}

function renderHome() {
  setBreadcrumb();
  const pane = $('#pane');
  pane.innerHTML = '';
  pane.appendChild(el('div', { class: 'max-w-4xl mx-auto pt-6' },
    el('div', { class: 'flex items-center gap-3 mb-2' },
      ribbleIcon(40),
      el('h1', { class: 'text-2xl font-semibold' }, 'RoboLab Dashboard')),
    el('p', { class: 'mb-8 text-sm', style: { color: 'var(--text-2)' } },
      'Browse scenes and tasks here, and use the results dashboard to view experiment results at a glance.'),
    el('div', { class: 'grid grid-cols-1 md:grid-cols-3 gap-4' },
      el('div', { class: 'home-tile', onclick: () => setRoute('scenes') },
        el('h3', {}, 'Scenes'),
        el('p', {}, 'Scene library: See available scenes in your local repository.'),
        el('div', { class: 'arrow' }, 'Browse →')),
      el('div', { class: 'home-tile', onclick: () => setRoute('tasks') },
        el('h3', {}, 'Tasks'),
        el('p', {}, 'Task libraries: See benchmarks, task-specific information, and benchmark metrics.'),
        el('div', { class: 'arrow' }, 'Browse →')),
      el('div', { class: 'home-tile', onclick: () => setRoute('results') },
        el('h3', {}, 'Results'),
        el('p', {}, 'Use this dashboard to inspect your experiment results, experiment comparisons, per-episode videos and time-series data.'),
        el('div', { class: 'arrow' }, 'Browse →')))));
}

// ---- Tasks view -----------------------------------------------------------

async function ensureTaskFolders() {
  if (state.catalog.taskFolders) return state.catalog.taskFolders;
  state.catalog.taskFolders = await fetchJSON('/api/tasks/folders');
  return state.catalog.taskFolders;
}

// localStorage-backed accessors for the active task-folders set. Stored as a
// JSON array, kept in `state.tasksFolders` as a Set for cheap membership/add/
// remove. First read seeds with the backend default so a fresh user sees
// `benchmark` selected.
const TASKS_FOLDERS_KEY = 'dashboardTasksFolders';
const TASKS_KNOWN_CUSTOM_KEY = 'dashboardTasksKnownCustomFolders';
function _loadTasksFoldersFromStorage(defaultFolder) {
  if (state.tasksFolders instanceof Set) return state.tasksFolders;
  let arr = null;
  try {
    const raw = localStorage.getItem(TASKS_FOLDERS_KEY);
    if (raw) arr = JSON.parse(raw);
  } catch (e) { /* fall through */ }
  state.tasksFolders = new Set(Array.isArray(arr) ? arr : []);
  if (state.tasksFolders.size === 0 && defaultFolder) state.tasksFolders.add(defaultFolder);
  return state.tasksFolders;
}
function _saveTasksFoldersToStorage() {
  try {
    localStorage.setItem(TASKS_FOLDERS_KEY, JSON.stringify([...state.tasksFolders]));
  } catch (e) { /* private mode etc. — fall through */ }
}
function _loadTasksKnownCustomFromStorage() {
  if (state.tasksKnownCustomFolders instanceof Set) return state.tasksKnownCustomFolders;
  let arr = null;
  try {
    const raw = localStorage.getItem(TASKS_KNOWN_CUSTOM_KEY);
    if (raw) arr = JSON.parse(raw);
  } catch (e) { /* fall through */ }
  state.tasksKnownCustomFolders = new Set(Array.isArray(arr) ? arr : []);
  return state.tasksKnownCustomFolders;
}
function _saveTasksKnownCustomToStorage() {
  try {
    localStorage.setItem(TASKS_KNOWN_CUSTOM_KEY,
      JSON.stringify([...state.tasksKnownCustomFolders]));
  } catch (e) { /* fall through */ }
}

async function validateTaskFolder(path) {
  return await fetchJSON(`/api/tasks/validate?path=${encodeURIComponent(path)}`);
}

async function refreshTaskFolderStatuses() {
  // Validate both active and known-but-inactive custom folders so the
  // sidebar can group inactive customs by their resolved base directory.
  // Presets don't need a per-render hit — they're always under
  // robolab/tasks/.
  _loadTasksKnownCustomFromStorage();
  const toValidate = new Set([
    ...state.tasksFolders,
    ...state.tasksKnownCustomFolders,
  ]);
  const entries = await Promise.all([...toValidate].map(async (f) => {
    try { return [f, await validateTaskFolder(f)]; }
    catch (e) { return [f, { ok: false, reason: 'error', message: String(e.message || e) }]; }
  }));
  state.tasksFolderStatus = Object.fromEntries(entries);
}

// Multi-folder tasks fetch — union of metadata entries across active folders.
async function fetchTasksUnion(folders) {
  if (!folders.length) return [];
  const qs = folders.map((f) => `folder=${encodeURIComponent(f)}`).join('&');
  return await fetchJSON(`/api/tasks?${qs}`);
}

// Populate the Tasks sidebar (`#sidebar-content-tasks`) with the combined
// folder list (presets + known customs, indented to reflect hierarchy) and
// the inline Add input row inside the same header — mirrors the Results
// sidebar shape exactly.
function _renderTasksSidebar(activeFolders, presetList) {
  const host = $('#sidebar-content-tasks');
  if (!host) return;
  host.innerHTML = '';
  const header = el('header', {
    class: 'p-3',
    style: { borderBottom: '1px solid var(--border)' },
  },
    el('h1', {
      class: 'text-xs font-semibold uppercase tracking-wider mb-2',
      style: { color: 'var(--text-2)' },
    }, 'Task directories'),
    _buildTaskFolderList(presetList),
    _buildCustomPathInput(),
  );
  host.appendChild(header);
  // Spacer keeps the flex layout consistent with the Results sidebar.
  host.appendChild(el('div', { class: 'flex-1' }));
}

// Build the combined list, grouped by the base directory each folder was
// resolved against. Presets all share the same base (robolab/tasks/), so
// they group together; user-added customs from elsewhere get their own
// group headers.
function _buildTaskFolderList(presetList) {
  // Combine presets + known customs, dedup by path.
  const seen = new Set();
  const entries = [];
  for (const p of presetList) {
    if (seen.has(p)) continue;
    seen.add(p); entries.push({ path: p, kind: 'preset' });
  }
  for (const c of [...state.tasksKnownCustomFolders]) {
    if (seen.has(c)) continue;
    seen.add(c); entries.push({ path: c, kind: 'custom' });
  }

  // Group by `base` from the validation result. Folders without a status
  // entry yet (or whose validation failed) get their own '(unresolved)'
  // bucket so they're still visible.
  const TASKS_BASE_PRESET = '__robolab_tasks__';
  const groups = new Map();
  for (const entry of entries) {
    const st = state.tasksFolderStatus[entry.path];
    let base = (st && st.base) ? st.base : null;
    // Presets don't always have a status entry (only validated when active),
    // but we know they live under robolab/tasks/ — assign them to that group
    // unconditionally so the sidebar layout is stable across re-renders.
    if (entry.kind === 'preset') base = TASKS_BASE_PRESET;
    const key = base || '(unresolved)';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(entry);
  }

  const container = el('div', { class: 'task-folder-groups' });
  for (const [base, rows] of groups) {
    container.appendChild(_buildTaskFolderGroup(base, rows));
  }
  return container;
}

// Render one group of folders (those resolved against the same base
// directory). Header shows the base path; body is the list of rows.
function _buildTaskFolderGroup(base, rows) {
  const TASKS_BASE_PRESET = '__robolab_tasks__';
  let label;
  if (base === TASKS_BASE_PRESET) {
    label = 'robolab/tasks/';
  } else if (base === '(unresolved)') {
    label = '(unresolved — folder may have moved)';
  } else {
    // Shorten common prefixes so the header doesn't dominate the sidebar.
    label = base.replace(/^.*\/(?=[^/]+\/[^/]+\/?$)/, '…/');
    // Show the original on hover via title attribute.
  }
  const group = el('div', { class: 'task-folder-group' });
  group.appendChild(el('div', {
    class: 'task-folder-group-header',
    title: base === TASKS_BASE_PRESET ? '' : base,
  }, label));
  const list = el('ul', { class: 'task-folder-list' });
  for (const { path, kind } of rows) {
    list.appendChild(_buildTaskFolderRow(path, kind));
  }
  group.appendChild(list);
  return group;
}

// Render one row of the task-folder list. Active rows show the metadata
// count and an × to deactivate. Inactive rows are greyed and clickable
// (whole row) to re-activate. Custom rows that fail revalidation when
// re-activated bubble the error up via the chip row's status icon.
function _buildTaskFolderRow(path, kind) {
  const isActive = state.tasksFolders.has(path);
  const status = state.tasksFolderStatus[path];
  const depth = (path.match(/\//g) || []).length;
  const indent = depth * 22; // px per level — nested folders visibly stepped in

  // Status only matters for active rows (we validate active ones each
  // render). Inactive rows just show the path.
  const isWarning = isActive && status && status.ok && status.reason;
  const isError   = isActive && status && !status.ok;
  const tone = !isActive ? 'inactive'
             : isError ? 'fail'
             : isWarning ? 'warn'
             : 'ok';

  const row = el('li', {
    class: `task-folder-row task-folder-row-${tone}`,
    // Top-level rows get a small base indent so they sit visibly under the
    // (flush-left) group header; each nested level adds `indent` px on top.
    style: { paddingLeft: `${0.6 + indent / 16}rem` },
    title: (status && (status.message || status.resolved)) || path,
  });

  // Label: the path itself, monospaced. Click-to-activate when inactive.
  const label = el('span', { class: 'task-folder-name' }, path);
  if (!isActive) {
    label.style.cursor = 'pointer';
    label.addEventListener('click', async () => {
      // Re-validate before activating — a custom path saved in a previous
      // session may no longer exist, in which case we keep it greyed and
      // surface the reason.
      let result;
      try { result = await validateTaskFolder(path); }
      catch (e) { result = { ok: false, message: String(e.message || e) }; }
      if (!result.ok) {
        // Stash the failed status so the row renders red with the message
        // available via tooltip.
        state.tasksFolderStatus[path] = result;
        // For preset folders, we trust them — only stash for customs.
        if (kind === 'custom') {
          await renderTasksIndex();
          return;
        }
      }
      state.tasksFolders.add(path);
      _saveTasksFoldersToStorage();
      await renderTasksIndex();
    });
  }
  row.appendChild(label);

  // Right side: count (active) / indicator + × button (active).
  const right = el('span', { class: 'task-folder-row-right' });
  if (isActive) {
    if (status && status.ok && !status.reason) {
      right.appendChild(el('span', { class: 'task-folder-count' },
        String(status.metadata_count)));
    } else if (status && status.ok && status.reason) {
      right.appendChild(el('span', { class: 'task-folder-warn-icon' }, '⚠'));
    } else {
      right.appendChild(el('span', { class: 'task-folder-fail-icon' }, '✕'));
    }
    right.appendChild(el('button', {
      class: 'task-folder-x',
      title: 'deactivate (stays in list, grey out)',
      onclick: async (ev) => {
        ev.stopPropagation();
        state.tasksFolders.delete(path);
        _saveTasksFoldersToStorage();
        await renderTasksIndex();
      },
    }, '×'));
  }
  row.appendChild(right);
  return row;
}

// Perpetual custom-path section at the bottom of the sidebar. Adding a
// custom path validates against /api/tasks/validate; failures show inline
// without ever silently dropping the path. Successful adds insert the path
// into both the active set AND the persistent "known customs" set, so it
// stays in the list even after later deactivation.
function _buildCustomPathInput() {
  // Fragment-style holder so the input row sits inline inside the
  // "Task directories" header (no extra section title).
  const wrap = el('div', { class: 'mt-2' });
  const errorDiv = el('div', {
    class: 'text-xs mt-1',
    style: { color: 'var(--fail)', display: 'none', wordBreak: 'break-word' },
  });
  const input = el('input', {
    type: 'text',
    placeholder: 'Add a task directory…',
    class: 'flex-1 min-w-0 text-xs px-2 py-1 rounded font-mono',
    style: { background: 'var(--bg-0)', color: 'var(--text-0)', border: '1px solid var(--border)', outline: 'none' },
    onfocus: function () { this.style.borderColor = 'var(--accent)'; },
    onblur: function () { this.style.borderColor = 'var(--border)'; },
  });
  const tryAdd = async () => {
    const v = (input.value || '').trim();
    if (!v) return;
    errorDiv.style.display = 'none';
    let result;
    try { result = await validateTaskFolder(v); }
    catch (e) {
      errorDiv.textContent = `validate request failed: ${e.message}`;
      errorDiv.style.display = '';
      return;
    }
    if (!result.ok) {
      errorDiv.textContent = result.message || `cannot add "${v}": ${result.reason}`;
      errorDiv.style.display = '';
      return;
    }
    state.tasksFolders.add(v);
    state.tasksKnownCustomFolders.add(v);
    _saveTasksFoldersToStorage();
    _saveTasksKnownCustomToStorage();
    input.value = '';
    await renderTasksIndex();
  };
  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') { ev.preventDefault(); tryAdd(); }
  });
  const addBtn = el('button', {
    class: 'text-xs px-2 py-1 rounded',
    style: { background: 'var(--accent)', color: 'white', border: '1px solid var(--accent)', cursor: 'pointer' },
    onclick: tryAdd,
  }, '+');
  wrap.appendChild(el('div', { class: 'flex gap-1' }, input, addBtn));
  wrap.appendChild(errorDiv);
  return wrap;
}

async function renderTasksIndex() {
  setBreadcrumb();
  const pane = $('#pane');
  pane.innerHTML = '';

  // Load folder presets first so the Add-folder popover knows what's
  // available on disk.
  let foldersMeta;
  try { foldersMeta = await ensureTaskFolders(); }
  catch (e) {
    pane.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--fail)' } },
      `Error loading task folders: ${e.message}`));
    return;
  }
  const presetList = foldersMeta.folders || [];
  const defaultFolder = foldersMeta.default || presetList[0] || 'benchmark';

  // Lazy-init active folders set from localStorage; seed with the default if
  // empty so a fresh user lands on `benchmark`. Also load the persistent set
  // of "known custom" folders (paths the user has added previously, even if
  // currently inactive — they stay listed greyed-out so they can be
  // re-activated with a click).
  _loadTasksFoldersFromStorage(defaultFolder);
  _loadTasksKnownCustomFromStorage();
  const activeFolders = [...state.tasksFolders];

  // Validate every active folder in parallel — fills state.tasksFolderStatus
  // with {ok, py_count, metadata_count, reason, message, resolved} per path.
  await refreshTaskFolderStatuses();

  // Title in the main pane (just the heading — folder management lives in
  // the sidebar so it stays visible while you scroll the task list).
  pane.appendChild(el('h2', { class: 'text-xl font-semibold mb-4' }, 'Tasks'));

  // Sidebar: chip strip + Add-folder panel, populated fresh on every render.
  _renderTasksSidebar(activeFolders, presetList);

  // Summary bullets + task list — both driven by the union of active folders
  // that passed validation (ok=true). Folders without metadata still count
  // as valid for the union — they just won't contribute any rows.
  const summaryHost = el('div', { class: 'mb-5' });
  pane.appendChild(summaryHost);
  const listHost = el('div', {});
  pane.appendChild(listHost);

  const validFolders = activeFolders.filter((f) => {
    const s = state.tasksFolderStatus[f];
    return s && s.ok;
  });
  if (!validFolders.length) {
    listHost.appendChild(el('div', { class: 'panel p-4 text-sm', style: { color: 'var(--text-2)' } },
      'No valid task folders selected. Click "+ Add folder" above to choose one.'));
    return;
  }

  const qs = validFolders.map((f) => `folder=${encodeURIComponent(f)}`).join('&');
  const [summary, tasks] = await Promise.all([
    fetchJSON(`/api/tasks/summary?${qs}`),
    fetchTasksUnion(validFolders),
  ]);
  summaryHost.appendChild(_buildTaskSummaryCard(summary));

  if (!tasks.length) {
    listHost.appendChild(el('div', { class: 'panel p-4 text-sm', style: { color: 'var(--text-2)' } },
      el('p', { class: 'mb-2' },
        'No metadata entries for the selected folder(s). task_metadata.json ' +
        'may not cover these paths — regenerate it to include them.'),
      el('p', { class: 'font-mono text-xs', style: { color: 'var(--text-1)' } },
        'python robolab/tasks/_utils/generate_task_metadata.py')));
    return;
  }

  // Apply current filter (set by clicking a mini card) and sort (set by
  // clicking a column header). Both live on state and survive re-renders.
  const filtered = _applyTaskFilter(tasks, state.tasksFilter);
  const sorted = _applyTaskSort(filtered, state.tasksSort);

  // Active-filter row — render one chip per selected bucket across both kinds.
  const activeChips = [];
  for (const kind of Object.keys(state.tasksFilter)) {
    for (const v of state.tasksFilter[kind]) {
      activeChips.push(el('span', {
        class: 'chip accent cursor-pointer',
        title: 'click to remove',
        onclick: async () => {
          state.tasksFilter[kind].delete(v);
          await renderTasksIndex();
        },
      }, `${kind}: ${v}`));
    }
  }
  if (activeChips.length) {
    listHost.appendChild(el('div', { class: 'mb-3 flex items-center gap-2 text-sm flex-wrap' },
      el('span', { style: { color: 'var(--text-2)' } }, 'Filter'),
      ...activeChips,
      el('span', { style: { color: 'var(--text-2)' } },
        `${sorted.length} of ${tasks.length}`),
      el('button', {
        class: 'text-xs px-2 py-0.5 rounded',
        style: { background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text-1)' },
        onclick: async () => {
          for (const k of Object.keys(state.tasksFilter)) state.tasksFilter[k].clear();
          await renderTasksIndex();
        },
      }, 'clear')));
  }

  // Sortable header helper — only the active column shows an arrow
  // (▲ asc / ▼ desc). Inactive columns are plain headers.
  const sortTh = (key, label, alignRight) => {
    const active = state.tasksSort.key === key;
    const arrow = active ? (state.tasksSort.dir === 'asc' ? '▲' : '▼') : '';
    return el('th', {
      class: `px-3 py-2 sort-th ${active ? 'active' : ''} ${alignRight ? 'text-right' : 'text-left'}`,
      onclick: async () => {
        if (state.tasksSort.key === key) {
          state.tasksSort.dir = state.tasksSort.dir === 'asc' ? 'desc' : 'asc';
        } else {
          state.tasksSort = { key, dir: 'asc' };
        }
        await renderTasksIndex();
      },
    }, label, arrow ? el('span', { class: 'arrow' }, ` ${arrow}`) : null);
  };

  const table = el('table', { class: 'panel w-full text-sm overflow-hidden' },
    el('thead', { class: 'text-xs uppercase tracking-wider' },
      el('tr', {},
        sortTh('task_name', 'Task'),
        el('th', { class: 'px-3 py-2 text-left' }, 'Instruction'),
        sortTh('scene', 'Scene'),
        sortTh('attributes', 'Attributes'),
        sortTh('difficulty_score', 'Difficulty', true),
        sortTh('episode_s', 'Ep len', true))),
    el('tbody', {},
      ...sorted.map((t) =>
        el('tr', { class: 'tbl-row cursor-pointer tbl-row-clickable',
                   onclick: () => selectTaskDetail(t.task_name) },
          el('td', { class: 'px-3 py-1.5 font-medium' }, t.task_name || '?'),
          el('td', { class: 'px-3 py-1.5 text-left truncate max-w-md', style: { color: 'var(--text-1)' }, title: t.instruction || '' },
             t.instruction || '—'),
          el('td', { class: 'px-3 py-1.5 text-left font-mono text-xs', style: { color: 'var(--text-2)' }, title: t.scene || '' },
             t.scene || '—'),
          el('td', { class: 'px-3 py-1.5 text-left text-xs', style: { color: 'var(--text-2)' } }, t.attributes || '—'),
          el('td', { class: 'px-3 py-1.5 text-right text-xs', style: { color: 'var(--text-2)' } },
             t.difficulty_label || (t.difficulty_score != null ? String(t.difficulty_score) : '—')),
          el('td', { class: 'px-3 py-1.5 text-right tabular-nums' }, t.episode_s || '—')))));
  listHost.appendChild(table);
}

function _applyTaskFilter(tasks, f) {
  // f shape: {difficulty: Set, attribute: Set}. A kind with an empty set is
  // a no-op (no filter on that kind). Within a kind, membership = OR. Across
  // kinds, conjunction = AND.
  const dSet = f && f.difficulty;
  const aSet = f && f.attribute;
  const filterDiff = dSet && dSet.size > 0;
  const filterAttr = aSet && aSet.size > 0;
  if (!filterDiff && !filterAttr) return tasks;
  return tasks.filter((t) => {
    if (filterDiff && !dSet.has(t.difficulty_label || '')) return false;
    if (filterAttr) {
      const tags = (t.attributes || '').split(',').map((s) => s.trim());
      if (!tags.some((tag) => aSet.has(tag))) return false;
    }
    return true;
  });
}

function _applyTaskSort(tasks, s) {
  if (!s || !s.key) return tasks;
  const numericKeys = new Set(['difficulty_score', 'episode_s', 'num_subtasks', 'num_atomic_conditions']);
  const isNum = numericKeys.has(s.key);
  const dir = s.dir === 'desc' ? -1 : 1;
  return [...tasks].sort((a, b) => {
    let av = a[s.key];
    let bv = b[s.key];
    if (isNum) {
      av = Number(av);
      bv = Number(bv);
      const aMiss = !Number.isFinite(av), bMiss = !Number.isFinite(bv);
      if (aMiss && bMiss) return 0;
      if (aMiss) return 1;             // missing always sorts last
      if (bMiss) return -1;
      return (av - bv) * dir;
    }
    const as = (av == null ? '' : String(av)).toLowerCase();
    const bs = (bv == null ? '' : String(bv)).toLowerCase();
    if (as === bs) return 0;
    if (!as) return 1;
    if (!bs) return -1;
    return as < bs ? -1 * dir : 1 * dir;
  });
}

function _buildTaskSummaryCard(s) {
  if (!s || !s.total) {
    return el('div', { class: 'panel p-4 text-sm', style: { color: 'var(--text-2)' } },
      `No tasks in ${s ? s.folder : '?'}.`);
  }
  const host = el('div', { class: 'space-y-3' });

  // Top row: small stat cards for the headline numbers.
  const topCards = el('div', { class: 'grid grid-cols-2 md:grid-cols-4 gap-3' },
    statCard('Tasks', String(s.total)),
    statCard('Unique scenes', String(s.unique_scenes ?? 0)),
    statCard('Avg variants/task',
      s.avg_instruction_variants != null ? s.avg_instruction_variants.toFixed(1) : '—'),
    statCard('Avg episode (s)',
      s.avg_episode_s != null ? s.avg_episode_s.toFixed(1) : '—'));
  host.appendChild(topCards);

  // Sub-bucket cards: difficulty + attributes side by side on wide screens.
  const subRow = el('div', { class: 'grid grid-cols-1 lg:grid-cols-2 gap-3' });
  if (s.difficulty && Object.keys(s.difficulty).length) {
    subRow.appendChild(_bucketGroupCard('Difficulty', s.difficulty, s.total, 'difficulty'));
  }
  if (s.attributes && Object.keys(s.attributes).length) {
    subRow.appendChild(_bucketGroupCard('Attributes', s.attributes, s.total, 'attribute'));
  }
  host.appendChild(subRow);

  return host;
}

// A card with a section heading and a row of small sub-cards (one per bucket).
// Total is used to compute the small "X%" subtitle on each chip. kind is
// either "difficulty" or "attribute" — clicking a mini card sets that filter.
function _bucketGroupCard(title, buckets, total, kind) {
  const items = Object.entries(buckets);
  return el('div', { class: 'panel p-3' },
    el('div', { class: 'text-xs uppercase tracking-wider mb-2',
                style: { color: 'var(--text-2)' } }, title),
    el('div', { class: 'flex flex-wrap gap-2' },
      ...items.map(([k, v]) => _bucketMiniCard(k, v, total, kind))));
}

function _bucketMiniCard(label, count, total, kind) {
  const pct = total > 0 ? (100 * count / total).toFixed(0) : '0';
  const set = state.tasksFilter[kind];
  const active = set.has(label);
  return el('div', {
    class: `mini-card ${active ? 'active' : ''}`,
    title: `${count} of ${total} (${pct}%) — click to filter, Ctrl/Cmd-click to stack`,
    onclick: async (ev) => {
      const set = state.tasksFilter[kind];
      if (ev.ctrlKey || ev.metaKey) {
        // Stack mode: toggle this value, leave the other kind's set alone.
        if (set.has(label)) set.delete(label);
        else set.add(label);
      } else {
        // Plain click: select this value as the only filter (across all kinds).
        const wasOnly = set.size === 1 && set.has(label);
        for (const k of Object.keys(state.tasksFilter)) state.tasksFilter[k].clear();
        if (!wasOnly) set.add(label);
      }
      await renderTasksIndex();
    },
  },
    el('div', { class: 'text-xs truncate', title: label, style: { color: 'var(--text-1)' } }, label),
    el('div', { class: 'flex items-baseline justify-between gap-2' },
      el('span', { class: 'text-base font-semibold tabular-nums' }, String(count)),
      el('span', { class: 'text-xs', style: { color: 'var(--text-2)' } }, `${pct}%`)));
}

async function selectTaskDetail(name) {
  setBreadcrumb(
    el('a', { class: 'hover:underline cursor-pointer', onclick: renderTasksIndex }, 'Tasks'),
    name,
  );
  const pane = $('#pane');
  pane.innerHTML = '';
  pane.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'Loading…'));
  let t;
  try { t = await fetchJSON(`/api/tasks/${encodeURIComponent(name)}`); }
  catch (e) { pane.innerHTML = ''; pane.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--fail)' } }, `Error: ${e.message}`)); return; }
  pane.innerHTML = '';

  pane.appendChild(el('div', { class: 'mb-4' },
    el('h2', { class: 'text-2xl font-semibold mb-1' }, t.task_name),
    el('p', { class: 'text-sm font-mono', style: { color: 'var(--text-2)' } }, t.filename || '')));

  // Instruction variants
  const variants = t.instruction_variants || (t.instruction ? { default: t.instruction } : {});
  if (Object.keys(variants).length) {
    pane.appendChild(el('h3', { class: 'text-sm font-semibold uppercase tracking-wider mt-4 mb-2',
                                style: { color: 'var(--text-2)' } }, 'Instructions'));
    const tbl = el('table', { class: 'panel w-full text-sm overflow-hidden' },
      el('tbody', {}, ...Object.entries(variants).map(([k, v]) =>
        el('tr', { class: 'tbl-row' },
          el('td', { class: 'px-3 py-1.5 font-mono text-xs', style: { color: 'var(--text-2)', width: '120px' } }, k),
          el('td', { class: 'px-3 py-1.5' }, v)))));
    pane.appendChild(tbl);
  }

  // Scene preview — give the user a quick visual of where this task lives
  // without forcing a click into the scene detail. Click to drill in.
  if (t.scene) {
    pane.appendChild(el('h3', { class: 'text-sm font-semibold uppercase tracking-wider mt-6 mb-2',
                                style: { color: 'var(--text-2)' } }, 'Scene'));
    const img = el('img', {
      src: `/api/scenes/${encodeURIComponent(t.scene)}/image`,
      class: 'block w-full',
      style: { background: '#000' },
      loading: 'lazy',
      onerror: function () { this.parentElement.style.display = 'none'; },
    });
    pane.appendChild(el('div', {
      class: 'panel overflow-hidden inline-block cursor-pointer mb-2',
      style: { maxWidth: '480px' },
      title: 'open scene detail',
      onclick: () => selectSceneDetail(t.scene),
    },
      img,
      el('div', { class: 'px-3 py-1.5 text-xs font-mono', style: { color: 'var(--text-2)' } }, t.scene)));
  }

  // Key fields
  pane.appendChild(el('h3', { class: 'text-sm font-semibold uppercase tracking-wider mt-6 mb-2',
                              style: { color: 'var(--text-2)' } }, 'Definition'));
  const kv = [
    ['Scene', t.scene
      ? el('a', { class: 'hover:underline cursor-pointer', style: { color: 'var(--accent-fg)' },
                  onclick: () => selectSceneDetail(t.scene) }, t.scene)
      : '—'],
    ['Episode length (s)', t.episode_s || '—'],
    ['Attributes', t.attributes || '—'],
    ['Difficulty', `${t.difficulty_label || ''}${t.difficulty_score != null ? ` (${t.difficulty_score})` : ''}` || '—'],
    ['Contact objects', t.contact_objects || '—'],
    ['Terminations', t.terminations || '—'],
    ['Subtasks', t.subtasks || '—'],
    ['# sequential stages', t.num_sequential_stages != null ? String(t.num_sequential_stages) : '—'],
    ['# atomic conditions', t.num_atomic_conditions != null ? String(t.num_atomic_conditions) : '—'],
    ['# subtasks', t.num_subtasks != null ? String(t.num_subtasks) : '—'],
  ];
  pane.appendChild(el('table', { class: 'panel w-full text-sm overflow-hidden' },
    el('tbody', {}, ...kv.map(([k, v]) =>
      el('tr', { class: 'tbl-row' },
        el('td', { class: 'px-3 py-1.5', style: { color: 'var(--text-2)', width: '200px' } }, k),
        el('td', { class: 'px-3 py-1.5' }, v))))));

}

// ---- Scenes view ----------------------------------------------------------

async function ensureScenesCatalog() {
  if (state.catalog.scenes) return state.catalog.scenes;
  state.catalog.scenes = await fetchJSON('/api/scenes');
  return state.catalog.scenes;
}

async function renderScenesIndex() {
  setBreadcrumb();
  const pane = $('#pane');
  pane.innerHTML = '';
  pane.appendChild(el('div', { class: 'mb-4' },
    el('h2', { class: 'text-xl font-semibold' }, 'Scenes'),
    el('p', { class: 'text-sm', style: { color: 'var(--text-2)' } },
      'Loaded from assets/scenes/_metadata/scene_metadata.json')));
  const loading = el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'Loading…');
  pane.appendChild(loading);
  let data;
  try { data = await ensureScenesCatalog(); }
  catch (e) { loading.textContent = `Error: ${e.message}`; loading.style.color = 'var(--fail)'; return; }
  loading.remove();

  const scenes = data.scenes || [];
  if (data.error) {
    pane.appendChild(el('div', { class: 'panel p-4 mb-4 text-sm', style: { color: 'var(--text-1)' } },
      data.error));
  }
  pane.appendChild(el('p', { class: 'text-sm mb-3', style: { color: 'var(--text-2)' } },
    `${scenes.length} scenes · metadata dir: ${data.metadata_dir || '(none)'}`));

  // Card grid with preview thumbnails (≥ md), with a thumbnail in each card.
  const grid = el('div', { class: 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3' });
  for (const s of scenes) {
    const imgEl = s.has_image
      ? el('img', {
          src: `/api/scenes/${encodeURIComponent(s.scene)}/image`,
          class: 'w-full aspect-video object-cover ep-thumb', loading: 'lazy',
        })
      : el('div', { class: 'w-full aspect-video flex items-center justify-center text-xs cam-empty' },
          'no preview');
    grid.appendChild(el('div', {
      class: 'panel overflow-hidden cursor-pointer ep-card',
      onclick: () => selectSceneDetail(s.scene),
    },
      imgEl,
      el('div', { class: 'p-2' },
        el('div', { class: 'font-mono text-xs truncate', title: s.scene }, s.scene),
        el('div', { class: 'flex items-center justify-between mt-1 text-xs',
                    style: { color: 'var(--text-2)' } },
          el('span', {}, `${s.num_objects} obj`),
          el('span', { title: (s.used_by || []).join(', ') },
             `${(s.used_by || []).length} task${(s.used_by || []).length === 1 ? '' : 's'}`)))));
  }
  pane.appendChild(grid);
}

async function selectSceneDetail(filename) {
  setRoute('scenes');   // keep top nav highlight on scenes
  setBreadcrumb(
    el('a', { class: 'hover:underline cursor-pointer', onclick: renderScenesIndex }, 'Scenes'),
    filename,
  );
  const pane = $('#pane');
  pane.innerHTML = '';
  pane.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'Loading…'));
  let sc;
  try { sc = await fetchJSON(`/api/scenes/${encodeURIComponent(filename)}`); }
  catch (e) { pane.innerHTML = ''; pane.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--fail)' } }, `Error: ${e.message}`)); return; }
  pane.innerHTML = '';

  pane.appendChild(el('div', { class: 'mb-4' },
    el('h2', { class: 'text-2xl font-semibold mb-1 font-mono' }, sc.scene),
    el('p', { class: 'text-sm', style: { color: 'var(--text-2)' } },
      `${sc.objects.length} objects · ${sc.all_prims.length} total prims`)));

  if (sc.has_image) {
    pane.appendChild(el('div', { class: 'panel overflow-hidden mb-6 inline-block',
                                  style: { maxWidth: '700px' } },
      el('img', {
        src: `/api/scenes/${encodeURIComponent(sc.scene)}/image`,
        class: 'w-full block',
        style: { background: '#000' },
      })));
  }

  if ((sc.used_by || []).length) {
    pane.appendChild(el('h3', { class: 'text-sm font-semibold uppercase tracking-wider mb-2',
                                style: { color: 'var(--text-2)' } }, 'Used by'));
    pane.appendChild(el('div', { class: 'flex flex-wrap gap-2 mb-6' },
      ...sc.used_by.map((tname) =>
        el('span', {
          class: 'chip accent cursor-pointer',
          onclick: () => { setRoute('tasks'); selectTaskDetail(tname); },
          title: 'open task',
        }, tname))));
  }

  pane.appendChild(el('h3', { class: 'text-sm font-semibold uppercase tracking-wider mb-2',
                              style: { color: 'var(--text-2)' } }, 'Objects'));
  pane.appendChild(el('table', { class: 'panel w-full text-sm overflow-hidden' },
    el('thead', { class: 'text-xs uppercase tracking-wider' },
      el('tr', {},
        el('th', { class: 'px-3 py-2 text-left' }, 'Name'),
        el('th', { class: 'px-3 py-2 text-left' }, 'Payload (USD asset)'),
        el('th', { class: 'px-3 py-2 text-left' }, 'Description'),
        el('th', { class: 'px-3 py-2 text-left' }, 'Static'))),
    el('tbody', {}, ...sc.objects.map((p) =>
      el('tr', { class: 'tbl-row' },
        el('td', { class: 'px-3 py-1.5 font-mono text-xs' }, p.name || '?'),
        el('td', { class: 'px-3 py-1.5 font-mono text-xs', style: { color: 'var(--text-2)' } },
           (p.payload && p.payload[0]) || '—'),
        el('td', { class: 'px-3 py-1.5 text-xs', style: { color: 'var(--text-1)' } },
           p.description || '—'),
        el('td', { class: 'px-3 py-1.5 text-xs', style: { color: 'var(--text-2)' } },
           p.static_body === true ? 'yes' : p.static_body === false ? 'no' : String(p.static_body)))))));
}

// ---- sidebar chrome (collapse toggle + drag-to-resize) ---------------------

// Persist user's preferred width and collapsed state across reloads.
const SIDEBAR_WIDTH_KEY = 'robolab.sidebar.width';
const SIDEBAR_COLLAPSED_KEY = 'robolab.sidebar.collapsed';
const SIDEBAR_MIN_W = 200;
const SIDEBAR_MAX_W = 600;
const SIDEBAR_DEFAULT_W = 288;

function initSidebarChrome() {
  const pane = $('#sidebar-pane');
  const toggle = $('#sidebar-toggle');
  const handle = $('#sidebar-resize-handle');
  if (!pane || !toggle || !handle) return;

  // Restore width.
  const storedW = parseInt(localStorage.getItem(SIDEBAR_WIDTH_KEY), 10);
  const initW = Number.isFinite(storedW)
    ? Math.min(SIDEBAR_MAX_W, Math.max(SIDEBAR_MIN_W, storedW))
    : SIDEBAR_DEFAULT_W;
  pane.style.width = initW + 'px';

  const setCollapsed = (collapsed) => {
    pane.classList.toggle('collapsed', collapsed);
    toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    toggle.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
    toggle.textContent = collapsed ? '»' : '«';
  };

  // Restore collapsed state.
  setCollapsed(localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === '1');

  toggle.addEventListener('click', () => {
    const nowCollapsed = !pane.classList.contains('collapsed');
    setCollapsed(nowCollapsed);
    localStorage.setItem(SIDEBAR_COLLAPSED_KEY, nowCollapsed ? '1' : '0');
  });

  // Drag-to-resize. New width = pointer X minus pane's left edge, clamped.
  // Suppress transitions during the drag for sub-pixel responsiveness.
  handle.addEventListener('mousedown', (ev) => {
    if (pane.classList.contains('collapsed')) return;
    ev.preventDefault();
    const startLeft = pane.getBoundingClientRect().left;
    pane.classList.add('is-resizing');
    document.body.classList.add('sidebar-resizing');

    const onMove = (e) => {
      const w = Math.min(SIDEBAR_MAX_W, Math.max(SIDEBAR_MIN_W, e.clientX - startLeft));
      pane.style.width = w + 'px';
    };
    const onUp = () => {
      pane.classList.remove('is-resizing');
      document.body.classList.remove('sidebar-resizing');
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      const finalW = parseInt(pane.style.width, 10);
      if (Number.isFinite(finalW)) localStorage.setItem(SIDEBAR_WIDTH_KEY, String(finalW));
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  });

  // Double-clicking the handle resets to the default width.
  handle.addEventListener('dblclick', () => {
    if (pane.classList.contains('collapsed')) return;
    pane.style.width = SIDEBAR_DEFAULT_W + 'px';
    localStorage.setItem(SIDEBAR_WIDTH_KEY, String(SIDEBAR_DEFAULT_W));
  });
}

async function boot() {
  initSidebarChrome();
  $('#overview-btn').addEventListener('click', () => { selectOverview(); renderSidebar(); });
  $('#source-add').addEventListener('click', addSourceFromInput);
  $('#source-input').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') { ev.preventDefault(); addSourceFromInput(); }
  });
  for (const btn of document.querySelectorAll('.topnav-tab, .topnav-brand')) {
    btn.addEventListener('click', () => setRoute(btn.dataset.route));
  }
  // Wire the topnav-brand Ribble — mouseenter/mouseleave on the button
  // toggles .ribble-happy on the inner SVG, which triggers the smile-eyes,
  // arm wave, and one-shot bounce via CSS.
  const brand = document.querySelector('.topnav-brand');
  const brandSvg = brand && brand.querySelector('.ribble-icon');
  if (brand && brandSvg) wireRibble(brandSvg, brand);

  // Land on Home FIRST so the user never sees the Results sidebar flash
  // on a fresh page load. The Results-only state (sources, runs list) loads
  // in the background and is ready by the time they click into Results.
  setRoute('home');
  try {
    await renderSources();
    state.runs = await fetchJSON('/api/runs');
  } catch (e) {
    $('#sidebar').innerHTML = `<div class="p-3 text-sm" style="color: var(--fail);">Failed to load: ${e.message}</div>`;
    return;
  }
  renderSidebar();
}

boot();
