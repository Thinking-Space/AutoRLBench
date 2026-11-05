(function () {
    'use strict';

    /* Traces index (traces.html). Published agent runs are grouped by failure
       category, then by task; every task list is collapsed until opened. A
       search box filters by task title or run model / agent, a category
       dropdown filters sections, a model dropdown keeps runs of one model,
       a level dropdown keeps only issue or solution runs, and a result
       dropdown keeps only passed or failed runs inside each table. Rows link
       to the trace viewer (trace.html?trace=…). */

    const TASKS = (window.AUTORLBENCH_TASKS || [])
        .concat(window.AUTORLBENCH_TRACE_TASKS || []);
    /* Only runs of listed tasks (catalog plus trace-only tasks) are shown. */
    const KNOWN = new Set(TASKS.map(t => t.id));
    const RUNS = (window.AUTORLBENCH_RUNS || []).filter(r => KNOWN.has(r.task));

    const TASK_CATEGORIES = [
        'Configuration',
        'Reward',
        'Data',
        'Environment',
        'Algorithm',
        'Infrastructure'
    ];

    let query = '';
    let category = 'all';
    let result = 'all';
    let level = 'all';
    let model = 'all';

    /* ---- Formatting helpers ------------------------------------------ */

    function escHtml(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function fmtNum(n) {
        return Number(n).toLocaleString('en-US');
    }

    function fmtCost(n) {
        return '$' + Number(n).toFixed(2);
    }

    function fmtDuration(ms) {
        const s = Math.round(ms / 1000);
        if (s < 60) return s + 's';
        if (s < 3600) return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
        return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
    }

    function fmtScore(score) {
        return score === null || score === undefined ? null : (score * 100).toFixed(0) + '%';
    }

    function dash() {
        return '<span class="traces-dash">—</span>';
    }

    /* Result cell: pass/fail mark only — the numeric score stays in the
       hover title. Hacking-flagged runs show the tag in place of the mark. */
    function resultCell(run) {
        if (run.passed === null || run.passed === undefined) return dash();
        if (run.hacking) {
            return '<span class="traces-hacking-tag" title="The submitted fix games the reward instead of repairing it, or load-bearingly relies on network-fetched upstream material; no credit.">hacking</span>';
        }
        const score = fmtScore(run.score);
        const title = score != null ? `judge score ${score}` : (run.passed ? 'passed' : 'failed');
        const mark = run.passed ? '✅' : '❌';
        return `<span class="traces-result" title="${escHtml(title)}">${mark}</span>`;
    }

    /* ---- Data shaping -------------------------------------------------- */

    function runsByTask() {
        const map = {};
        for (const run of RUNS) {
            (map[run.task] = map[run.task] || []).push(run);
        }
        /* Stable table order: by model, then agent. */
        for (const id of Object.keys(map)) {
            map[id].sort((a, b) =>
                String(a.model || '').localeCompare(String(b.model || ''))
                || String(a.agent || '').localeCompare(String(b.agent || '')));
        }
        return map;
    }

    /* ---- Filtering ------------------------------------------------------ */

    function taskMatches(task, runs) {
        if (category !== 'all' && task.family !== category) return false;
        const q = query.trim().toLowerCase();
        if (!q) return true;
        if (task.title.toLowerCase().includes(q)) return true;
        return runs.some(run =>
            String(run.model || '').toLowerCase().includes(q)
            || String(run.agent || '').toLowerCase().includes(q));
    }

    function visibleRuns(runs) {
        if (model !== 'all') {
            runs = runs.filter(r => String(r.model || '') === model);
        }
        if (level !== 'all') {
            runs = runs.filter(r => (r.level || 'issue') === level);
        }
        if (result !== 'all') {
            runs = runs.filter(r => result === 'passed' ? r.passed === true : r.passed === false);
        }
        return runs;
    }

    /* A task is shown when it matches the search / category and, under a
       model, result, or level filter, still has at least one matching run. */
    function visibleEntries() {
        const byTask = runsByTask();
        const entries = [];
        const narrowing = model !== 'all' || result !== 'all' || level !== 'all';
        for (const task of TASKS) {
            const runs = byTask[task.id] || [];
            if (!taskMatches(task, runs)) continue;
            const vis = visibleRuns(runs);
            if (narrowing && vis.length === 0) continue;
            entries.push({ task, runs: narrowing ? vis : runs, total: runs.length });
        }
        return entries;
    }

    /* ---- Rendering ------------------------------------------------------ */

    function agentCell(run) {
        return `<span class="traces-agent-name">${escHtml(run.agent || '—')}</span>`;
    }

    function levelCell(run) {
        /* Same chip style as the tasks page (Issue / Solution). */
        const level = run.level || 'issue';
        const label = level.charAt(0).toUpperCase() + level.slice(1);
        return `<span class="task-level-tag">${label}</span>`;
    }

    function runRow(run) {
        const cells = [
            `<td class="traces-col-model">${escHtml(run.model || '—')}</td>`,
            `<td class="traces-col-agent">${agentCell(run)}</td>`,
            `<td class="traces-col-effort">${run.effort ? escHtml(run.effort) : dash()}</td>`,
            `<td class="traces-col-level">${levelCell(run)}</td>`,
            `<td class="traces-col-result">${resultCell(run)}</td>`,
            `<td class="traces-col-num">${run.durationMs != null ? escHtml(fmtDuration(run.durationMs)) : dash()}</td>`,
            `<td class="traces-col-num">${run.steps != null ? fmtNum(run.steps) : dash()}</td>`,
            `<td class="traces-col-num">${run.costUsd != null ? escHtml(fmtCost(run.costUsd)) : dash()}</td>`,
            `<td class="traces-col-open">${run.slug ? `<a class="traces-open" href="trace.html?trace=${encodeURIComponent(run.slug)}">Open trace</a>` : dash()}</td>`
        ];
        return `<tr${run.slug ? ' class="traces-row-link" data-href="trace.html?trace=' + encodeURIComponent(run.slug) + '"' : ''}>${cells.join('')}</tr>`;
    }

    function runTable(runs) {
        return `
            <div class="traces-table-scroll">
                <table class="traces-table">
                    <thead>
                        <tr>
                            <th class="traces-col-model">model</th>
                            <th class="traces-col-agent">agent</th>
                            <th class="traces-col-effort">effort</th>
                            <th class="traces-col-level">level</th>
                            <th class="traces-col-result">result</th>
                            <th class="traces-col-num">duration</th>
                            <th class="traces-col-num">turns</th>
                            <th class="traces-col-num">cost</th>
                            <th class="traces-col-open">trace</th>
                        </tr>
                    </thead>
                    <tbody>${runs.map(runRow).join('')}</tbody>
                </table>
            </div>`;
    }

    function groupMeta(runs) {
        const parts = [`${runs.length} run${runs.length === 1 ? '' : 's'}`];
        /* Under a result filter the shown runs are all of that kind, so
           count that kind instead of the (always-zero) complement. */
        if (result === 'failed') {
            const failed = runs.filter(r => r.passed === false).length;
            if (failed) parts.push(`${failed} failed`);
        } else if (runs.some(r => r.passed !== null && r.passed !== undefined)) {
            parts.push(`${runs.filter(r => r.passed === true).length} passed`);
        }
        return parts.join(' · ');
    }

    /* Task groups start collapsed; the page copy stays quiet until the
       reader opens a run table. */
    function renderGroup(task, runs, total) {
        const meta = runs.length ? groupMeta(runs, total) : 'no runs yet';
        const body = runs.length
            ? runTable(runs)
            : '<div class="traces-empty">No traces published for this task yet — agent runs will appear here.</div>';
        return `
            <details class="traces-group">
                <summary>
                    <span class="traces-group-caret" aria-hidden="true">›</span>
                    <span class="traces-group-title">${escHtml(task.title)}</span>
                    <span class="traces-group-meta">${escHtml(meta)}</span>
                </summary>
                <div class="traces-group-body">${body}</div>
            </details>`;
    }

    function renderCategory(family, groups, runCount) {
        if (!groups.length) return '';
        const meta = [
            `${groups.length} task${groups.length === 1 ? '' : 's'}`,
            `${runCount} run${runCount === 1 ? '' : 's'}`
        ].join(' · ');
        return `
            <section class="traces-category">
                <div class="traces-category-head">
                    <span class="traces-category-title">${escHtml(family)}</span>
                    <span class="traces-category-meta">${escHtml(meta)}</span>
                </div>
                ${groups.join('\n')}
            </section>`;
    }

    function renderGroups() {
        const entries = visibleEntries();
        const list = document.getElementById('traces-list');
        if (!list) return;

        let html = '';
        for (const family of TASK_CATEGORIES) {
            const inFamily = entries.filter(e => e.task.family === family);
            const runCount = inFamily.reduce((sum, e) => sum + e.runs.length, 0);
            html += renderCategory(family, inFamily.map(e => renderGroup(e.task, e.runs)), runCount);
        }

        /* Model list derived from the data so every run is selectable. */
        const filtered = query.trim() || category !== 'all' || result !== 'all'
            || level !== 'all' || model !== 'all';
        list.innerHTML = html
            || '<div class="traces-empty traces-empty--page">'
              + (filtered ? 'No traces match the current search and filters.' : 'No traces published yet.')
              + '</div>';

        const tasksCount = document.getElementById('traces-tasks-count');
        if (tasksCount) {
            tasksCount.textContent = entries.length === TASKS.length
                ? `${TASKS.length} tasks`
                : `${entries.length} of ${TASKS.length} tasks`;
        }
        const runsCount = document.getElementById('traces-runs-count');
        if (runsCount) {
            const visible = entries.reduce((sum, e) => sum + e.runs.length, 0);
            runsCount.textContent = `${visible} runs`;
        }
    }

    function renderFilters() {
        const search = document.getElementById('traces-search');
        if (search) {
            search.addEventListener('input', event => {
                query = event.target.value;
                renderGroups();
            });
        }
        const categorySelect = document.getElementById('traces-category-select');
        if (categorySelect) {
            categorySelect.addEventListener('change', event => {
                category = event.target.value;
                renderGroups();
            });
        }
        const resultSelect = document.getElementById('traces-result-select');
        if (resultSelect) {
            resultSelect.addEventListener('change', event => {
                result = event.target.value;
                renderGroups();
            });
        }
        const levelSelect = document.getElementById('traces-level-select');
        if (levelSelect) {
            levelSelect.addEventListener('change', event => {
                level = event.target.value;
                renderGroups();
            });
        }
        const modelSelect = document.getElementById('traces-model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', event => {
                model = event.target.value;
                renderGroups();
            });
        }
    }

    function renderPage() {
        const mount = document.getElementById('traces-root');
        if (!mount) return;

        /* Model list derived from the data so every run is selectable. */
        const models = Array.from(new Set(RUNS.map(r => r.model).filter(Boolean)))
            .sort((a, b) => a.localeCompare(b));
        const header = `
            <div class="traces-header">
                <div class="task-header-tags">
                    <span>Run catalog</span>
                    <span id="traces-tasks-count">${TASKS.length} tasks</span>
                    <span id="traces-runs-count">${RUNS.length} runs</span>
                </div>
                <h1>Traces</h1>
                <p class="traces-intro">Every published agent run, grouped by category and task. Open a run to inspect its full step-by-step trace — reasoning, tool calls, and outputs.</p>
                <div class="tasks-filters">
                    <div class="tasks-search-wrap">
                        <svg class="tasks-search-icon" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
                        <input id="traces-search" class="tasks-search" type="search" placeholder="Search by task, model, or agent" autocomplete="off" aria-label="Search traces by task, model, or agent">
                    </div>
                    <select id="traces-category-select" class="tasks-category-select" aria-label="Filter by failure category">
                        <option value="all">All Categories</option>
                        ${TASK_CATEGORIES.map(family => `<option value="${escHtml(family)}">${escHtml(family)}</option>`).join('')}
                    </select>
                    <select id="traces-result-select" class="tasks-category-select" aria-label="Filter by run result">
                        <option value="all">All Results</option>
                        <option value="passed">Passed</option>
                        <option value="failed">Failed</option>
                    </select>
                    <select id="traces-model-select" class="tasks-category-select" aria-label="Filter by model">
                        <option value="all">All Models</option>
                        ${models.map(m => `<option value="${escHtml(m)}">${escHtml(m)}</option>`).join('')}
                    </select>
                    <select id="traces-level-select" class="tasks-category-select" aria-label="Filter by evaluation level">
                        <option value="all">All Levels</option>
                        <option value="issue">Issue</option>
                        <option value="solution">Solution</option>
                    </select>
                </div>
            </div>`;

        mount.innerHTML = `${header}<div class="traces-list" id="traces-list"></div>`;
        renderFilters();
        renderGroups();
        bindRowNavigation(mount);
    }

    /* Rows are clickable anywhere except the explicit link. */
    function bindRowNavigation(mount) {
        mount.addEventListener('click', event => {
            if (event.target.closest('a, button, summary')) return;
            const row = event.target.closest('tr.traces-row-link');
            if (row) window.location.href = row.dataset.href;
        });
    }

    document.addEventListener('DOMContentLoaded', renderPage);
})();
