(function () {
    'use strict';

    /* Traces index (traces.html). Lists every task in the benchmark; each
       task expands into a table of published agent runs (model, agent,
       effort, level, result, duration, turns, cost), following the
       run-catalog layout of PostTrainBench's traces browser. Rows link to
       the trace viewer (trace.html?trace=…). */

    const RUNS = window.AUTORLBENCH_RUNS || [];
    const TASKS = window.AUTORLBENCH_TASKS || [];

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
       hover title. */
    function resultCell(run) {
        if (run.passed === null || run.passed === undefined) return dash();
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

    /* ---- Rendering ------------------------------------------------------ */

    function agentCell(run) {
        return `<span class="traces-agent-name">${escHtml(run.agent || '—')}</span>`;
    }

    function levelCell(run) {
        const level = run.level || 'issue';
        return escHtml(level.charAt(0).toUpperCase() + level.slice(1));
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
                            <th>model</th>
                            <th>agent</th>
                            <th>effort</th>
                            <th>level</th>
                            <th>result</th>
                            <th>duration</th>
                            <th>turns</th>
                            <th>cost</th>
                            <th>trace</th>
                        </tr>
                    </thead>
                    <tbody>${runs.map(runRow).join('')}</tbody>
                </table>
            </div>`;
    }

    function groupMeta(runs) {
        const parts = [`${runs.length} run${runs.length === 1 ? '' : 's'}`];
        const passed = runs.filter(r => r.passed === true).length;
        if (runs.some(r => r.passed !== null && r.passed !== undefined)) {
            parts.push(`${passed} passed`);
        }
        return parts.join(' · ');
    }

    function renderGroup(task, runs, open) {
        const meta = runs.length ? groupMeta(runs) : 'no runs yet';
        const body = runs.length
            ? runTable(runs)
            : '<div class="traces-empty">No traces published for this task yet — agent runs will appear here.</div>';
        return `
            <details class="traces-group"${open ? ' open' : ''}>
                <summary>
                    <span class="traces-group-caret" aria-hidden="true">›</span>
                    <span class="traces-group-title">${escHtml(task.title)}</span>
                    <span class="traces-group-tag">${escHtml(task.family)}</span>
                    <span class="traces-group-meta">${escHtml(meta)}</span>
                </summary>
                <div class="traces-group-body">${body}</div>
            </details>`;
    }

    function renderPage() {
        const mount = document.getElementById('traces-root');
        if (!mount) return;

        const byTask = runsByTask();
        const withRuns = TASKS.filter(t => (byTask[t.id] || []).length);

        const header = `
            <div class="traces-header">
                <div class="task-header-tags">
                    <span>Run catalog</span>
                    <span>${escHtml(String(TASKS.length))} tasks</span>
                    <span>${escHtml(String(RUNS.length))} run${RUNS.length === 1 ? '' : 's'}</span>
                </div>
                <h1>Traces</h1>
                <p class="traces-intro">Every published agent run, grouped by task. Open a run to inspect its full step-by-step trace — reasoning, tool calls, and outputs.</p>
            </div>`;

        const groups = TASKS.map(task => {
            const runs = byTask[task.id] || [];
            return renderGroup(task, runs, runs.length > 0);
        }).join('\n');

        mount.innerHTML = `
            ${header}
            <div class="traces-list">${groups}</div>
            ${withRuns.length === 0 ? '<div class="traces-empty traces-empty--page">No traces published yet.</div>' : ''}`;

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
