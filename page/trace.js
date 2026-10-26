(function () {
    'use strict';

    /* Agent trace viewer.
     *
     * Renders an ATIF-format agent session trace (the JSON files under
     * page/agent-trace/) in a two-column console layout modeled on
     * PostTrainBench's run view: a narrow left rail with the run summary,
     * and a structured event timeline as the main column — thread line,
     * turn markers, and block cards (thought / text / tool call). Thoughts
     * and tool calls are collapsed by default; long outputs clip with a
     * per-call expand button. Not yet linked from the homepage.
     */

    /* Run registry (trace-data.js): each entry carries the trace file and
       the metadata (task, effort, result) this viewer adds around the ATIF
       JSON itself. Keyed by slug — the same slug the Traces page links
       against via trace.html?trace=<slug>. */
    const RUNS = window.AUTORLBENCH_RUNS || [];
    const TRACES = {};
    const TASK_SLUGS = {};
    const TRACE_META = {};
    for (const run of RUNS) {
        if (!run.slug || !run.file) continue;
        TRACES[run.slug] = run.file;
        TASK_SLUGS[run.slug] = run.task;
        TRACE_META[run.slug] = { effort: run.effort, passed: run.passed, score: run.score };
    }

    const OUT_COLLAPSE_CHARS = 1200;
    const ARG_COLLAPSE_LINES = 14;

    const state = {
        trace: null,
        fileName: null,
        expanded: {}     // call key -> output expanded
    };

    /* ---- Formatting helpers ---------------------------------------- */

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

    function fmtTime(iso) {
        const d = new Date(iso);
        return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    function fmtDuration(ms) {
        const s = Math.round(ms / 1000);
        if (s < 60) return s + 's';
        return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
    }

    function fmtElapsed(ms) {
        const s = Math.max(0, Math.round(ms / 1000));
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        const sec = s % 60;
        if (h > 0) return h + ':' + String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0');
        return m + ':' + String(sec).padStart(2, '0');
    }

    function fmtTokens(n) {
        if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
        if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
        return String(n);
    }

    /* ---- Trace indexing --------------------------------------------- */

    function indexObservations(trace) {
        trace._obsByCall = {};
        for (const step of trace.steps) {
            const results = (step.observation && step.observation.results) || [];
            for (const r of results) {
                if (r.source_call_id) {
                    (trace._obsByCall[r.source_call_id] = trace._obsByCall[r.source_call_id] || []).push(r.content || '');
                }
            }
        }
    }

    function observationFor(trace, callId) {
        const results = (trace._obsByCall && trace._obsByCall[callId]) || [];
        return results.join('\n\n');
    }

    function traceStats(trace) {
        const steps = trace.steps || [];
        let toolCalls = 0;
        let thoughts = 0;
        for (const step of steps) {
            toolCalls += (step.tool_calls || []).length;
            if (step.reasoning_content) thoughts++;
        }
        const durationMs = steps.length > 1
            ? new Date(steps[steps.length - 1].timestamp) - new Date(steps[0].timestamp)
            : 0;
        return { steps: steps.length, toolCalls, thoughts, durationMs };
    }

    function traceTitle(trace) {
        const first = (trace.steps || []).find(s => s.source === 'user');
        const heading = first && first.message && String(first.message).match(/^#\s+(.+)$/m);
        return heading ? heading[1].trim() : 'Agent run';
    }

    /* Resolve the trace's slug (the file-name segment matched against the
       TASK_SLUGS / TRACE_META registries). Falls back to stripping the
       .json extension for hand-named files like glm53-aime25-so__gRxe6uw. */
    function traceSlug(fileName) {
        const base = fileName.split('/').pop().replace(/\.json$/, '');
        for (const part of base.split('__')) {
            if (TASK_SLUGS[part] || TRACE_META[part]) return part;
        }
        return base;
    }

    function taskFor(fileName) {
        const id = TASK_SLUGS[traceSlug(fileName)];
        if (!id) return null;
        const task = (window.AUTORLBENCH_TASKS || []).find(t => t.id === id)
            || (window.AUTORLBENCH_TRACE_TASKS || []).find(t => t.id === id);
        return { id, title: task ? task.title : id };
    }

    /* ---- Tool argument rendering (per-tool structure) --------------- */

    function truncateLine(s, max) {
        s = String(s).trim();
        return s.length > max ? s.slice(0, max - 1) + '…' : s;
    }

    function preBlock(text, cls) {
        return `<pre class="trace-pre${cls ? ' ' + cls : ''}"><code>${escHtml(text)}</code></pre>`;
    }

    /* Edit calls render as a mini diff: old string red, new string green. */
    function renderDiffLines(text, sign, cls) {
        const lines = String(text).replace(/\n$/, '').split('\n');
        return lines.map(l => `<pre class="trace-pre trace-diff-${cls}"><code>${escHtml(sign + ' ' + l)}</code></pre>`).join('');
    }

    /* Tool arguments may arrive as a JSON string rather than an object. */
    function normalizeArgs(args) {
        if (typeof args === 'string') {
            try { return JSON.parse(args); } catch (err) { return null; }
        }
        return args;
    }

    function renderToolArgs(fn, rawArgs) {
        const args = normalizeArgs(rawArgs);
        if (args === null || args === undefined || typeof args !== 'object') {
            if (rawArgs === undefined || rawArgs === null) return '<div class="trace-note">No arguments</div>';
            return preBlock(String(rawArgs));
        }
        const description = typeof args.description === 'string' && args.description
            ? `<div class="trace-arg-desc">${escHtml(args.description)}</div>`
            : '';

        if (fn === 'Bash' && typeof args.command === 'string') {
            return `<div class="trace-bash-cmd"><code>${escHtml(args.command)}</code></div>${description}`;
        }
        if ((fn === 'Read' || fn === 'Write' || fn === 'Edit') && typeof args.file_path === 'string') {
            let html = `<div class="trace-file-path">${escHtml(args.file_path)}</div>`;
            if (fn === 'Edit') {
                html += renderDiffLines(args.old_string || '', '-', 'del');
                html += renderDiffLines(args.new_string || '', '+', 'add');
            } else if (fn === 'Write' && typeof args.content === 'string') {
                const content = args.content;
                html += `
                    <details class="trace-content-details">
                        <summary>content · ${fmtNum(content.length)} chars</summary>
                        ${preBlock(content)}
                    </details>`;
            }
            return html + description;
        }

        // Generic tool: pretty-printed JSON, clamped when huge.
        let text;
        try { text = JSON.stringify(args, null, 2); }
        catch (err) { text = String(args); }
        if (!text) return '<div class="trace-note">No arguments</div>';
        const lines = text.split('\n');
        const tooLong = lines.length > ARG_COLLAPSE_LINES || text.length > OUT_COLLAPSE_CHARS;
        const shown = tooLong ? lines.slice(0, ARG_COLLAPSE_LINES).join('\n') : text;
        return preBlock(shown) + description;
    }

    /* Short chip label for the collapsed header of a tool call: the
       command line, the file path, or the first line of the arguments. */
    function toolCallLabel(fn, rawArgs) {
        const args = normalizeArgs(rawArgs);
        if (args && typeof args === 'object') {
            if (typeof args.command === 'string' && args.command) return args.command;
            if (typeof args.file_path === 'string') return args.file_path;
            if (typeof args.path === 'string') return args.path;
            if (typeof args.pattern === 'string') return args.pattern;
        }
        return fn;
    }

    function renderToolOutput(text) {
        if (!text) return '<div class="trace-note">No output</div>';
        const tooLong = text.length > OUT_COLLAPSE_CHARS;
        const shown = tooLong ? text.slice(0, OUT_COLLAPSE_CHARS) : text;
        return preBlock(shown);
    }

    /* ---- Event rendering --------------------------------------------- */

    /* Elapsed time since the run's first step; wall clock on hover. */
    function markerTime(step, t0) {
        const elapsed = new Date(step.timestamp) - t0;
        const wall = fmtTime(step.timestamp);
        return `<div class="trace-marker-time" title="${escHtml(wall)} (wall clock)">${escHtml(fmtElapsed(elapsed))}</div>`;
    }

    function renderUserStep(step, t0) {
        return `
            <article class="trace-event trace-event--user" id="step-${step.step_id}">
                <aside class="trace-marker">
                    <div class="trace-marker-role">user</div>
                    ${markerTime(step, t0)}
                </aside>
                <div class="trace-event-body">
                    <div class="trace-block trace-block--text">
                        <div class="trace-md">${renderDocument(step.message)}</div>
                    </div>
                </div>
            </article>`;
    }

    function renderThought(text) {
        const words = String(text).trim().split(/\s+/).filter(Boolean).length;
        return `
            <details class="trace-thinking" data-thought="1">
                <summary>Thought <span class="trace-thinking-meta">${words} words</span></summary>
                <div class="trace-md trace-thinking-body">${renderDocument(text)}</div>
            </details>`;
    }

    function renderToolCall(step, call) {
        const fn = call.function_name || 'tool';
        const args = call.arguments;
        const label = truncateLine(toolCallLabel(fn, args), 96);
        const output = observationFor(state.trace, call.tool_call_id);
        const tooLong = output.length > OUT_COLLAPSE_CHARS;
        const key = `${step.step_id}:${call.tool_call_id}`;
        const idTail = call.tool_call_id ? String(call.tool_call_id).slice(-8) : '';
        return `
            <details class="trace-tool" data-call-key="${escHtml(key)}" data-overlong="${tooLong ? '1' : '0'}">
                <summary>
                    <span class="trace-tool-kicker">tool</span>
                    <span class="trace-tool-name">${escHtml(fn)}</span>
                    <code class="trace-tool-label">${escHtml(label)}</code>
                    ${idTail ? `<span class="trace-tool-id">${escHtml(idTail)}</span>` : ''}
                </summary>
                <div class="trace-tool-args">
                    <div class="trace-tool-sec">Input</div>
                    ${renderToolArgs(fn, args)}
                </div>
                <div class="trace-tool-out">
                    <div class="trace-tool-sec">Output</div>
                    <div class="trace-tool-out-body" data-call-key="${escHtml(key)}">${renderToolOutput(output)}</div>
                    ${tooLong ? `
                        <button class="trace-expand-btn" type="button" data-call-key="${escHtml(key)}">Show full output</button>` : ''}
                </div>
            </details>`;
    }

    function renderMetrics(m) {
        const items = [
            ['in', fmtTokens(m.prompt_tokens)],
            ['cached', fmtTokens(m.cached_tokens)],
            ['out', fmtTokens(m.completion_tokens)]
        ];
        return `
            <div class="trace-metrics">
                ${items.map(([k, v]) => `<span class="trace-metric"><span>${k}</span><strong>${escHtml(v)}</strong></span>`).join('')}
            </div>`;
    }

    function renderAgentStep(step, t0, turnNum) {
        const blocks = [];
        if (step.reasoning_content) blocks.push(renderThought(step.reasoning_content));
        if (step.message) {
            blocks.push(`<div class="trace-block trace-block--text"><div class="trace-md">${renderDocument(step.message)}</div></div>`);
        }
        for (const call of (step.tool_calls || [])) {
            blocks.push(renderToolCall(step, call));
        }
        const metrics = step.metrics ? renderMetrics(step.metrics) : '';

        return `
            <article class="trace-event trace-event--agent" id="step-${step.step_id}">
                <aside class="trace-marker">
                    <div class="trace-turn-num">${turnNum}</div>
                    ${markerTime(step, t0)}
                </aside>
                <div class="trace-event-body">
                    ${blocks.join('\n')}
                    ${metrics}
                </div>
            </article>`;
    }

    function renderTimeline() {
        const trace = state.trace;
        const t0 = new Date(trace.steps[0].timestamp);
        let turnNum = 0;
        const html = trace.steps.map(step => {
            if (step.source === 'user') return renderUserStep(step, t0);
            turnNum++;
            return renderAgentStep(step, t0, turnNum);
        }).join('\n');
        mount.querySelector('.trace-events').innerHTML = html;
    }

    /* ---- Left rail --------------------------------------------------- */

    function renderRail(trace, fileName, stats) {
        const fm = trace.final_metrics || {};
        const agent = trace.agent || {};
        const task = taskFor(fileName);
        const agentName = agent.name || '—';
        const meta = TRACE_META[traceSlug(fileName)] || {};
        const cost = fm.total_cost_usd !== undefined ? fmtCost(fm.total_cost_usd) : null;

        const rows = [
            ['agent', agentName],
            ['model', agent.model_name || '—'],
            meta.effort ? ['effort', meta.effort] : null,
            task ? ['task', task.title] : null,
            ['trace format', trace.schema_version || '—'],
            ['duration', fmtDuration(stats.durationMs)],
            ['steps', fmtNum(stats.steps)],
            ['tool calls', fmtNum(stats.toolCalls)],
            ['thoughts', fmtNum(stats.thoughts)],
            ['prompt tokens', fmtNum(fm.total_prompt_tokens)],
            ['cached tokens', fmtNum(fm.total_cached_tokens)],
            ['completion tokens', fmtNum(fm.total_completion_tokens)],
            cost ? ['cost', cost] : null
        ];

        const verdict = meta.passed === true || meta.passed === false
            ? (meta.passed ? '✅ Passed' : '❌ Failed') : null;
        if (verdict) rows.push(['result', verdict]);

        const detailRows = rows.filter(Boolean);

        /* Published tasks link to their task page; catalog-only tasks
           (AUTORLBENCH_TRACE_TASKS) have no page yet — link to the run
           catalog instead. */
        const published = (window.AUTORLBENCH_TASKS || []).some(t => t.id === (task && task.id));
        const backHref = !task ? 'index.html'
            : published ? `task.html?id=${encodeURIComponent(task.id)}`
            : 'traces.html';

        return `
            <a class="trace-back" href="${escHtml(backHref)}">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
                Back to ${!task ? 'home' : published ? 'task' : 'traces'}
            </a>
            <div class="trace-card">
                <h2 class="trace-summary-title">${escHtml(traceTitle(trace))}</h2>
                <div class="trace-summary-sub">${escHtml(agentName)} · ${escHtml(agent.model_name || '—')}</div>

                <div class="trace-score">
                    <div class="trace-score-big">${escHtml(fmtDuration(stats.durationMs))}</div>
                    <div class="trace-score-sub">${escHtml(fmtNum(stats.steps))} steps · ${cost ? escHtml(cost) : '—'}</div>
                </div>

                <button class="trace-details-toggle" type="button" aria-expanded="true">
                    <span>Run details</span><span class="trace-details-caret" aria-hidden="true" style="transform: rotate(90deg)">›</span>
                </button>
                <dl class="trace-summary-stats open">
                    ${detailRows.map(([k, v]) => `<dt>${escHtml(k)}</dt><dd>${escHtml(v)}</dd>`).join('')}
                </dl>

                <div class="trace-summary-footer">
                    <a class="trace-btn trace-btn--wide" href="${escHtml(fileName)}" download>Download trace</a>
                </div>
            </div>`;
    }

    /* ---- Shell -------------------------------------------------------- */

    function renderShell(trace, fileName) {
        const stats = traceStats(trace);

        mount.innerHTML = `
            <div class="trace-layout">
                <aside class="trace-rail">${renderRail(trace, fileName, stats)}</aside>
                <main class="trace-main trace-viewport">
                    <header class="trace-toolbar">
                        <div class="trace-toolbar-count">
                            ${escHtml(fmtNum(stats.steps))} steps · ${escHtml(fmtNum(stats.toolCalls))} tool calls · ${escHtml(fmtNum(stats.thoughts))} thoughts
                        </div>
                    </header>
                    <div class="trace-events"></div>
                </main>
            </div>`;

        renderTimeline();
        bindRail();

        /* The site footer travels with the trace content: it appears only
           after the event thread is scrolled to the very end. */
        adoptFooter(mount.querySelector('.trace-viewport'));
    }

    /* ---- Output expansion -------------------------------------------- */

    /* Expand or clip one tool output in place. */
    function setOutputExpanded(details, expanded) {
        const key = details.dataset.callKey;
        state.expanded[key] = expanded;
        const call = findCallByKey(key);
        const text = call ? observationFor(state.trace, call.tool_call_id) : '';
        const body = details.querySelector('.trace-tool-out-body');
        const shown = expanded ? text : text.slice(0, OUT_COLLAPSE_CHARS);
        body.innerHTML = text ? preBlock(shown) : '<div class="trace-note">No output</div>';
        const btn = details.querySelector('.trace-expand-btn');
        if (btn) btn.textContent = expanded ? 'Show less' : 'Show full output';
    }

    function toggleOutput(btn) {
        const key = btn.dataset.callKey;
        const details = btn.closest('details.trace-tool');
        setOutputExpanded(details, !state.expanded[key]);
    }

    function findCallByKey(key) {
        const stepId = key.split(':')[0];
        for (const step of state.trace.steps) {
            if (String(step.step_id) === String(stepId)) {
                for (const call of (step.tool_calls || [])) {
                    if (`${step.step_id}:${call.tool_call_id}` === key) return call;
                }
            }
        }
        return null;
    }

    /* ---- Events ------------------------------------------------------- */

    function bindRail() {
        const toggle = mount.querySelector('.trace-details-toggle');
        const stats = mount.querySelector('.trace-summary-stats');
        if (toggle && stats) {
            toggle.addEventListener('click', () => {
                const open = stats.classList.toggle('open');
                toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
                toggle.querySelector('.trace-details-caret').style.transform = open ? 'rotate(90deg)' : '';
            });
        }
    }

    function bindEvents() {
        mount.addEventListener('click', event => {
            const btn = event.target.closest('.trace-expand-btn');
            if (btn) toggleOutput(btn);
        });
    }

    /* ---- Markdown (compact renderer for trace text) ------------------ */

    function inlineMd(s) {
        return escHtml(s)
            .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>');
    }

    function renderMd(text) {
        if (!text) return '';
        const lines = String(text).replace(/\r\n/g, '\n').split('\n');
        const out = [];
        let para = [];
        let list = null;
        let quote = [];

        function flushPara() {
            if (para.length) { out.push(`<p>${inlineMd(para.join(' '))}</p>`); para = []; }
        }
        function flushList() {
            if (list) {
                out.push(`<${list.tag}>${list.items.map(i => `<li>${inlineMd(i)}</li>`).join('')}</${list.tag}>`);
                list = null;
            }
        }
        function flushQuote() {
            if (quote.length) { out.push(`<blockquote>${renderMd(quote.join('\n'))}</blockquote>`); quote = []; }
        }
        function flushAll() { flushPara(); flushList(); flushQuote(); }

        for (const line of lines) {
            const t = line.trim();

            if (t === '') { flushAll(); continue; }

            // Pre-extracted fenced code block: emit verbatim, never wrapped in <p>.
            if (t.startsWith('\x00C')) { flushAll(); out.push(t); continue; }

            // Stray unclosed fence: swallow the marker line.
            if (/^```\w*$/.test(t)) { flushAll(); continue; }

            const heading = t.match(/^(#{1,4})\s+(.*)$/);
            if (heading) {
                flushAll();
                const level = heading[1].length + 1;
                out.push(`<h${level}>${inlineMd(heading[2])}</h${level}>`);
                continue;
            }

            if (/^(-{3,}|\*{3,})$/.test(t)) { flushAll(); out.push('<hr>'); continue; }

            if (line.startsWith('>')) {
                flushPara(); flushList();
                quote.push(t.replace(/^>\s?/, ''));
                continue;
            }

            const bullet = t.match(/^[-*]\s+(.*)$/);
            if (bullet) {
                flushPara(); flushQuote();
                if (!list || list.tag !== 'ul') { flushList(); list = { tag: 'ul', items: [] }; }
                list.items.push(bullet[1]);
                continue;
            }

            const ordered = t.match(/^\d+\.\s+(.*)$/);
            if (ordered) {
                flushPara(); flushQuote();
                if (!list || list.tag !== 'ol') { flushList(); list = { tag: 'ol', items: [] }; }
                list.items.push(ordered[1]);
                continue;
            }

            flushList(); flushQuote();
            para.push(t);
        }
        flushAll();

        return out.join('\n');
    }

    /* Fenced code blocks carry multiline content, so run a first pass that
       extracts them before the line-based parser above sees the body. */
    function renderDocument(text) {
        if (!text) return '';
        const blocks = [];
        const source = String(text).replace(/```(\w*)\n([\s\S]*?)```/g, function (all, lang, code) {
            void all; void lang;
            const idx = blocks.length;
            blocks.push(`<pre class="trace-pre trace-code"><code>${escHtml(code.replace(/\n$/, ''))}</code></pre>`);
            return `\x00C${idx}\x00`;
        });
        let html = renderMd(source);
        blocks.forEach((block, i) => { html = html.replace(`\x00C${i}\x00`, block); });
        return html;
    }

    /* ---- Loading ------------------------------------------------------ */

    const mount = document.getElementById('trace-root');

    async function loadTrace(fileName) {
        const res = await fetch(fileName);
        if (!res.ok) throw new Error('failed to load ' + fileName);
        return res.json();
    }

    /* Move the site footer into the given container so it appears after
       the trace content instead of below the fixed-height console. */
    function adoptFooter(container) {
        const footer = document.querySelector('body > footer');
        if (footer && container) container.appendChild(footer);
    }

    async function init() {
        if (!mount) return;
        const slugs = Object.keys(TRACES);
        if (!slugs.length) {
            mount.innerHTML = '<div class="task-panel-note">No traces available yet.</div>';
            adoptFooter(mount);
            return;
        }
        const wanted = new URLSearchParams(window.location.search).get('trace');
        const slug = wanted && TRACES[wanted] ? wanted : slugs[0];
        const fileName = TRACES[slug];
        mount.innerHTML = `<div class="task-loading"><span class="task-spinner" aria-hidden="true"></span>Loading trace…</div>`;
        try {
            const trace = await loadTrace(fileName);
            indexObservations(trace);
            state.trace = trace;
            state.fileName = fileName;
            renderShell(trace, fileName);
        } catch (err) {
            mount.innerHTML = `<div class="task-panel-note">Could not load the trace (${escHtml(err.message)}).</div>`;
            adoptFooter(mount);
        }
    }

    document.addEventListener('DOMContentLoaded', () => { init().then(bindEvents); });
})();
