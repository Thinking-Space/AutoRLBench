(function () {
    'use strict';

    /* Task detail page.
     *
     * The header (title, tags, settings table) comes from the catalog entry in
     * task-data.js, which scripts/generate_task_data.py syncs from each
     * task.toml. The tab contents (instruction / environment / verifier files)
     * are read live from the tasks/ tree of the repository. Files
     * are fetched from raw.githubusercontent.com; directory listings come from
     * the GitHub contents API, with a filename probe over the canonical task
     * layout (skills/package-task) as a fallback when the API is rate-limited.
     * When the page is served from a local checkout of the repository root, the
     * local ../tasks/ tree is used instead.
     */

    const REPO = 'Thinking-Space/AutoRLBench';
    const BRANCH = 'main';

    const tasks = window.AUTORLBENCH_TASKS || [];
    const params = new URLSearchParams(window.location.search);
    const taskId = params.get('id');
    const task = tasks.find(item => item.id === taskId);
    const mount = document.getElementById('task-detail-root');

    const taskDir = `tasks/${taskId}`;
    const rawUrl = path => `https://raw.githubusercontent.com/${REPO}/${BRANCH}/${path}`;
    const blobUrl = path => `https://github.com/${REPO}/blob/${BRANCH}/${encodeURI(path)}`;
    const treeUrl = path => `https://github.com/${REPO}/tree/${BRANCH}/${encodeURI(path)}`;
    const apiDirUrl = path => `https://api.github.com/repos/${REPO}/contents/${encodeURI(path)}?ref=${BRANCH}`;

    /* '' fetches from the raw CDN; '../' from a local checkout of the repo
       root (local dev only, resolved by detectBase). */
    let rawBase = '';

    const textCache = new Map();
    const dirCache = new Map();

    /* Canonical task layout — probed directly when the contents API is
       unavailable. Keys are paths relative to the task directory. */
    const PROBE_FILES = {
        'environment': ['prepare.sh', 'docker-compose.yaml', 'data_manifest.json', 'source_manifest.json', 'build_index.py'],
        'evaluations/issue/verifier': ['judge_prompt.md', 'judge.py'],
        'evaluations/solution/verifier': ['README.md', 'gold.yaml', 'tests/grade_submission.py']
    };

    const FILE_RANK = ['README.md', 'instruction.md', 'judge_prompt.md', 'judge.py', 'prepare.sh', 'docker-compose.yaml', 'test.sh', 'gold.yaml'];

    const TABS = [
        { key: 'instruction', label: 'Instruction' },
        { key: 'environment', label: 'Environment' },
        { key: 'issue-verifier', label: 'Issue Verifier' },
        { key: 'solution-verifier', label: 'Solution Verifier', requiresSolution: true }
    ];

    /* Reward-visualization figures, generated per task from the recorded run
       artifacts by scripts/generate_reward_charts.py (page/scripts/). Keyed by
       task id; the file lives at page/images/rewards/<file>.svg. */
    const REWARD_FIGS = {
        'algorithm/verl_grpo_uid_after_repeat': {
            file: 'algorithm-verl_grpo_uid_after_repeat',
            alt: 'Training reward per step for the issue run'
        },
        'reward/slime_retool_length_reward_bonus': {
            file: 'reward-slime_retool_length_reward_bonus',
            alt: 'Logged eval score per checkpoint, including the length bonus'
        },
        'data/verl_taco_difficulty_mismatch': {
            file: 'data-verl_taco_difficulty_mismatch',
            alt: 'Training reward per step'
        },
        'environment/search_r1_zero_query_embedding': {
            file: 'environment-search_r1_zero_query_embedding',
            alt: 'Held-out exact match over training'
        },
        'infrastructure/verl_moe_token_normalization': {
            file: 'infrastructure-verl_moe_token_normalization',
            alt: 'Held-out accuracy per checkpoint, defective and repaired'
        },
        'configuration/verl_aime25_single_sample_protocol': {
            file: 'configuration-verl_aime25_single_sample_protocol',
            alt: 'Accuracy of each of the 32 repeated evaluations of the same checkpoint'
        }
    };

    const state = {
        hasSolution: false,
        activeTab: null,
        tabs: {} // key -> { entries: [{path, label, text}], selected, missing, error }
    };

    function escHtml(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    /* Prefer the local ../tasks/ tree when the page is served from a checkout
       of the repository root; on GitHub Pages it 404s and we fall back to the
       raw CDN. */
    async function detectBase() {
        try {
            const res = await fetch(`../tasks/${taskId}/task.toml`);
            if (res.ok) rawBase = '../';
        } catch (err) { /* keep CDN base */ }
    }

    async function fetchText(path) {
        if (textCache.has(path)) return textCache.get(path);
        let value = null;
        try {
            const res = await fetch(rawBase ? rawBase + path : rawUrl(path));
            if (res.ok) value = await res.text();
        } catch (err) { /* offline or blocked */ }
        textCache.set(path, value);
        return value;
    }

    /* List the files of a task directory. Prefers the GitHub contents API
       (exact, handles unexpected files), falls back to probing the canonical
       file names on the raw CDN when the API is unavailable. */
    async function listDir(dir) {
        if (dirCache.has(dir)) return dirCache.get(dir);
        let files = null;

        if (!rawBase) {
            try {
                const res = await fetch(apiDirUrl(dir));
                if (res.ok) {
                    const entries = await res.json();
                    files = [];
                    for (const entry of entries) {
                        if (entry.type === 'file') {
                            if (entry.name === '.gitkeep' || entry.size === 0) continue;
                            files.push({ path: entry.path, label: entry.path.slice(dir.length + 1) });
                        } else if (entry.type === 'dir') {
                            files.push(...await listDir(entry.path));
                        }
                    }
                }
            } catch (err) { /* fall through to probing */ }
        }

        if (files === null) {
            const relDir = dir.slice(taskDir.length + 1);
            files = [];
            for (const name of (PROBE_FILES[relDir] || [])) {
                const text = await fetchText(`${dir}/${name}`);
                if (text !== null) files.push({ path: `${dir}/${name}`, label: name });
            }
        }

        files.sort((a, b) => {
            const ra = FILE_RANK.indexOf(a.label.split('/').pop());
            const rb = FILE_RANK.indexOf(b.label.split('/').pop());
            return (ra === -1 ? 99 : ra) - (rb === -1 ? 99 : rb) || a.path.localeCompare(b.path);
        });

        dirCache.set(dir, files);
        return files;
    }

    function inlineMd(s) {
        return escHtml(s)
            .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>');
    }

    function renderMd(text) {
        if (!text) return '';
        const lines = text.replace(/\r\n/g, '\n').split('\n');
        const out = [];
        let para = [];
        let list = null; // { tag, items: [] }
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
            blocks.push(`<pre class="task-code"><code>${escHtml(code.replace(/\n$/, ''))}</code></pre>`);
            return `\x00C${idx}\x00`;
        });
        let html = renderMd(source);
        blocks.forEach((block, i) => { html = html.replace(`\x00C${i}\x00`, block); });
        return html;
    }

    function renderFile(path, text) {
        if (text === null) {
            return `<div class="task-panel-note">Could not load this file from the repository. <a href="${blobUrl(path)}" target="_blank" rel="noopener noreferrer">View it on GitHub ↗</a></div>`;
        }
        if (/\.md$/i.test(path)) return `<div class="task-md">${renderDocument(text)}</div>`;
        return `<pre class="task-code task-code--file"><code>${escHtml(text)}</code></pre>`;
    }

    async function init() {
        if (!mount) return;
        if (!task) {
            mount.innerHTML = '<div class="task-not-found"><h1>Task not found</h1><p>Return to the <a href="index.html#tasks">task catalog</a>.</p></div>';
            return;
        }

        await detectBase();

        state.hasSolution = await fetchText(`${taskDir}/evaluations/solution/instruction.md`) !== null;

        renderShell();
        const available = TABS.filter(t => !t.requiresSolution || state.hasSolution);
        const requested = params.get('tab');
        activateTab(available.some(t => t.key === requested) ? requested : 'instruction');
    }

    function renderShell() {
        const domain = task.family.toLowerCase();
        const tags = task.tags || [];
        const title = task.title;

        document.title = `${title} - AutoRLBench`;

        const settings = [
            ['Model', task.model || '—'],
            ['Dataset', task.dataset || '—'],
            ['Framework', task.framework || '—'],
            ['Image', task.image || '—'],
            ['Created', task.created || '—'],
            ['License', task.license || '—']
        ];
        const fig = REWARD_FIGS[taskId] || null;

        mount.innerHTML = `
            <div class="task-breadcrumb"><a href="index.html#tasks">Tasks</a><span>/</span><span>${escHtml(task.family)}</span><span>/</span><span>${escHtml(taskId.split('/')[1])}</span></div>
            <header class="task-page-header">
                <div class="task-header-top">
                    <div class="task-header-tags">
                        <span class="task-chip-domain">${escHtml(domain)}</span>
                        ${tags.map(tag => `<span>#${escHtml(tag)}</span>`).join('')}
                    </div>
                    <a class="task-github-btn" href="${treeUrl(taskDir)}" target="_blank" rel="noopener noreferrer">
                        <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3.3-.4 6.8-1.6 6.8-7A5.4 5.4 0 0 0 19.4 4 5 5 0 0 0 19.3.5S18.2.1 15 1.8a13.4 13.4 0 0 0-7 0C4.8.1 3.7.5 3.7.5A5 5 0 0 0 3.6 4a5.4 5.4 0 0 0-1.4 3.7c0 5.4 3.5 6.6 6.8 7A4.8 4.8 0 0 0 8 18v4"/><path d="M8 19c-3 .9-3-1.5-4.2-2"/></svg>
                        <span>View on GitHub</span>
                    </a>
                </div>
                <h1>${escHtml(title)}</h1>
                <p class="task-issue-text">${escHtml(task.issue)}</p>
                <div class="task-header-grid">
                    <div class="task-settings">
                        ${settings.map(([label, value]) => `<div><span>${label}</span><strong title="${escHtml(value)}">${escHtml(value)}</strong></div>`).join('')}
                    </div>
                    ${fig ? `
                    <figure class="task-reward-fig">
                        <img src="images/rewards/${fig.file}.svg" alt="${escHtml(fig.alt)}" loading="lazy" width="560" height="280">
                    </figure>` : ''}
                </div>
            </header>

            <div class="task-content-section">
                <div class="task-tabs" role="tablist" aria-label="Task content">
                    ${TABS.filter(t => !t.requiresSolution || state.hasSolution).map(t => `
                        <button class="task-tab" role="tab" type="button" data-tab="${t.key}" aria-selected="false" aria-controls="task-panel-${t.key}">${t.label}</button>
                    `).join('')}
                </div>
                <div class="task-panels">
                    ${TABS.filter(t => !t.requiresSolution || state.hasSolution).map(t => `
                        <div class="task-tab-panel" role="tabpanel" id="task-panel-${t.key}" data-panel="${t.key}" hidden></div>
                    `).join('')}
                </div>
            </div>
        `;

        mount.querySelector('.task-tabs').addEventListener('click', event => {
            const button = event.target.closest('.task-tab');
            if (button) activateTab(button.dataset.tab);
        });
        mount.querySelector('.task-tabs').addEventListener('keydown', event => {
            if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
            const buttons = Array.from(mount.querySelectorAll('.task-tab'));
            const current = buttons.findIndex(b => b.dataset.tab === state.activeTab);
            if (current === -1) return;
            const next = (current + (event.key === 'ArrowRight' ? 1 : buttons.length - 1)) % buttons.length;
            buttons[next].focus();
            activateTab(buttons[next].dataset.tab);
        });
        mount.addEventListener('click', event => {
            const chip = event.target.closest('.task-file-chip');
            if (chip && chip.closest('.task-tab-panel')) {
                selectFile(chip.closest('.task-tab-panel').dataset.panel, Number(chip.dataset.idx));
            }
        });
    }

    function activateTab(key) {
        state.activeTab = key;
        mount.querySelectorAll('.task-tab').forEach(button => {
            const isActive = button.dataset.tab === key;
            button.classList.toggle('is-active', isActive);
            button.setAttribute('aria-selected', isActive ? 'true' : 'false');
        });
        mount.querySelectorAll('.task-tab-panel').forEach(panel => { panel.hidden = panel.dataset.panel !== key; });

        const url = new URL(window.location.href);
        url.searchParams.set('tab', key);
        history.replaceState(null, '', url);

        if (!state.tabs[key]) loadTab(key).then(() => renderTab(key));
    }

    function panelEl(key) {
        return mount.querySelector(`.task-tab-panel[data-panel="${key}"]`);
    }

    function relDirFor(key) {
        if (key === 'instruction') return 'evaluations';
        if (key === 'environment') return 'environment';
        if (key === 'issue-verifier') return 'evaluations/issue/verifier';
        return 'evaluations/solution/verifier';
    }

    async function loadTab(key) {
        const entry = { entries: [], selected: 0, missing: false, error: false };
        state.tabs[key] = entry;

        if (key === 'instruction') {
            const issuePath = `${taskDir}/evaluations/issue/instruction.md`;
            const solutionPath = `${taskDir}/evaluations/solution/instruction.md`;
            const [issue, solution] = await Promise.all([fetchText(issuePath), fetchText(solutionPath)]);
            if (issue !== null) entry.entries.push({ path: issuePath, label: 'Issue', text: issue });
            if (solution !== null) entry.entries.push({ path: solutionPath, label: 'Solution', text: solution });
            if (!entry.entries.length) entry.error = true;
            return;
        }

        const dir = `${taskDir}/${relDirFor(key)}`;
        try {
            const files = await listDir(dir);
            entry.entries = files.map(f => ({ path: f.path, label: f.label, text: null }));
            if (!entry.entries.length) entry.missing = true;
        } catch (err) {
            entry.error = true;
        }
    }

    function renderTab(key) {
        const entry = state.tabs[key];
        const panel = panelEl(key);
        if (!panel || !entry) return;

        if (entry.error) {
            panel.innerHTML = '<div class="task-panel-note">Could not load this section from the repository.</div>';
            return;
        }
        if (entry.missing) {
            panel.innerHTML = '<div class="task-panel-note">This section is not available for this task yet — it is being prepared.</div>';
            return;
        }

        panel.innerHTML = `
            <div class="task-panel-toolbar">
                <div class="task-file-bar" role="tablist" aria-label="Files">
                    ${entry.entries.map((e, idx) => `
                        <button class="task-file-chip${idx === entry.selected ? ' is-active' : ''}" role="tab" type="button" data-idx="${idx}" aria-selected="${idx === entry.selected}">${escHtml(e.label)}</button>
                    `).join('')}
                </div>
                <a class="task-file-github-link" href="${blobUrl(entry.entries[entry.selected].path)}" target="_blank" rel="noopener noreferrer" data-github-link>View file on GitHub ↗</a>
            </div>
            <div class="task-panel-doc" data-doc></div>
        `;
        renderSelectedFile(key);
    }

    function selectFile(key, idx) {
        const entry = state.tabs[key];
        if (!entry || !entry.entries[idx] || idx === entry.selected) return;
        entry.selected = idx;
        const chips = panelEl(key).querySelectorAll('.task-file-chip');
        chips.forEach(chip => {
            const isActive = Number(chip.dataset.idx) === idx;
            chip.classList.toggle('is-active', isActive);
            chip.setAttribute('aria-selected', isActive ? 'true' : 'false');
        });
        renderSelectedFile(key);
    }

    async function renderSelectedFile(key) {
        const entry = state.tabs[key];
        const panel = panelEl(key);
        const doc = panel && panel.querySelector('[data-doc]');
        const githubLink = panel && panel.querySelector('[data-github-link]');
        if (!entry || !doc) return;

        const file = entry.entries[entry.selected];

        // Update GitHub link href
        if (githubLink) {
            githubLink.href = blobUrl(file.path);
        }

        doc.innerHTML = `<div class="task-loading"><span class="task-spinner" aria-hidden="true"></span>Loading…</div>`;

        if (file.text === null) file.text = await fetchText(file.path);
        const loading = doc.querySelector('.task-loading');
        if (loading) loading.remove();
        doc.insertAdjacentHTML('beforeend', renderFile(file.path, file.text));
    }

    document.addEventListener('DOMContentLoaded', init);
})();
