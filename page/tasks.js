(function () {
    'use strict';

    /* Tasks index (tasks.html). Lists every task in the benchmark with the
       same card idiom as the home page's sample grid, but without images.
       A search box filters by title, a category dropdown filters by failure
       category, and a level dropdown filters by evaluation level (all tasks
       support issue-level evaluation; non-issue-only tasks also support
       solution-level evaluation). Cards link to the task detail page. */

    const TASKS = window.AUTORLBENCH_TASKS || [];

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
    let level = 'all';

    function escHtml(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function taskCard(task) {
        const levels = task.tags && task.tags.includes('issue-only')
            ? '<span class="task-level-tag">Issue</span>'
            : '<span class="task-level-tag">Issue</span><span class="task-level-tag">Solution</span>';
        return `
            <a href="task.html?id=${encodeURIComponent(task.id)}" class="task-card task-card--noimage">
                <div class="task-card-body">
                    <h3>${escHtml(task.title)}</h3>
                    <div class="task-tag-row">
                        <span class="task-tag">${escHtml(task.family)}</span>
                        ${levels}
                    </div>
                    <p>${escHtml(task.brief || task.issue)}</p>
                    <div class="task-card-meta">
                        <span>${escHtml(task.framework || '—')}</span>
                        <span>${escHtml(task.model || '—')}</span>
                    </div>
                </div>
            </a>`;
    }

    function matchesLevel(task) {
        // Every task has an issue-level evaluation; only non-issue-only
        // tasks also support solution-level evaluation.
        return level === 'solution'
            ? !(task.tags && task.tags.includes('issue-only'))
            : true;
    }

    function filteredTasks() {
        const q = query.trim().toLowerCase();
        return TASKS.filter(task =>
            (category === 'all' || task.family === category)
            && matchesLevel(task)
            && (task.title.toLowerCase().includes(q)
                || String(task.brief || task.issue).toLowerCase().includes(q)));
    }

    function renderCards() {
        const mount = document.getElementById('tasks-grid');
        if (!mount) return;
        const tasks = filteredTasks();
        const count = document.getElementById('tasks-count');
        if (count) {
            count.textContent = tasks.length === TASKS.length
                ? `${TASKS.length} task${TASKS.length === 1 ? '' : 's'}`
                : `${tasks.length} of ${TASKS.length} task${TASKS.length === 1 ? '' : 's'}`;
        }
        mount.innerHTML = tasks.length
            ? tasks.map(taskCard).join('')
            : '<div class="tasks-empty">No tasks match the current search and filters.</div>';
    }

    function renderFilters() {
        const search = document.getElementById('tasks-search');
        if (search) {
            search.addEventListener('input', event => {
                query = event.target.value;
                renderCards();
            });
        }
        const select = document.getElementById('tasks-category-select');
        if (select) {
            select.addEventListener('change', event => {
                category = event.target.value;
                renderCards();
            });
        }
        const levelSelect = document.getElementById('tasks-level-select');
        if (levelSelect) {
            levelSelect.addEventListener('change', event => {
                level = event.target.value;
                renderCards();
            });
        }
    }

    function renderPage() {
        const mount = document.getElementById('tasks-root');
        if (!mount) return;

        const header = `
            <div class="tasks-header">
                <div class="task-header-tags">
                    <span>Task catalog</span>
                    <span id="tasks-count">${TASKS.length} task${TASKS.length === 1 ? '' : 's'}</span>
                    <span>${TASK_CATEGORIES.length} categories</span>
                </div>
                <h1>Tasks</h1>
                <p class="tasks-intro">Every task in the benchmark. Each task packages a real RL run that failed — the agent reads the workspace and diagnoses or repairs it. </p>
                <div class="tasks-filters">
                    <div class="tasks-search-wrap">
                        <svg class="tasks-search-icon" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
                        <input id="tasks-search" class="tasks-search" type="search" placeholder="Search tasks by name or description" autocomplete="off" aria-label="Search tasks by name or description">
                    </div>
                    <select id="tasks-category-select" class="tasks-category-select" aria-label="Filter by failure category">
                        <option value="all">All Categories</option>
                        ${TASK_CATEGORIES.map(family => `<option value="${escHtml(family)}">${escHtml(family)}</option>`).join('')}
                    </select>
                    <select id="tasks-level-select" class="tasks-category-select" aria-label="Filter by evaluation level">
                        <option value="all">All Levels</option>
                        <option value="issue">Issue</option>
                        <option value="solution">Solution</option>
                    </select>
                </div>
            </div>`;

        mount.innerHTML = header + '<div class="tasks-grid" id="tasks-grid"></div>';
        renderFilters();
        renderCards();
    }

    document.addEventListener('DOMContentLoaded', renderPage);
})();
