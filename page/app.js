const TASKS = window.AUTORLBENCH_TASKS || [];

const TASK_CATEGORIES = [
    'Configuration',
    'Reward',
    'Data',
    'Environment',
    'Algorithm',
    'Infrastructure'
];

const CATEGORY_BLURBS = {
    Configuration: 'Bad hyperparameters and training settings.',
    Reward: 'Rewards that miss or punish the right behavior.',
    Data: 'Input, format, or distribution / difficulty issues.',
    Environment: 'Unstable or mis-specified training environments.',
    Algorithm: 'Algorithm design flaws, unstable updates, and exploration failures.',
    Infrastructure: 'Runtime, engine, and systems failures.'
};

// Tasks per category: the samples in task-data.js. Collection is ongoing.
const CATEGORY_COUNTS = Object.fromEntries(
    TASK_CATEGORIES.map(family => [family, TASKS.filter(task => task.family === family).length])
);

const ISSUE_RESULTS = [
    { model: 'GPT-6-Astra', agent: 'Codex', score: 0.64 },
    { model: 'GPT-6.1-Sol', agent: 'Codex', score: 0.57 },
    { model: 'Claude-Fable-5.1', agent: 'Claude Code', score: 0.86 },
    // { model: 'Claude-Opus-5.5', agent: 'Claude Code', score: 0.43 },
    { model: 'GLM-5.3', agent: 'Claude Code', score: 0.64 },
    { model: 'Kimi-K3', agent: 'Claude Code', score: 0.50 },
    { model: 'DeepSeek-V4-Pro', agent: 'Claude Code', score: 0.29 },
    { model: 'Gemini-4-Argon', agent: 'Claude Code', score: 0.43 },
    { model: 'Qwen3.8-Max', agent: 'Qwen Code', score: 0.57 },
    { model: 'Grok-4.7', agent: 'Grok Build', score: 0.79 }
];

function tasksForFamily(family) {
    return TASKS.filter(task => task.family === family);
}

function renderCategoryStrip() {
    const mount = document.getElementById('category-strip');
    if (!mount) return;

    const categoryCount = document.getElementById('category-count');
    const activeCount = document.getElementById('active-task-count');
    if (categoryCount) categoryCount.textContent = String(TASK_CATEGORIES.length);
    if (activeCount) activeCount.textContent = '10'; // total tasks in the benchmark; only one sample per category is listed below
    const heroTasks = document.getElementById('hero-task-count');
    if (heroTasks) heroTasks.textContent = String(TASKS.length);

    mount.innerHTML = TASK_CATEGORIES.map((family, index) => {
        const count = CATEGORY_COUNTS[family] || 0; void count; // kept for leaderboard use
        return `
            <div class="cat-card">
                <div class="cat-card-inner">
                    <div class="cat-card-face cat-card-front">
                        <img src="images/domain-${family.toLowerCase()}.png" alt="" loading="lazy">
                        <span class="cat-card-label">${family}</span>
                    </div>
                    <div class="cat-card-face cat-card-back">
                        <span class="cat-card-label">${family}</span>
                        <span class="cat-card-blurb">${CATEGORY_BLURBS[family] || ''}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Duplicate cards for seamless infinite scroll
    mount.innerHTML += mount.innerHTML;
    startCategoryScroll(mount);
}

function startCategoryScroll(track) {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) return;
    let raf, last = null, offset = 0;
    const speed = 0.025; // px per ms — very slow, adjustable
    function step(ts) {
        if (last !== null) offset += speed * (ts - last);
        last = ts;
        const halfW = track.scrollWidth / 2;
        if (offset >= halfW) offset -= halfW;
        track.style.transform = `translateX(-${offset}px)`;
        raf = requestAnimationFrame(step);
    }
    raf = requestAnimationFrame(step);
    // Pause on hover/focus
    track.closest('.category-strip-wrap').addEventListener('mouseenter', () => cancelAnimationFrame(raf));
    track.closest('.category-strip-wrap').addEventListener('mouseleave', () => { last = null; raf = requestAnimationFrame(step); });
}

function renderTaskCatalog() {
    const mount = document.getElementById('sample-cards');
    if (!mount) return;
    mount.innerHTML = TASK_CATEGORIES.map(family => {
        const task = tasksForFamily(family)[0];
        if (!task) {
            return `
                <div class="task-card is-empty">
                    <span class="task-tag">${family}</span>
                    <h3>Sample coming soon</h3>
                </div>
            `;
        }
        const img = `images/task-${task.family.toLowerCase()}.jpeg`;
        return `
            <a href="task.html?id=${encodeURIComponent(task.id)}" class="task-card">
                <div class="task-card-image">
                    <img src="${img}" alt="${task.family} task illustration" loading="lazy">
                </div>
                <div class="task-card-body">
                    <h3>${task.title}</h3>
                    <span class="task-tag">${task.family}</span>
                    <p>${task.brief || task.issue}</p>
                </div>
            </a>
        `;
    }).join('');
}

const MEDALS = { 1: '🥇', 2: '🥈', 3: '🥉' };

function renderLeaderboard() {
    const mount = document.querySelector('.leaderboard-list');
    if (!mount) return;
    const rows = ISSUE_RESULTS
        .map((result, index) => ({ ...result, index }))
        .sort((a, b) => b.score - a.score || a.index - b.index);
    let displayedRank = 0;
    let prevScore = null;
    let prevRank = 0;
    mount.innerHTML = rows.map((result, position) => {
        if (result.score !== prevScore) {
            displayedRank = position + 1;
            prevRank = displayedRank;
            prevScore = result.score;
        } else {
            displayedRank = prevRank;
        }
        return `
            <div class="lb-row">
                ${MEDALS[displayedRank]
                    ? `<div class="lb-rank is-medal" role="img" aria-label="Rank ${displayedRank}">${MEDALS[displayedRank]}</div>`
                    : `<div class="lb-rank">${displayedRank}</div>`}
                <div class="lb-model-cell"><span class="lb-model-name">${result.model}</span></div>
                <div class="lb-agent-cell"><span class="lb-agent-name">${result.agent}<span class="lb-max-tag">MAX</span></span></div>
                <div class="lb-scores">
                    <div class="lb-score-item">
                        <div class="lb-score-bar"><div class="lb-score-fill" style="width:${result.score * 100}%"></div></div>
                    </div>
                    <div class="lb-issue-score">${result.score.toFixed(2)}</div>
                </div>
            </div>
        `}).join('');
}

function initLeaderboard() {}

function initScrollMotion() {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    document.documentElement.classList.add('motion-ready');
    const sections = document.querySelectorAll('body > section:not(.hero), body > footer');
    sections.forEach(section => section.classList.add('scroll-reveal'));
    const revealAll = () => sections.forEach(section => section.classList.add('is-visible'));
    requestAnimationFrame(() => {
        sections.forEach(section => {
            const rect = section.getBoundingClientRect();
            if (rect.top < window.innerHeight * 1.15 && rect.bottom > 0) section.classList.add('is-visible');
        });
    });
    if (!('IntersectionObserver' in window)) return revealAll();
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
        });
    }, { threshold: 0.08, rootMargin: '0px 0px -8% 0px' });
    sections.forEach(section => observer.observe(section));
    window.setTimeout(revealAll, 1200);
}

document.addEventListener('DOMContentLoaded', () => {
    renderCategoryStrip();
    renderTaskCatalog();
    renderLeaderboard();
    initLeaderboard();
    initScrollMotion();
});
