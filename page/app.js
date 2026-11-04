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

const LEADERBOARD_MODELS = [
    { model: 'GPT-6-Astra', agent: 'Codex', logo: 'openai' },
    { model: 'GPT-6.1-Sol', agent: 'Codex', logo: 'openai' },
    { model: 'Claude-Fable-5.1', agent: 'Claude Code', logo: 'anthropic' },
    { model: 'GLM-5.3', agent: 'Claude Code', logo: 'glm' },
    { model: 'DeepSeek-V4-Pro', agent: 'Claude Code', logo: 'deepseek' },
    { model: 'Kimi-K3', agent: 'Claude Code', logo: 'kimi' },
    { model: 'Kimi-K3.1-Pro', agent: 'Claude Code', logo: 'kimi' },
    { model: 'Gemini-4-Argon', agent: 'Claude Code', logo: 'googlegemini' },
    { model: 'Qwen3.8-Max', agent: 'Qwen Code', logo: 'qwen' },
    { model: 'Grok-4.7', agent: 'Grok Build', logo: 'xai' }
];

// The leaderboard and trace pages share the same pass/fail records. Preserve
// the published convention of averaging the displayed one-decimal rates.
function leaderboardResults() {
    const runs = window.AUTORLBENCH_RUNS || [];
    return LEADERBOARD_MODELS.map(model => {
        const rate = level => {
            const selected = runs.filter(run => run.model === model.model && run.level === level
                && typeof run.passed === 'boolean');
            return selected.length
                ? Number((100 * selected.filter(run => run.passed).length / selected.length).toFixed(1))
                : null;
        };
        return { ...model, issue: rate('issue'), solution: rate('solution') };
    }).filter(result => result.issue !== null && result.solution !== null);
}

function tasksForFamily(family) {
    return TASKS.filter(task => task.family === family);
}

function renderCategoryStrip() {
    const mount = document.getElementById('category-strip');
    if (!mount) return;

    const categoryCount = document.getElementById('category-count');
    const activeCount = document.getElementById('active-task-count');
    if (categoryCount) categoryCount.textContent = String(TASK_CATEGORIES.length);
    if (activeCount) activeCount.textContent = '26'; // total tasks in the benchmark; only one sample per category is listed below
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
    const rows = leaderboardResults()
        .map((result, index) => ({ ...result, overall: (result.issue + result.solution) / 2, index }))
        .sort((a, b) => b.overall - a.overall || a.index - b.index);
    let displayedRank = 0;
    let prevScore = null;
    let prevRank = 0;
    mount.innerHTML = rows.map((result, position) => {
        if (result.overall !== prevScore) {
            displayedRank = position + 1;
            prevRank = displayedRank;
            prevScore = result.overall;
        } else {
            displayedRank = prevRank;
        }
        return `
            <tr class="lb-tr">
                <td class="lb-cell-rank">
                    ${MEDALS[displayedRank]
                        ? `<span class="lb-rank is-medal" role="img" aria-label="Rank ${displayedRank}">${MEDALS[displayedRank]}</span>`
                        : `<span class="lb-rank">${displayedRank}</span>`}
                </td>
                <td class="lb-model-cell"><img class="lb-model-logo" src="images/logos/${result.logo}.png" alt="" width="18" height="18" loading="lazy"><span class="lb-model-name">${result.model}</span></td>
                <td class="lb-cell-agent"><span class="lb-agent-name">${result.agent}</span></td>
                <td class="lb-cell-effort"><span class="lb-max-tag">MAX</span></td>
                <td class="lb-cell-score"><span class="lb-score-value">${result.issue.toFixed(1)} %</span></td>
                <td class="lb-cell-score"><span class="lb-score-value">${result.solution.toFixed(1)} %</span></td>
                <td class="lb-cell-score"><span class="lb-score-value is-avg">${result.overall.toFixed(1)} %</span></td>
            </tr>
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
