function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function initialsOf(name) {
    const parts = String(name).trim().split(/\s+/).filter(Boolean);
    if (parts.length === 0) return '?';
    const last = parts.length > 1 ? parts[parts.length - 1][0] : '';
    return (parts[0][0] + last).toUpperCase();
}

function renderTeam() {
    const grid = document.getElementById('team-grid');
    if (!grid) return;

    grid.innerHTML = TEAM.map(person => {
        // The handle is only used to fetch the avatar; no profile link is rendered.
        const avatar = person.handle
            ? `<img src="https://github.com/${encodeURIComponent(person.handle)}.png?size=112" alt="" width="56" height="56" loading="lazy">`
            : '';
        return `
            <li class="team-card">
                <span class="team-avatar" data-initials="${escapeHtml(initialsOf(person.name))}">${avatar}</span>
                <div class="team-info">
                    <h3>${escapeHtml(person.name)}</h3>
                    ${person.affiliation ? `<p>${escapeHtml(person.affiliation)}</p>` : ''}
                </div>
            </li>
        `;
    }).join('');

    // Missing or blocked avatar: drop the <img> so the initials underneath show.
    grid.querySelectorAll('.team-avatar img').forEach(img => {
        img.addEventListener('error', () => img.remove());
    });
}

renderTeam();
