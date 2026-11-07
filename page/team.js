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

function institutionNames() {
    const base = typeof INSTITUTIONS === 'undefined' ? [] : INSTITUTIONS.slice();
    const inserts = typeof INSTITUTION_INSERTS === 'undefined' ? [] : INSTITUTION_INSERTS;
    for (const { after, name } of inserts) {
        const at = base.indexOf(after);
        if (at !== -1 && !base.includes(name)) base.splice(at + 1, 0, name);
    }
    return base;
}

function renderInstitutions() {
    const grid = document.getElementById('institution-grid');
    const names = institutionNames();

    // Without names there is nothing to show: drop the whole section so
    // the heading never sits above an empty card.
    if (!names.length) {
        grid?.closest('section')?.remove();
        return;
    }

    if (grid) grid.innerHTML = names.map(name => {
        const hasCell = typeof INSTITUTION_LOGOS !== 'undefined' && Object.prototype.hasOwnProperty.call(INSTITUTION_LOGOS, name);
        // Each cell cuts its logo out of the single strip image
        // (120x60 per cell, one row).
        const logo = hasCell
            ? `<span class="institution-logo" style="--cell:${INSTITUTION_LOGOS[name]}" aria-hidden="true"></span>`
            : '';
        return `<li>${logo}<span class="institution-name">${escapeHtml(name)}</span></li>`;
    }).join('');
}

renderTeam();
renderInstitutions();
