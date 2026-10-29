/* Enhance static channel links when a version manifest is available. */
(() => {
    const picker = document.querySelector('[data-versions-url]');
    if (!picker || !/^https?:$/.test(location.protocol)) return;
    const manifest = new URL(picker.dataset.versionsUrl, document.baseURI);
    if (manifest.origin !== location.origin) return;
    const validVersion = value => typeof value === 'string'
        && /^[0-9]+\.[0-9]+\.[0-9]+[a-zA-Z0-9.+-]*$/.test(value);
    fetch(manifest, {credentials: 'omit'}).then(response => {
        if (!response.ok) throw new Error('Version manifest unavailable');
        return response.json();
    }).then(data => {
        if (!data || !validVersion(data.stable) || data.dev !== 'dev'
            || !Array.isArray(data.versions) || !data.versions.every(validVersion)) return;
        const select = picker.querySelector('select');
        const choices = [['stable', `Stable (${data.stable})`], ['dev', 'Development'],
            ...[...new Set(data.versions)].map(version => [version, version])];
        for (const [value, title] of choices) {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = title;
            select.append(option);
        }
        const current = picker.dataset.currentVersion;
        select.value = choices.some(([value]) => value === current) ? current : 'stable';
        const allowed = new Set(choices.map(([value]) => value));
        const root = new URL('.', manifest);
        select.addEventListener('change', () => {
            if (!allowed.has(select.value)) return;
            const target = new URL(`${select.value}/index.html`, root);
            if (target.origin === location.origin && target.pathname.startsWith(root.pathname)) {
                location.assign(target.href);
            }
        });
        picker.hidden = false;
    }).catch(() => { /* Static links remain usable offline or without a manifest. */ });
})();
