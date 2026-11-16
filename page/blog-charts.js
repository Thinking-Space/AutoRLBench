/* AutoRLBench article figures.
 * Every chart below is constructed from data and SVG primitives in the browser.
 * There are no image or pre-rendered SVG dependencies: the IntersectionObserver
 * only controls when the marks enter and leave their animated state. */
(() => {
    'use strict';

    const NS = 'http://www.w3.org/2000/svg';
    const REDUCE = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const C = {
        ink: '#243447', muted: '#68727c', grid: '#dce5e8', navy: '#174f86', blue: '#3d82b7',
        paleBlue: '#9ec6dc', red: '#bf2536', peach: '#f2a07e', green: '#557f68', sand: '#e6c7a7',
        warm: '#d86f58', light: '#f7f6f2', white: '#fbfaf8'
    };

    const data = {
        profile: {
            affiliations: [['THUNLP', 12], ['Tsinghua', 4], ['FusionBrain', 2], ['Brigham Young', 3], ['Johns Hopkins', 1], ['Tohoku', 2], ['UPenn', 1], ['Independent', 2], ['Fudan', 2], ['SII', 2], ['SW Jiaotong', 2], ['Duke', 1], ['SJTU', 1]],
            errors: [['Infrastructure', 12], ['Algorithm', 9], ['Configuration', 5], ['Reward', 5], ['Data', 4], ['Environment', 1]],
            difficulty: [['Very hard', 13], ['Hard', 13], ['Medium', 6], ['Easy', 3]]
        },
        models: [['GPT-6-Astra', 35.2], ['GPT-6.1-Sol', 33.3], ['Gemini-4-Argon', 29.6], ['GLM-5.3', 25.9], ['Kimi-K3.1-Pro', 24.1], ['Fable-5.1', 20.4], ['Qwen3.8-Max', 18.5], ['Grok-4.7', 16.7], ['Kimi-K3', 5.6], ['DeepSeek-V4-Pro', 1.9]],
        // Per-model [issue, solution] pass rates behind the overall bars;
        // the leaderboard tooltip reads these in the overall view and the
        // categoryByModel entries in a category view.
        modelLevels: {
            'GPT-6-Astra': [34.3, 38.9],
            'GPT-6.1-Sol': [31.4, 38.9],
            'Gemini-4-Argon': [17.1, 55.6],
            'GLM-5.3': [22.9, 33.3],
            'Kimi-K3.1-Pro': [17.1, 38.9],
            'Fable-5.1': [17.1, 27.8],
            'Qwen3.8-Max': [14.3, 27.8],
            'Grok-4.7': [17.1, 16.7],
            'Kimi-K3': [0, 16.7],
            'DeepSeek-V4-Pro': [0, 5.6]
        },
        averageLevels: [16.7, 30.0],
        modelHarness: ['Codex', 'Codex', 'Claude Code', 'Claude Code', 'Claude Code', 'Claude Code', 'Qwen Code', 'Grok Build', 'Claude Code', 'Claude Code'],
        modelDisplayNames: {
            Fable: 'Fable-5.1',
            DeepSeek: 'DeepSeek-V4-Pro',
            GLM: 'GLM-5.3',
            Astra: 'GPT-6-Astra',
            Sol: 'GPT-6.1-Sol',
            Gemini: 'Gemini-4-Argon',
            'Kimi K3': 'Kimi-K3',
            'Kimi 3.1': 'Kimi-K3.1-Pro'
        },
        leaderboardColors: ['#104680', '#317CB7', '#6DADD1', '#B6D7E8', '#E9F1F4', '#FBE3D5', '#F6B293', '#DC6D57', '#B72230', '#6D011F'],
        // Figure 2 uses the same four-step blue-to-red palette as the
        // four-stage trajectory in Figure 8: two blue outcomes followed by
        // two red outcomes.
        outcomes: [['Pass', 60, '#174f86'], ['Wrong cause', 193, '#3d82b7'], ['Wrong evidence', 33, '#d86f58'], ['Fail', 59, '#bf2536']],
        // Failed repair trajectories grouped by the primary error mode. The
        // values are percentages of the 119 failed repair trajectories and
        // are rendered as animated marks in Figure 7.
        repairErrors: [
            ['No edit', 21, '#104680'],
            ['Wrong target', 48, '#6DADD1'],
            ['Incorrect / incomplete', 30, '#DC6D57'],
            ['New Failure', 6, '#B72230'],
            ['Cheating', 11, '#6D011F']
        ],
        // Per-task counts use the same mutually exclusive outcome categories
        // as the summary bar above: Pass, Wrong cause, Wrong evidence, Fail.
        outcomeTasks: [
            ['Data relocation', 1, 2, 5, 2],
            ['Search-R1', 8, 0, 0, 2],
            ['AdamW epsilon', 0, 2, 0, 8],
            ['PiSSA reference', 2, 2, 6, 0],
            ['Rollout top-p', 1, 7, 1, 1],
            ['Critic head', 5, 5, 0, 0],
            ['PPO minibatch', 1, 7, 2, 0],
            ['Token alignment', 1, 2, 3, 4],
            ['PiSSA adapter', 1, 6, 3, 0],
            ['DAPO normalization', 1, 5, 3, 1]
        ],
        curves: [
            { title: 'LoRA sync', x: [0,5,10,15,20,25,30,35,40,45,50,55,60], a: [7.885,73.01,73.389,75.815,76.194,77.028,76.649,77.559,78.089,77.786,77.938,78.165,78.772], b: [8.036,7.733,7.506,8.567,7.733,8.188,8.036,7.809,8.264,8.34,7.582,7.809,8.188], aLabel: 'Astra', bLabel: 'Kimi K3', gate: 40, max: 85, colors: [C.navy, C.peach] },
            { title: 'PPO mini-batch', x: [0,10,20,30,40,50,60,70,80,90,100], a: [48.825,49.735,51.327,51.403,52.995,55.118,56.406,55.269,56.558,56.179,58.453], b: [47.991,50.417,49.128,41.698,34.268,19.257,11.145,13.04,12.813,2.123,2.502], aLabel: 'GLM', bLabel: 'Astra', gate: 53, max: 65, colors: [C.red, C.navy] },
            { title: 'GRPO broadcast', x: [0,50,100,150,200,250,300,350,400], a: [5.435,7.065,8.696,10.326,9.783,11.957,10.87,11.957,11.413], b: [5.978,0,0,0,0,0,0,0,0], aLabel: 'Fable', bLabel: 'Kimi K3', gate: 10, max: 15, colors: [C.blue, C.peach] },
            { title: 'AdamW epsilon', x: [0,10,20,30,40,50,60,70,80,90,100], a: [.076,48.749,52.464,55.648,58.074,56.027,57.165,58.302,56.861,59.136,58.378], b: [.076,.455,10.69,27.521,40.561,47.309,48.976,48.446,48.749,48.976,49.431], aLabel: 'Astra', bLabel: 'Kimi K3', gate: 53, max: 65, colors: [C.navy, C.peach] }
        ],
        categories: [['Configuration', 16.0, 45.0], ['Reward', 16.0, 23.3], ['Data', 10.0, 30.0], ['Environment', 0, 0], ['Algorithm', 14.4, 20.0], ['Infrastructure', 22.5, 27.1]],
        // Per model and category: [diagnosis, repair, combined] pass rates
        // over that category's published runs. The combined value keeps the
        // fig-1 category drill-down on the same metric as the overall
        // leaderboard (issue + solution passes over all runs); fig-4 reads
        // the first two entries for its paired bars.
        categoryByModel: {
            'GPT-6-Astra': [[20, 25, 22.2], [60, 66.7, 62.5], [0, 50, 16.7], [0, 0, 0], [22.2, 0, 18.2], [54.5, 50, 52.9]],
            'GPT-6.1-Sol': [[40, 25, 33.3], [20, 33.3, 25], [0, 50, 16.7], [0, 0, 0], [11.1, 0, 9.1], [63.6, 66.7, 64.7]],
            'Gemini-4-Argon': [[20, 75, 44.4], [20, 33.3, 25], [0, 50, 16.7], [0, 0, 0], [11.1, 50, 18.2], [27.3, 66.7, 41.2]],
            'GLM-5.3': [[20, 50, 33.3], [20, 0, 12.5], [25, 50, 33.3], [0, 0, 0], [11.1, 50, 18.2], [36.4, 33.3, 35.3]],
            'Kimi-K3.1-Pro': [[20, 50, 33.3], [20, 33.3, 25], [0, 50, 16.7], [0, 0, 0], [22.2, 0, 18.2], [18.2, 50, 29.4]],
            'Fable-5.1': [[20, 75, 44.4], [0, 0, 0], [25, 0, 16.7], [0, 0, 0], [22.2, 50, 27.3], [18.2, 16.7, 17.6]],
            'Qwen3.8-Max': [[0, 75, 33.3], [20, 33.3, 25], [0, 50, 16.7], [0, 0, 0], [22.2, 0, 18.2], [18.2, 0, 11.8]],
            'Grok-4.7': [[20, 25, 22.2], [0, 33.3, 12.5], [50, 0, 33.3], [0, 0, 0], [22.2, 50, 27.3], [9.1, 0, 5.9]],
            'DeepSeek-V4-Pro': [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 16.7, 5.9]],
            'Kimi-K3': [[0, 50, 22.2], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 16.7, 5.9]]
        },
        dataCurve: { x: [0,1,2,3,4,5,6,7], train: [37.5,43.8,37.5,43.8,75,81.3,62.5,62.5], validationX: [0,3,7], validation: [50,37.5,31.25], hard: [50,50,50,50,0,0,0,0] },
        // Issue-level mask-window task: the reward curve reads healthy while
        // the logged masked-token fraction explodes. Steps 1-87 of the frozen
        // run, sampled every third step plus the 75% peak at step 74
        // (rs_filtered_fraction / task_reward from the run's main.log).
        evidence: {
            x: [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64,67,70,73,74,76,79,82,85,87],
            mask: [0.0161,0.0224,0.0298,0.0374,0.0395,0.0438,0.0474,0.0527,0.0549,0.0578,0.0627,0.0639,0.0643,0.0711,0.0902,0.1538,0.2982,0.335,0.4669,0.3949,0.1241,0.7143,0.6877,0.6904,0.6759,0.75,0.1784,0.039,0.0253,0.5986,0.6724],
            reward: [0.7363,0.7256,0.7227,0.752,0.7529,0.7812,0.7617,0.7793,0.7656,0.7891,0.7998,0.7881,0.7852,0.7764,0.7148,0.5537,0.4932,0.6328,0.6484,0.6133,0.6846,0.6318,0.6367,0.5889,0.6504,0.6143,0.6016,0.6465,0.6435,0.6846,0.7022]
        },
        hacking: [
            {
                step: 'STEP 35', title: 'GET SOURCE', subtitle: 'Obtain a reference', color: C.navy,
                thought: '“Direct downloads are failing via the proxy. I will fetch the upstream implementation and compare it with the workspace.”',
                evidence: 'fetch upstream source', shortEvidence: 'fetch source'
            },
            {
                step: 'STEP 36–46', title: 'DIAGNOSE', subtitle: 'Compare and localize', color: C.blue,
                thought: '“Every query returns the same three documents, with score 0.0. The query embeddings reaching the index are all zeros.”',
                evidence: '144 requests · identical results · zero scores', shortEvidence: 'identical results · score 0.0'
            },
            {
                step: 'STEP 77', title: 'PATCH', subtitle: 'Port the reference fix', color: C.warm,
                thought: '“Replace the broken conversion with the upstream CPU-transfer code. Applying the fix.”',
                evidence: 'copy → CPU → array · edit applied', shortEvidence: 'copy → CPU → array'
            },
            {
                step: 'STEP 78–79', title: 'VERIFY', subtitle: 'Check local agreement', color: C.red,
                thought: '“The file is now identical to upstream. Syntax OK.”',
                evidence: 'local diff · syntax check', shortEvidence: 'identical · syntax OK'
            }
        ],
        costs: {
            issue: [['Astra', 5.03, 34.3], ['Sol', 0.79, 31.4], ['Gemini', 13.01, 17.1], ['GLM', 17.77, 22.9], ['Kimi 3.1', 5.42, 17.1], ['Fable', 14.47, 17.1], ['Qwen', 0.97, 14.3], ['Grok', 2.96, 17.1], ['DeepSeek', 13.31, 0.0], ['Kimi K3', 4.34, 0.0]],
            solution: [['Astra', 47.0, 38.9], ['Sol', 6.04, 38.9], ['Gemini', 43.42, 55.6], ['GLM', 45.48, 33.3], ['Kimi 3.1', 4.82, 38.9], ['Fable', 34.69, 27.8], ['DeepSeek', 38.76, 5.6], ['Kimi K3', 13.16, 16.7]]
        },
        modelLogos: {
            Fable: 'anthropic.png', DeepSeek: 'deepseek.png', GLM: 'glm.png', Astra: 'openai.png',
            Sol: 'openai.png', Gemini: 'googlegemini.png', 'Kimi K3': 'kimi.png', 'Kimi 3.1': 'kimi.png'
        },
        times: {
            issue: [['Astra',64.13,11.68], ['Sol',17.82,13.38], ['Gemini',23.64,19.29], ['GLM',24.95,17.71], ['Kimi 3.1',27.14,20.77], ['Fable',31.19,16.65], ['Qwen',30.23,28.46], ['Grok',15.06,13.11], ['DeepSeek',37.73,33.52], ['Kimi K3',42.16,17.16]],
            solution: [['Astra',114.3,34.07], ['Sol',105.0,25.36], ['Gemini',100.83,77.3], ['GLM',60.66,52.23], ['Kimi 3.1',21.42,20.07], ['Fable',47.97,10.66], ['Qwen',43.36,37.37], ['Grok',20.71,16.65], ['DeepSeek',83.03,52.69], ['Kimi K3',48.62,27.7]]
        }
    };

    const svgEl = (tag, attrs, parent) => {
        const node = document.createElementNS(NS, tag);
        if (attrs) Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
        if (parent) parent.appendChild(node);
        return node;
    };
    const text = (parent, x, y, value, cls = '', attrs = {}) => {
        const node = svgEl('text', { x, y, class: cls, ...attrs }, parent);
        node.textContent = value;
        return node;
    };
    const group = (parent, cls = '') => svgEl('g', { class: cls }, parent);
    const pct = value => `${Number(value).toFixed(value % 1 ? 1 : 0)}%`;
    const linePath = points => points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');
    const addGrid = (root, x, y, w, h, ticks, max, formatter = v => String(v)) => {
        ticks.forEach(tick => {
            const yy = y + h - (tick / max) * h;
            svgEl('line', { x1:x, y1:yy, x2:x+w, y2:yy, class:'chart-grid' }, root);
            text(root, x - 10, yy + 4, formatter(tick), 'chart-axis chart-axis-y', {'text-anchor':'end'});
        });
        svgEl('line', { x1:x, y1:y+h, x2:x+w, y2:y+h, class:'chart-axis-line' }, root);
    };
    const make = (el, width, height, label) => {
        el.querySelectorAll('svg').forEach(node => node.remove());
        el.setAttribute('aria-label', label);
        const svg = svgEl('svg', { viewBox:`0 0 ${width} ${height}`, role:'img', 'aria-label':label }, el);
        svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        return svg;
    };
    const modelOptions = () => [['all', 'Overall'], ...data.models.map(item => [item[0], item[0]])];
    const categoryOptions = () => [['all', 'Overall'], ...data.profile.errors.map(item => [item[0], item[0]])];
    const chartSelect = (el, label, options, selected, onChange) => {
        let controls = el.querySelector('.chart-controls');
        if (!controls) {
            controls = document.createElement('div');
            controls.className = 'chart-controls';
            el.insertBefore(controls, el.querySelector('svg'));
        }
        let field = controls.querySelector('select');
        if (!field) {
            if (label) {
                const labelNode = document.createElement('span');
                labelNode.className = 'chart-control-label';
                labelNode.textContent = label;
                controls.appendChild(labelNode);
            }
            field = document.createElement('select');
            field.className = 'chart-select';
            controls.appendChild(field);
        }
        field.setAttribute('aria-label', label);
        field.innerHTML = '';
        options.forEach(([value, name]) => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = name;
            field.appendChild(option);
        });
        field.value = selected;
        field.onchange = () => onChange(field.value);
        return field;
    };
    const legend = (root, items, x, y, gap = 18) => {
        let cursor = x;
        items.forEach(item => {
            svgEl('rect', { x:cursor, y:y-10, width:12, height:12, rx:2, fill:item.color }, root);
            text(root, cursor+18, y, item.label, 'chart-legend-text', {'text-anchor':'start'});
            cursor += 18 + item.label.length * 7 + gap;
        });
    };
    const centeredLegend = (root, items, centerX, y, gap = 18) => {
        const width = items.reduce((sum, item) => sum + 18 + item.label.length * 7 + gap, -gap);
        legend(root, items, centerX - width / 2, y, gap);
    };
    const addTitle = (root, title, x, y, anchor='start') => text(root, x, y, title, 'chart-title', {'text-anchor':anchor});
    const animatePath = (path, delay=0) => {
        try { const length = path.getTotalLength(); path.style.setProperty('--length', length); } catch (_) {}
        path.style.setProperty('--delay', `${delay}ms`);
        path.classList.add('chart-line');
    };
    const animateMark = (node, delay=0, mode='pop') => {
        node.style.setProperty('--delay', `${delay}ms`);
        node.classList.add(`chart-${mode}`);
    };

    const polar = (cx, cy, radius, angle) => [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
    const pieTextColor = hex => {
        const value = hex.replace('#', '');
        const rgb = [0, 2, 4].map(offset => parseInt(value.slice(offset, offset + 2), 16) / 255);
        const linear = rgb.map(channel => channel <= .04045 ? channel / 12.92 : Math.pow((channel + .055) / 1.055, 2.4));
        const luminance = .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2];
        // Keep the same contrast decision as the publication renderer: the
        // light peach and sky-blue wedges use ink text while saturated wedges
        // retain white labels.
        return luminance > .245 ? C.ink : '#fff';
    };
    const sectorPath = (cx, cy, radius, start, end) => {
        const [sx, sy] = polar(cx, cy, radius, start);
        const [ex, ey] = polar(cx, cy, radius, end);
        const large = end - start > Math.PI ? 1 : 0;
        return `M ${cx} ${cy} L ${sx} ${sy} A ${radius} ${radius} 0 ${large} 1 ${ex} ${ey} Z`;
    };
    const pieLegend = (root, items, x, y, cols = 1, rowHeight = 14) => {
        const colWidth = cols === 1 ? 150 : 135;
        items.forEach((item, i) => {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const xx = x + col * colWidth;
            const yy = y + row * rowHeight;
            svgEl('rect', { x: xx, y: yy - 8, width: 8, height: 8, rx: 1, fill: item.color }, root);
            text(root, xx + 12, yy, `${item.label} (${item.value})`, 'chart-pie-label', { 'text-anchor': 'start' });
        });
    };
    const drawPie = (root, panel, index) => {
        const total = panel.items.reduce((sum, item) => sum + item[1], 0);
        const cx = panel.cx, cy = panel.cy, radius = panel.radius;
        addTitle(root, panel.title, cx, 25, 'middle');
        let angle = -Math.PI / 2;
        const legendItems = [];
        panel.items.forEach((item, i) => {
            const value = item[1];
            const end = angle + (value / total) * Math.PI * 2;
            const isSelected = panel.selected && panel.selected !== 'all';
            const active = !isSelected || panel.selected === item[0];
            const wedge = svgEl('path', { d: sectorPath(cx, cy, radius, angle, end), fill: panel.colors[i % panel.colors.length], stroke: C.white, 'stroke-width': 2, class: 'chart-wedge' }, root);
            if (isSelected && !active) wedge.classList.add('chart-wedge-muted');
            wedge.style.setProperty('--delay', `${index * 160 + i * 55}ms`);
            const share = value / total;
            if (share >= .09) {
                const mid = (angle + end) / 2;
                const [tx, ty] = polar(cx, cy, radius * .58, mid);
                const valueLabel = text(root, tx, ty, `${(share * 100).toFixed(1)}%`, 'chart-pie-value');
                valueLabel.style.fill = pieTextColor(panel.colors[i % panel.colors.length]);
            }
            legendItems.push({ label: item[0], value, color: panel.colors[i % panel.colors.length] });
            angle = end;
        });
        pieLegend(root, legendItems, panel.legendX, panel.legendY, panel.legendCols, panel.legendRow);
    };

    function barTip(el, name, issue, solution, ev) {
        let tip = el.querySelector('.chart-bar-tip');
        if (!tip) {
            tip = document.createElement('div');
            tip.className = 'chart-bar-tip';
            tip.setAttribute('role', 'tooltip');
            el.appendChild(tip);
        }
        tip.innerHTML = '';
        const title = document.createElement('strong');
        title.textContent = name;
        const l1 = document.createElement('span');
        l1.textContent = `Issue ${issue}%`;
        const l2 = document.createElement('span');
        l2.textContent = `Solution ${solution}%`;
        tip.append(title, l1, l2);
        tip.setAttribute('aria-hidden', 'false');
        tip.classList.add('is-open');
        const box = el.getBoundingClientRect();
        const tx = ev.clientX - box.left, ty = ev.clientY - box.top;
        tip.style.left = Math.min(Math.max(tx + 14, 6), box.width - tip.offsetWidth - 8) + 'px';
        tip.style.top = Math.max(ty - tip.offsetHeight - 12, 4) + 'px';
    }

    function barTipHide(el) {
        const tip = el.querySelector('.chart-bar-tip');
        if (tip) { tip.classList.remove('is-open'); tip.setAttribute('aria-hidden', 'true'); }
    }

    function combinedProfile(el, selectedCategory = 'all') {
        const svg = make(el, 1000, 790, 'Benchmark profile and leaderboard: affiliations, categories, difficulty, and model pass rates');
        chartSelect(el, '', categoryOptions(), selectedCategory, value => combinedProfile(el, value));
        const affiliationColors = ['#104680', '#6D011F', '#2B72AD', '#A91C2D', '#569BC7', '#CE5148', '#8DBFDB', '#E78B71', '#C4DEEB', '#F7C0A6', '#EBF0F1', '#F9E4D8'];
        const categoryColors = ['#104680', '#F6B293', '#317CB7', '#DC6D57', '#6DADD1', '#6D011F'];
        const difficultyColors = ['#6D011F', '#B72230', '#F6B293', '#317CB7'];
        drawPie(svg, { title: '(a) Affiliations', cx: 166, cy: 142, radius: 96, items: data.profile.affiliations, colors: affiliationColors, legendX: 22, legendY: 276, legendCols: 2, legendRow: 14 }, 0);
        drawPie(svg, { title: '(b) Categories', cx: 500, cy: 142, radius: 96, items: data.profile.errors, colors: categoryColors, selected: selectedCategory, legendX: 408, legendY: 276, legendCols: 2, legendRow: 14 }, 1);
        drawPie(svg, { title: '(c) Difficulty', cx: 834, cy: 142, radius: 96, items: data.profile.difficulty, colors: difficultyColors, legendX: 766, legendY: 276, legendCols: 1, legendRow: 16 }, 2);

        const selectedCategoryData = data.profile.errors.find(item => item[0] === selectedCategory);
        const categoryIndex = selectedCategoryData ? data.categories.findIndex(item => item[0] === selectedCategory) : -1;
        const leaderboard = categoryIndex >= 0
            ? data.models.map(item => [item[0], data.categoryByModel[item[0]][categoryIndex][2]])
                .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
            : data.models;
        addTitle(svg, selectedCategoryData ? `(d) Leaderboard · ${selectedCategory}` : '(d) Leaderboard · Overall', 500, 365, 'middle');
        // Keep a gap between the (d) title and the bar-top value labels:
        // in a category view every bar can reach 100%, and the labels sit
        // just above the plot's top edge.
        const x = 70, y = 420, w = 875, h = 260, max = selectedCategoryData ? 100 : 45;
        addGrid(svg, x, y, w, h, selectedCategoryData ? [0, 25, 50, 75, 100] : [0, 10, 20, 30, 40], max, value => `${value}%`);
        const step = w / leaderboard.length;
        const barWidth = step * .62;
        leaderboard.forEach((item, i) => {
            const bx = x + i * step + (step - barWidth) / 2;
            const bh = item[1] / max * h;
            const by = y + h - bh;
            const modelIndex = data.models.findIndex(model => model[0] === item[0]);
            const bar = svgEl('rect', { x: bx, y: by, width: barWidth, height: bh, rx: 2, fill: data.leaderboardColors[i], class: 'chart-bar' }, svg);
            animateMark(bar, 560 + i * 85, 'rise');
            // Hovering a bar splits its overall score into the two levels:
            // issue + solution in the overall view, or that category's
            // diagnosis + repair rates in a category view.
            const levels = selectedCategoryData
                ? data.categoryByModel[item[0]][categoryIndex]
                : data.modelLevels[item[0]];
            const showTip = ev => barTip(el, item[0], levels[0], levels[1], ev);
            bar.addEventListener('mouseenter', showTip);
            bar.addEventListener('mousemove', showTip);
            bar.addEventListener('mouseleave', () => barTipHide(el));
            text(svg, bx + barWidth / 2, by - 8, pct(item[1]), 'chart-value', {'text-anchor': 'middle'});
            // Model names are wider than one bar step, so the labels rotate.
            // Each label is centred on its own bar; the anchor sits far
            // enough below the axis that the raised end of the longest name
            // still stays clear of the plot area.
            const anchorX = bx + barWidth / 2, anchorY = y + h + 30;
            const label = svgEl('text', { x: anchorX, y: anchorY, class: 'chart-x-label', 'text-anchor': 'middle', transform: `rotate(-35 ${anchorX} ${anchorY})` }, svg);
            const modelLine = document.createElementNS(NS, 'tspan');
            modelLine.setAttribute('x', anchorX); modelLine.textContent = item[0]; label.appendChild(modelLine);
            const harnessLine = document.createElementNS(NS, 'tspan');
            harnessLine.setAttribute('x', anchorX); harnessLine.setAttribute('dy', '13'); harnessLine.textContent = data.modelHarness[modelIndex]; label.appendChild(harnessLine);
        });
    }

    function outcomeBars(el) {
        const svg = make(el, 1000, 500, 'Diagnosis outcomes and selected failure patterns');
        addTitle(svg,'Overall failure modes of diagnosis tasks',500,23,'middle');
        const total=data.outcomes.reduce((sum,item)=>sum+item[1],0), x=30,y=34,w=940,h=34; let cursor=x; const stack=group(svg,'chart-stack-group');
        data.outcomes.forEach((item,i)=>{ const bw=w*item[1]/total; svgEl('rect',{x:cursor,y,width:bw,height:h,fill:item[2],class:'chart-stack'},stack); text(stack,cursor+bw/2,y+h/2,pct(item[1]/total*100),'chart-stack-label',{'text-anchor':'middle','dominant-baseline':'middle'}); cursor+=bw; }); animateMark(stack,0,'rise');
        addTitle(svg,'Examples of failure modes by task',500,120,'middle');
        const left=190,right=950,row=29, top=141,max=10;
        data.outcomeTasks.forEach((item,i)=>{
            const yy=top+i*row;
            text(svg,left-10,yy+14,item[0],'chart-label',{'text-anchor':'end'});
            let offset=left;
            item.slice(1).forEach((value,j)=>{
                const segmentWidth=(right-left)*value/max;
                if (!segmentWidth) return;
                const bar=svgEl('rect',{x:offset,y:yy,width:segmentWidth,height:16,fill:data.outcomes[j][2],class:'chart-hbar'},svg);
                animateMark(bar,240+i*42+j*35,'rise');
                offset += segmentWidth;
            });
        });
        // The summary legend is shared by both views and lives below the task bars.
        centeredLegend(svg,data.outcomes.map(d=>({label:d[0],color:d[2]})),500,477,28);
    }

    function curves(el) {
        const svg = make(el, 1000, 620, 'Passed and failed repair trajectories from similar starting points');
        addTitle(svg, 'RL Training Curves for Four Repair Tasks', 500, 22, 'middle');
        const yAxisLabel = text(svg, 14, 310, 'validation score', 'chart-axis', {'text-anchor':'middle'});
        yAxisLabel.setAttribute('transform', 'rotate(-90 14 310)');
        const panels=data.curves, pw=455, ph=225, lefts=[28,515], tops=[52,326];
        panels.forEach((panel,pi)=>{
            const ox=lefts[pi%2], oy=tops[Math.floor(pi/2)], x=ox+44,y=oy+24,w=pw-66,h=ph-58;
            const panelTitle=addTitle(svg,`(${String.fromCharCode(97+pi)}) ${panel.title}`,ox+pw/2,oy+4,'middle');
            panelTitle.classList.add('chart-panel-title');
            addGrid(svg,x,y,w,h,[0,panel.max/2,panel.max],panel.max,v=>String(Math.round(v)));
            const points=(values)=>values.map((v,i)=>[x+(panel.x[i]-panel.x[0])/(panel.x[panel.x.length-1]-panel.x[0])*w,y+h-v/panel.max*h]);
            const baseY=y+h-panel.gate/panel.max*h; svgEl('line',{x1:x,y1:baseY,x2:x+w,y2:baseY,class:'chart-gate'},svg);
            [panel.a,panel.b].forEach((values,series)=>{
                const pts=points(values);
                pts.forEach((p,i)=>{ if(i===pts.length-1)return; const path=svgEl('path',{d:linePath([p,pts[i+1]]),stroke:panel.colors[series],class:'chart-segment',fill:'none'} ,svg); animatePath(path,series*80+i*45); });
                pts.forEach((p,i)=>{const dot=svgEl('circle',{cx:p[0],cy:p[1],r:4,fill:panel.colors[series],class:'chart-dot'},svg); animateMark(dot,series*80+i*45+240,'pop');});
            });
            text(svg,x+w/2,oy+ph-2,'RL Training Step','chart-axis',{'text-anchor':'middle'});
            centeredLegend(svg,[{label:panel.aLabel,color:panel.colors[0]},{label:panel.bLabel,color:panel.colors[1]}],x+w/2,oy+ph+22);
        });
    }

    function quadrants(el) {
        const svg=make(el,800,410,'Diagnosis and repair outcomes across shared tasks');
        const x=170,y=35,size=300,half=size/2; const cells=[
            [0,0,'Diagnosis fails','Repair fails',65.6,C.navy], [1,0,'Diagnosis fails','Repair passes',21.1,C.paleBlue],
            [0,1,'Diagnosis passes','Repair fails',4.4,'#e9f0f3'], [1,1,'Diagnosis passes','Repair passes',8.9,'#c8dce8']
        ];
        cells.forEach((cell,i)=>{
            const [cx,cy,_,__,v,color]=cell;
            const rect=svgEl('rect',{x:x+cx*half,y:y+cy*half,width:half,height:half,fill:color,class:'chart-cell'},svg);
            animateMark(rect,i*130,'pop');
            const centerX=x+cx*half+half/2;
            const centerY=y+cy*half+half/2;
            text(svg,centerX,centerY,pct(v),'chart-cell-value');
        });
        svgEl('rect',{x,y,width:size,height:size,class:'chart-frame'},svg);
        text(svg,x+half/2,y+size+34,'Repair fails','chart-quadrant-axis',{'text-anchor':'middle'});
        text(svg,x+half+half/2,y+size+34,'Repair passes','chart-quadrant-axis',{'text-anchor':'middle'});
        text(svg,x-18,y+half/2,'Diagnosis fails','chart-quadrant-axis',{'text-anchor':'end','dominant-baseline':'middle'});
        text(svg,x-18,y+half+half/2,'Diagnosis passes','chart-quadrant-axis',{'text-anchor':'end','dominant-baseline':'middle'});

        const sideX=535;
        text(svg,sideX,151,'360 trajectories','chart-quadrant-side-value');
        text(svg,sideX,211,'24','chart-quadrant-side-value');
        text(svg,sideX,231,'diagnosis passes','chart-quadrant-side-label');
        text(svg,sideX,271,'54','chart-quadrant-side-value');
        text(svg,sideX,291,'repair passes','chart-quadrant-side-label');
    }

    function repairErrors(el) {
        const svg = make(el, 900, 380, 'Repair error analysis across failed repair trajectories');
        addTitle(svg, 'Error Analysis of Repair Trajectories', 450, 28, 'middle');
        const x = 235, y = 61, w = 575, row = 56, max = 50;
        const failed = data.repairErrors.reduce((sum, item) => sum + item[1], 0);
        const plotHeight = row * data.repairErrors.length - 12;
        [0, 10, 20, 30, 40, 50].forEach(tick => {
            const xx = x + w * tick / max;
            svgEl('line', { x1: xx, y1: y, x2: xx, y2: y + plotHeight, class: 'chart-grid' }, svg);
            text(svg, xx, y + plotHeight + 21, `${tick}%`, 'chart-axis', { 'text-anchor': 'middle' });
        });
        svgEl('line', { x1: x, y1: y + plotHeight, x2: x + w, y2: y + plotHeight, class: 'chart-axis-line' }, svg);
        data.repairErrors.forEach((item, i) => {
            const [label, count, color] = item;
            const yy = y + i * row + 7;
            const value = count / failed * 100;
            text(svg, x - 14, yy + 18, label, 'chart-label', { 'text-anchor': 'end' });
            const bar = svgEl('rect', { x, y: yy, width: w * value / max, height: 28, rx: 3, fill: color, class: 'chart-hbar' }, svg);
            // Keep the horizontal scale animation so each row grows from the
            // baseline as it enters the viewport.
            animateMark(bar, i * 110, 'hbar');
            text(svg, x + w * value / max + 12, yy + 19, `${value.toFixed(1)}%`, 'chart-value');
        });
    }

    function categoryBars(el, selected = 'all') {
        const svg=make(el,900,430,'Diagnosis and repair pass rates across six failure categories');
        chartSelect(el, '', modelOptions(), selected, value => categoryBars(el, value));
        const categories = selected === 'all'
            ? data.categories
            : data.categories.map((item, i) => [item[0], ...data.categoryByModel[selected][i]]);
        addTitle(svg, selected === 'all' ? 'Diagnosis and repair pass rates · Overall' : `Diagnosis and repair pass rates · ${selected}`, 56, 25);
        const x=56,y=48,w=810,h=260,max=100; addGrid(svg,x,y,w,h,[0,25,50,75,100],max,v=>`${v}%`); const step=w/categories.length,groupWidth=step*.72,bw=groupWidth*.34,gap=groupWidth*.12;
        categories.forEach((item,i)=>{
            const cx=x+i*step+step/2;
            [item[1],item[2]].forEach((v,j)=>{
                const bh=v/max*h,bx=cx-groupWidth/2+j*(bw+gap),by=y+h-bh;
                const bar=svgEl('rect',{x:bx,y:by,width:bw,height:bh,rx:2,fill:j?C.red:C.navy,class:'chart-bar'},svg);
                animateMark(bar,i*100+j*70,'rise');
                if (v > 0) text(svg,bx+bw/2,by-8,pct(v),'chart-value',{'text-anchor':'middle'});
            });
            text(svg,cx,y+h+32,item[0],'chart-x-label',{'text-anchor':'middle'});
        });
        legend(svg,[{label:'Diagnosis',color:C.navy},{label:'Repair',color:C.red}],690,25);
    }

    function dataTrajectory(el) {
        const svg=make(el,900,430,'Data failure trajectory: reward rises while validation and hard-sample coverage fall');
        addTitle(svg,'An Example of a Data Failure Task',450,22,'middle');
        const x=65,y=38,w=790,h=260; addGrid(svg,x,y,w,h,[0,25,50,75,100],100,v=>(v/100).toFixed(2));
        const point=(step,value)=>[x+step/7*w,y+h-value/100*h];
        const allPoints=data.dataCurve.x.map((step,i)=>point(step,data.dataCurve.train[i]));
        const validationPoints=data.dataCurve.validationX.map((step,i)=>point(step,data.dataCurve.validation[i]));
        const hardPoints=data.dataCurve.x.map((step,i)=>point(step,data.dataCurve.hard[i]));
        const removalX=x+3.5/7*w;
        svgEl('rect',{x:removalX,y,width:x+w-removalX,height:h,class:'chart-region'},svg);
        const addSeries=(points,color,shape,dash,delay)=>{
            for(let i=0;i<points.length-1;i++){
                const attrs={d:linePath([points[i],points[i+1]]),stroke:color,class:'chart-segment',fill:'none','stroke-width':2.8};
                if(dash) attrs['stroke-dasharray']=dash;
                const path=svgEl('path',attrs,svg); animatePath(path,delay+i*70);
            }
            points.forEach((p,i)=>{
                let mark;
                if(shape==='circle') mark=svgEl('circle',{cx:p[0],cy:p[1],r:5.5,fill:color,class:'chart-dot'},svg);
                if(shape==='square') mark=svgEl('rect',{x:p[0]-5.5,y:p[1]-5.5,width:11,height:11,fill:color,class:'chart-point-square'},svg);
                if(shape==='triangle') mark=svgEl('polygon',{points:`${p[0]},${p[1]-6.5} ${p[0]-6.5},${p[1]+5.5} ${p[0]+6.5},${p[1]+5.5}`,fill:color,class:'chart-point-triangle'},svg);
                animateMark(mark,delay+240+i*55,'pop');
            });
        };
        addSeries(allPoints,'#104680','circle','',0);
        addSeries(validationPoints,'#6D011F','square','8 6',110);
        addSeries(hardPoints,'#F6B293','triangle','2 7',220);
        svgEl('line',{x1:removalX,y1:y,x2:removalX,y2:y+h,class:'chart-data-separator'},svg);
        const eventLabel=svgEl('text',{x:removalX+18,y:y+25,class:'chart-data-label'},svg);
        const firstLine=document.createElementNS(NS,'tspan'); firstLine.setAttribute('x',removalX+18); firstLine.textContent='hard samples removed;'; eventLabel.appendChild(firstLine);
        const secondLine=document.createElementNS(NS,'tspan'); secondLine.setAttribute('x',removalX+18); secondLine.setAttribute('dy','17'); secondLine.textContent='easy samples refill the batch'; eventLabel.appendChild(secondLine);
        data.dataCurve.x.forEach(step=>{const xx=x+step/7*w; svgEl('line',{x1:xx,y1:y+h,x2:xx,y2:y+h+5,class:'chart-axis-line'},svg); text(svg,xx,y+h+22,String(step),'chart-axis',{'text-anchor':'middle'});});
        text(svg,x+w/2,y+h+56,'RL Training Step','chart-axis',{'text-anchor':'middle'});
        const axisLabel=text(svg,x-45,y+h/2,'Accuracy','chart-axis',{'text-anchor':'middle'}); axisLabel.setAttribute('transform',`rotate(-90 ${x-45} ${y+h/2})`);
        centeredLegend(svg,[{label:'training reward',color:'#104680'},{label:'Validation Score',color:'#6D011F'},{label:'hard-sample coverage',color:'#F6B293'}],x+w/2,405);
    }

    function causeFan(el) {
        const svg=make(el,900,410,'A near-zero oscillating reward curve with four possible root causes');
        addTitle(svg,'One Symptom, Four Causes',450,24,'middle');
        const x=120,y=60,w=660,h=82;
        const domainTop=0.006, domainBottom=-0.01;
        const valueY=value=>y+(domainTop-value)/(domainTop-domainBottom)*h;
        [-0.01,0].forEach(value=>{
            const yy=valueY(value);
            svgEl('line',{x1:x,y1:yy,x2:x+w,y2:yy,class:'chart-grid'},svg);
            text(svg,x-10,yy+4,value.toFixed(2),'chart-axis chart-axis-y',{'text-anchor':'end'});
        });
        svgEl('line',{x1:x,y1:y,x2:x,y2:y+h,class:'chart-axis-line'},svg);
        svgEl('line',{x1:x,y1:y+h,x2:x+w,y2:y+h,class:'chart-axis-line'},svg);
        // Use one continuous jagged path. Turning points stay on a fixed grid so
        // the signal feels like a coherent training trace rather than random
        // spikes with uneven spacing.
        const steps=Array.from({length:241},(_,i)=>i);
        let seed=9173;
        const random=()=>{
            seed=(seed*1664525+1013904223)>>>0;
            return seed/4294967296;
        };
        const turnSpacing=8;
        const turningValues=steps.filter(i=>i%turnSpacing===0 || i===steps.length-1).map((step,index)=>{
            const magnitude=0.0007+random()*0.0034;
            const sign=index%2===0?1:-1;
            return {step,value:sign*magnitude};
        });
        const values=steps.map(i=>{
            const leftIndex=Math.min(Math.floor(i/turnSpacing),turningValues.length-2);
            const left=turningValues[leftIndex];
            const right=turningValues[leftIndex+1];
            const span=right.step-left.step;
            const t=span?((i-left.step)/span):0;
            return left.value+(right.value-left.value)*t;
        });
        const point=(step,value)=>[x+step/(steps.length-1)*w,valueY(value)];
        const rewardPoints=steps.map((step,i)=>point(step,values[i]));
        const curve=svgEl('path',{d:linePath(rewardPoints),stroke:C.navy,'stroke-width':2.4,class:'chart-segment',fill:'none'},svg);
        animatePath(curve,120);
        [0,20,40,60,80,100].forEach(step=>{
            const xx=x+step/100*w;
            svgEl('line',{x1:xx,y1:y+h,x2:xx,y2:y+h+5,class:'chart-axis-line'},svg);
            text(svg,xx,y+h+21,String(step),'chart-axis',{'text-anchor':'middle'});
        });
        text(svg,x+w/2,y+h+43,'training step','chart-axis',{'text-anchor':'middle'});
        const axisLabel=text(svg,x-50,y+h/2,'reward','chart-axis',{'text-anchor':'middle'});
        axisLabel.setAttribute('transform',`rotate(-90 ${x-50} ${y+h/2})`);
        const causes=[
            {tag:'DATA',color:'#6DADD1',lines:['the model cannot solve the','training problems, so the','all-tests-pass reward never fires'],task:'TACO difficulty mismatch'},
            {tag:'DATA',color:'#6DADD1',lines:['the training labels are wrong,','so correctly solved rollouts','still score zero'],task:'TACO wrong expected outputs'},
            {tag:'REWARD',color:'#DC6D57',lines:['the parser reads a key the','template never emits, so the','rewarded behavior never runs'],task:'ReTool parser mismatch'},
            {tag:'REWARD',color:'#DC6D57',lines:['a reward-worker thread fails','silently, and every answer is','scored 0.0 by default'],task:'math-verify zero reward'}
        ];
        causes.forEach((cause,i)=>{
            const cardX=32+i*215, cardY=220, cardW=191, cardH=138;
            const card=svgEl('rect',{x:cardX,y:cardY,width:cardW,height:cardH,rx:12,fill:'rgba(255,255,255,.55)',stroke:cause.color,'stroke-width':1.4,class:'chart-cell'},svg);
            animateMark(card,160+i*160,'pop');
            const chipWidth=Math.min(cause.tag.length*8+20,cardW-28);
            const chip=svgEl('rect',{x:cardX+14,y:cardY+16,width:chipWidth,height:17,rx:4,fill:cause.color,class:'chart-pop'},svg);
            animateMark(chip,200+i*160,'pop');
            const chipText=text(svg,cardX+22,cardY+28,cause.tag,'chart-cause-tag',{'text-anchor':'start'});
            animateMark(chipText,220+i*160,'pop');
            cause.lines.forEach((line,j)=>text(svg,cardX+14,cardY+62+j*15,line,'chart-hacking-evidence'));
            text(svg,cardX+14,cardY+122,`· ${cause.task}`,'chart-hacking-evidence');
        });
    }

    function evidenceCurve(el) {
        const svg=make(el,900,350,'Two continuous curves: training reward and masked-token fraction');
        addTitle(svg,'Healthy Curve, Hidden Failure',450,24,'middle');
        // The reward stays in a plausible, high band throughout this example;
        // the hidden failure is exposed by the separate masked-token trace.
        const healthyReward=data.evidence.x.map((_,index)=>{
            const progress=index/(data.evidence.x.length-1);
            return 0.765+0.035*progress+0.009*Math.sin(index*0.42)+0.004*Math.sin(index*1.15+0.4);
        });
        const panels=[
            {x:68,y:62,w:350,h:190,title:'Training reward',axis:'reward',color:'#104680',min:.68,max:.86,ticks:[.7,.78,.86],format:value=>value.toFixed(2)},
            {x:482,y:62,w:350,h:190,title:'Masked-token fraction',axis:'masked-token fraction',color:'#6D011F',min:0,max:.8,ticks:[0,.4,.8],format:value=>`${Math.round(value*100)}%`}
        ];
        const xTicks=[1,29,58,87];
        panels.forEach((panel,panelIndex)=>{
            text(svg,panel.x+panel.w/2,panel.y-22,panel.title,'chart-label',{'text-anchor':'middle'});
            panel.ticks.forEach(tick=>{
                const yy=panel.y+panel.h-(tick-panel.min)/(panel.max-panel.min)*panel.h;
                svgEl('line',{x1:panel.x,y1:yy,x2:panel.x+panel.w,y2:yy,class:'chart-grid'},svg);
                text(svg,panel.x-10,yy+4,panel.format(tick),'chart-axis chart-axis-y',{'text-anchor':'end'});
            });
            svgEl('line',{x1:panel.x,y1:panel.y,x2:panel.x,y2:panel.y+panel.h,class:'chart-axis-line'},svg);
            svgEl('line',{x1:panel.x,y1:panel.y+panel.h,x2:panel.x+panel.w,y2:panel.y+panel.h,class:'chart-axis-line'},svg);
            const raw=panelIndex===0?healthyReward:data.evidence.mask;
            const points=data.evidence.x.map((step,index)=>[
                panel.x+(step-1)/86*panel.w,
                panel.y+panel.h-(raw[index]-panel.min)/(panel.max-panel.min)*panel.h
            ]);
            const curve=svgEl('path',{d:linePath(points),stroke:panel.color,class:'chart-line',fill:'none','stroke-width':2.8},svg);
            animatePath(curve,panelIndex*180);
            xTicks.forEach(step=>{
                const xx=panel.x+(step-1)/86*panel.w;
                svgEl('line',{x1:xx,y1:panel.y+panel.h,x2:xx,y2:panel.y+panel.h+5,class:'chart-axis-line'},svg);
                text(svg,xx,panel.y+panel.h+22,String(step),'chart-axis',{'text-anchor':'middle'});
            });
            text(svg,panel.x+panel.w/2,panel.y+panel.h+49,'RL training step','chart-axis',{'text-anchor':'middle'});
            const axisLabel=text(svg,panel.x-43,panel.y+panel.h/2,panel.axis,'chart-axis',{'text-anchor':'middle'});
            axisLabel.setAttribute('transform',`rotate(-90 ${panel.x-43} ${panel.y+panel.h/2})`);
        });
    }

    function hacking(el) {
        const svg=make(el,1000,200,'Observable reasoning trace from a Search-R1 repair trajectory');
        const defs=svgEl('defs',{},svg);
        const marker=svgEl('marker',{id:'hacking-arrowhead',viewBox:'0 0 10 10',refX:'9',refY:'5',markerWidth:'6',markerHeight:'6',orient:'auto-start-reverse'},defs);
        svgEl('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:C.red},marker);

        const stages=data.hacking;
        const points=[150,383,616,849];
        const timelineY=92;
        svgEl('line',{x1:100,y1:timelineY,x2:900,y2:timelineY,class:'chart-timeline'},svg);
        stages.forEach((stage,i)=>{
            const x=points[i];
            if(i<points.length-1){
                const link=svgEl('line',{x1:x+18,y1:timelineY,x2:points[i+1]-18,y2:timelineY,stroke:C.red,'stroke-width':2,'marker-end':'url(#hacking-arrowhead)',class:'chart-hacking-link'},svg);
                animatePath(link, i*280+180);
            }

            const node=svgEl('g',{class:'chart-hacking-node',tabindex:'0',role:'button','aria-label':`${stage.step}: ${stage.title}`},svg);
            const halo=svgEl('circle',{cx:x,cy:timelineY,r:22,fill:stage.color,opacity:.12,class:'chart-hacking-halo'},node);
            animateMark(halo,i*280,'pop');
            const point=svgEl('circle',{cx:x,cy:timelineY,r:13,fill:stage.color,stroke:C.white,'stroke-width':3,class:'chart-hacking-point'},node);
            animateMark(point,i*280,'pop');
            const number=text(node,x,timelineY+5,String(i+1).padStart(2,'0'),'chart-node-number chart-pop',{'text-anchor':'middle','font-size':'12px'});
            animateMark(number,i*280+35,'pop');

            const step=text(node,x,timelineY-48,stage.step,'chart-hacking-step',{'text-anchor':'middle'});
            const title=text(node,x,timelineY-25,stage.title,'chart-hacking-title',{'text-anchor':'middle'});
            const subtitle=text(node,x,timelineY+48,stage.subtitle,'chart-hacking-subtitle',{'text-anchor':'middle'});
            const evidence=text(node,x,timelineY+70,stage.shortEvidence,'chart-hacking-evidence',{'text-anchor':'middle'});
            [step,title,subtitle,evidence].forEach((label,j)=>animateMark(label,i*280+90+j*20,'pop'));

            const show=()=>showHackingTooltip(el,stage,x);
            const hide=()=>hideHackingTooltip(el);
            node.addEventListener('mouseenter',show);
            node.addEventListener('mouseleave',hide);
            node.addEventListener('focus',show);
            node.addEventListener('blur',hide);
        });

        let tooltip=el.querySelector('.chart-cot-tooltip');
        if(!tooltip){
            tooltip=document.createElement('div');
            tooltip.className='chart-cot-tooltip';
            tooltip.setAttribute('role','tooltip');
            tooltip.setAttribute('aria-hidden','true');
            el.appendChild(tooltip);
        }
    }

    function showHackingTooltip(el,stage,x){
        const tooltip=el.querySelector('.chart-cot-tooltip');
        if(!tooltip)return;
        tooltip.innerHTML='';
        const heading=document.createElement('strong');
        heading.textContent=`${stage.step} · ${stage.title}`;
        const cotLabel=document.createElement('span');
        cotLabel.className='chart-cot-label';
        cotLabel.textContent='OBSERVABLE CoT';
        const quote=document.createElement('span');
        quote.className='chart-cot-quote';
        quote.textContent=stage.thought;
        const output=document.createElement('span');
        output.className='chart-cot-output';
        output.textContent=`Output: ${stage.evidence}`;
        tooltip.append(heading,cotLabel,quote,output);
        tooltip.classList.remove('edge-start','edge-end');
        if(x <= 200){
            tooltip.style.left='';
            tooltip.style.right='';
            tooltip.classList.add('edge-start');
        } else if(x >= 800){
            tooltip.style.left='';
            tooltip.style.right='';
            tooltip.classList.add('edge-end');
        } else {
            tooltip.style.right='';
            tooltip.style.left=`${x/10}%`;
        }
        tooltip.setAttribute('aria-hidden','false');
        tooltip.classList.add('is-open');
    }

    function hideHackingTooltip(el){
        const tooltip=el.querySelector('.chart-cot-tooltip');
        if(!tooltip)return;
        tooltip.classList.remove('is-open');
        tooltip.setAttribute('aria-hidden','true');
    }

    function scatter(el) {
        const svg=make(el,1000,410,'Cost and pass rate split by diagnosis and repair tasks');
        const panelW=390,panelH=300,panelY=48;
        [['Diagnosis',data.costs.issue,55,20,40,[0,10,20,30,40],[0,5,10,15,20]],['Repair',data.costs.solution,555,50,70,[0,20,40,60],[0,10,20,30,40,50]]].forEach((panel,pi)=>{
            const [label, rows, x, maxX, maxY, yTicks, xTicks] = panel;
            addTitle(svg,label,x,panelY-25);
            addGrid(svg,x,panelY,panelW,panelH,yTicks,maxY,v=>`${v}%`);
            xTicks.forEach(t=>{const xx=x+t/maxX*panelW; svgEl('line',{x1:xx,y1:panelY+panelH,x2:xx,y2:panelY+panelH+5,class:'chart-axis-line'},svg); text(svg,xx,panelY+panelH+22,`$${t}`,'chart-axis',{'text-anchor':'middle'});});
            // Each logo sits at its (cost, pass rate) point; the few logos
            // that share a spot are nudged apart by the minimum displacement,
            // and every name is hugged to its own logo -- right, left, below
            // or above, whichever is free first. No leader lines.
            const points=rows.map((item,i)=>({
                cx:x+item[1]/maxX*panelW,
                cy:panelY+panelH-item[2]/maxY*panelH,
                short:item[0],
                fullName:data.modelDisplayNames[item[0]] || item[0],
                delay:pi*180+i*70
            }));
            const size=28, margin=15;
            for (let round=0; round<24; round++) {
                let moved=false;
                for (let a=0;a<points.length;a++) for (let b=a+1;b<points.length;b++) {
                    const p=points[a], q=points[b];
                    const dx=q.cx-p.cx, dy=q.cy-p.cy;
                    if (Math.abs(dx)>=size || Math.abs(dy)>=size) continue;
                    const gapX=(size-Math.abs(dx))/2, gapY=(size-Math.abs(dy))/2;
                    if (gapX<=gapY) { const s=dx>=0?1:-1; p.cx-=s*gapX; q.cx+=s*gapX; }
                    else { const s=dy>=0?1:-1; p.cy-=s*gapY; q.cy+=s*gapY; }
                    moved=true;
                }
                points.forEach(p=>{
                    p.cx=Math.max(x+margin,Math.min(x+panelW-margin,p.cx));
                    p.cy=Math.max(panelY+margin,Math.min(panelY+panelH-margin,p.cy));
                });
                if (!moved) break;
            }
            const boxes=points.map(p=>({x:p.cx-13,y:p.cy-13,w:26,h:26}));
            const intersects=(a,b)=>a.x<b.x+b.w&&a.x+a.w>b.x&&a.y<b.y+b.h&&a.y+a.h>b.y;
            const placed=[];
            // Labels may reach a little into the gutter between the two
            // panels; they must stay inside the plot vertically.
            const outOfPanel=box=>box.x<x-30||box.x+box.w>x+panelW+30||box.y<panelY-10||box.y+box.h>panelY+panelH+16;
            const hits=box=>boxes.filter(o=>intersects(box,o)).length+placed.filter(o=>intersects(box,o)).length;
            const blocked=box=>outOfPanel(box)||hits(box)>0;
            [...points].sort((a,b)=>b.cx-a.cx).forEach(p=>{
                const w=p.fullName.length*7.8+4;
                const cands=[
                    {lx:p.cx+17,ly:p.cy,anchor:'start',box:{x:p.cx+14,y:p.cy-8,w,h:16}},
                    {lx:p.cx-17,ly:p.cy,anchor:'end',box:{x:p.cx-14-w,y:p.cy-8,w,h:16}},
                    {lx:p.cx,ly:p.cy+27,anchor:'middle',box:{x:p.cx-w/2,y:p.cy+19,w,h:15}},
                    {lx:p.cx,ly:p.cy-27,anchor:'middle',box:{x:p.cx-w/2,y:p.cy-34,w,h:15}}
                ];
                // Prefer the first collision-free side; when every side is
                // taken, fall back to the least-colliding one.
                const spot=cands.find(c=>!blocked(c.box))
                    ||cands.slice().sort((a,b)=>hits(a.box)-hits(b.box))[0];
                placed.push(spot.box);
                p.labelX=spot.lx; p.labelY=spot.ly; p.anchor=spot.anchor;
            });
            points.forEach(p=>{
                const logo=svgEl('image',{x:p.cx-13,y:p.cy-13,width:26,height:26,href:`images/logos/${data.modelLogos[p.short]}`,'preserveAspectRatio':'xMidYMid meet',class:'chart-scatter-logo'},svg);
                logo.setAttribute('aria-label', `${p.fullName} · ${label}`);
                animateMark(logo,p.delay,'pop');
                text(svg,p.labelX,p.labelY,p.fullName,'chart-point-label',{'text-anchor':p.anchor,'dominant-baseline':'middle'});
            });
            text(svg,x+panelW/2,panelY+panelH+52,'cost (USD)','chart-axis',{'text-anchor':'middle'});
        });
    }

    function timeBars(el) {
        const svg=make(el,1000,512,'Mean and median execution time for diagnosis and repair tasks');
        const panels=[['Diagnosis',data.times.issue,22,18],['Repair',data.times.solution,515,18]];
        panels.forEach((panel,pi)=>{
            const title=panel[0],rows=panel[1],ox=panel[2],oy=panel[3],max=130,w=420;
            addTitle(svg,title,ox,oy);
            rows.forEach((item,i)=>{
                const yy=oy+22+i*46;
                const fullName=data.modelDisplayNames[item[0]] || item[0];
                text(svg,ox+94,yy+13,fullName,'chart-label',{'text-anchor':'end'});
                [item[1],item[2]].forEach((v,j)=>{
                    const bar=svgEl('rect',{x:ox+106,y:yy+j*16,width:v/max*300,height:10,rx:2,fill:j?C.red:C.navy,class:'chart-hbar'},svg);
                    animateMark(bar,pi*180+i*45+j*30,'rise');
                    text(svg,ox+112+v/max*300,yy+j*16+9,`${v.toFixed(1)} min`,'chart-value',{'text-anchor':'start'});
                });
            });
        });
        centeredLegend(svg,[{label:'mean',color:C.navy},{label:'median',color:C.red}],500,498);
    }

    const renderers={
        'fig-1':combinedProfile,'fig-2':outcomeBars,'fig-3':curves,'fig-4':categoryBars,
        'fig-5':dataTrajectory,'fig-causes':causeFan,'fig-evi':evidenceCurve,'fig-6':quadrants,'fig-7':repairErrors,'fig-8':hacking,'fig-9':scatter,'fig-10':timeBars
    };

    function animateChart(el, on) {
        if (on) {
            el.classList.remove('is-reset');
            requestAnimationFrame(() => el.classList.add('is-visible'));
        } else if (!REDUCE) {
            if (el.dataset.chart === 'fig-8') hideHackingTooltip(el);
            el.classList.add('is-reset');
            el.classList.remove('is-visible');
            void el.offsetWidth;
            el.classList.remove('is-reset');
        }
    }
    function init() {
        document.querySelectorAll('.blog-chart[data-chart]').forEach(el => {
            const fn=renderers[el.dataset.chart]; if (fn) fn(el);
        });
        const charts=[...document.querySelectorAll('.blog-chart[data-chart]')];
        if (!('IntersectionObserver' in window)) { charts.forEach(c=>c.classList.add('is-visible')); return; }
        const io=new IntersectionObserver(entries=>entries.forEach(entry=>animateChart(entry.target,entry.isIntersecting)),{threshold:.2,rootMargin:'0px 0px -8% 0px'});
        charts.forEach(chart=>io.observe(chart));
        const progress=document.querySelector('.blog-progress span');
        const updateProgress=()=>{const max=document.documentElement.scrollHeight-window.innerHeight; if(progress)progress.style.width=`${max>0?window.scrollY/max*100:0}%`;};
        addEventListener('scroll',updateProgress,{passive:true}); updateProgress();
        const counters=document.querySelectorAll('[data-count]');
        if ('IntersectionObserver' in window) { const ci=new IntersectionObserver((entries,observer)=>entries.forEach(e=>{if(!e.isIntersecting)return;const target=Number(e.target.dataset.count),start=performance.now();const tick=now=>{const t=Math.min((now-start)/720,1);e.target.textContent=Math.round((1-Math.pow(1-t,3))*target);if(t<1)requestAnimationFrame(tick);};requestAnimationFrame(tick);observer.unobserve(e.target);}),{threshold:.65}); counters.forEach(c=>ci.observe(c)); }
        const toc=[...document.querySelectorAll('.blog-toc a')], headings=toc.map(a=>document.querySelector(a.getAttribute('href'))).filter(Boolean); if ('IntersectionObserver' in window) {const ti=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)toc.forEach(a=>a.classList.toggle('is-active',a.getAttribute('href')===`#${e.target.id}`));}),{rootMargin:'-18% 0px -68% 0px'});headings.forEach(h=>ti.observe(h));}
    }
    if (document.readyState==='loading') document.addEventListener('DOMContentLoaded',init); else init();
})();
