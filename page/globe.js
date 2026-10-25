/* Aurora orbs — canvas-drawn glowing blobs that visibly drift around the hero.
   Each orb is independent: own velocity, size, colour, and pulse cadence.
   Respects prefers-reduced-motion. */
(function () {
  'use strict';

  const canvas = document.getElementById('globe-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ── Orb config ──────────────────────────────────────────────── */
  /* r: radius px | a: peak alpha | spd: px/s | col: [r,g,b] */
  const CFG = [
    { col: [160,  58,  50], r: 260, a: 0.60, spd: 55, ang: 0.62  },  // brick
    { col: [ 38,  60,  92], r: 240, a: 0.55, spd: 48, ang: 2.30  },  // navy
    { col: [175,  95,  38], r: 200, a: 0.50, spd: 62, ang: 3.90  },  // amber
    { col: [ 85,  55,  95], r: 210, a: 0.45, spd: 44, ang: 1.10  },  // mauve
    { col: [ 38, 100,  90], r: 190, a: 0.42, spd: 58, ang: 4.70  },  // teal
  ];

  let W = 0, H = 0, dpr = 1;

  /* Live state: position, velocity, pulse phase */
  const orbs = [];

  function initOrbs() {
    orbs.length = 0;
    CFG.forEach(c => {
      orbs.push({
        x:  Math.random() * W,
        y:  Math.random() * H,
        vx: Math.cos(c.ang) * c.spd,
        vy: Math.sin(c.ang) * c.spd,
        ph: Math.random() * Math.PI * 2,   // pulse phase
        cfg: c,
      });
    });
  }

  function resize() {
    dpr = window.devicePixelRatio || 1;
    W   = canvas.offsetWidth;
    H   = canvas.offsetHeight;
    canvas.width  = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (orbs.length === 0) initOrbs();
  }

  /* ── Render loop ─────────────────────────────────────────────── */
  let prev = null;

  function frame(ts) {
    const dt = prev === null ? 0 : Math.min((ts - prev) / 1000, 0.05);
    prev = ts;

    ctx.clearRect(0, 0, W, H);

    for (const o of orbs) {
      if (!reduced) {
        o.x  += o.vx * dt;
        o.y  += o.vy * dt;
        o.ph += dt * 0.9;   // ~7 s pulse cycle

        /* Bounce off edges (allow slight overshoot for smooth turnaround) */
        const margin = o.cfg.r * 0.3;
        if (o.x < -margin)      { o.x = -margin;      o.vx =  Math.abs(o.vx); }
        if (o.x >  W + margin)  { o.x =  W + margin;  o.vx = -Math.abs(o.vx); }
        if (o.y < -margin)      { o.y = -margin;      o.vy =  Math.abs(o.vy); }
        if (o.y >  H + margin)  { o.y =  H + margin;  o.vy = -Math.abs(o.vy); }
      }

      /* Pulsing radius: ±12 % of base */
      const rad  = o.cfg.r * (1 + 0.12 * Math.sin(o.ph));
      const peak = o.cfg.a;
      const [r, g, b] = o.cfg.col;

      /* Steep gradient: full alpha at centre, nearly gone by 60 % radius */
      const grad = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, rad);
      grad.addColorStop(0,    `rgba(${r},${g},${b},${peak.toFixed(2)})`);
      grad.addColorStop(0.30, `rgba(${r},${g},${b},${(peak * 0.55).toFixed(2)})`);
      grad.addColorStop(0.60, `rgba(${r},${g},${b},${(peak * 0.14).toFixed(2)})`);
      grad.addColorStop(1,    `rgba(${r},${g},${b},0)`);

      ctx.beginPath();
      ctx.arc(o.x, o.y, rad, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    }

    requestAnimationFrame(frame);
  }

  resize();
  window.addEventListener('resize', resize);
  requestAnimationFrame(frame);
}());
