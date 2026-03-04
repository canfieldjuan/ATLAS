/**
 * ╔══════════════════════════════════════════════════════╗
 * ║              ATLAS ROBOT  —  robot.js                ║
 * ║                                                      ║
 * ║  Pure SVG robot with GSAP animation methods.         ║
 * ║                                                      ║
 * ║  HOW TO CUSTOMIZE:                                   ║
 * ║  • Colors  → edit the COLORS object below            ║
 * ║  • Size    → pass scale: N to the constructor        ║
 * ║  • Shape   → edit the P (proportions) object below   ║
 * ║  • New animation → add a method that returns a       ║
 * ║    gsap.timeline()                                   ║
 * ╚══════════════════════════════════════════════════════╝
 *
 * COORDINATE SYSTEM
 *  • Robot origin  (0, 0) = feet center at ground level
 *  • Y is negative going UP  (standard math, not SVG)
 *  • Each limb <g> id="r-{name}" is positioned so its
 *    (0,0) is the JOINT / PIVOT POINT.
 *  • Children hang DOWN from (0,0) with positive-y rect coords.
 *  • transformOrigin '50% 0%' therefore rotates around the joint.
 */

// ─── Colors ──────────────────────────────────────────────
// Change these to restyle the robot without touching geometry
const COLORS = {
  body:    '#6a8caf',   // main body panels
  dark:    '#2e4a68',   // dark panels / joints
  light:   '#9abdd8',   // highlight panels
  joint:   '#1e3350',   // joint dots
  eye:     '#00e5ff',   // eyes + antenna ball (cyan)
  foot:    '#4d7090',   // feet
  chest1:  '#7aaed0',   // chest accent stripe 1
  chest2:  '#1e3350',   // chest accent stripe 2
};

// ─── Proportions ─────────────────────────────────────────
// All values in robot-local units (1 unit ≈ 1px at scale=1).
// Y=0 is feet-ground-level; negative Y is upward.
const P = {
  // ── Legs ──
  hipOffX:  11,     // hip joints are ±11 from center
  hipY:    -52,     // hip joint Y (negative = above ground)
  thighW:   12, thighH: 22, thighRx: 5,   // thigh hanging from hip
  shinW:    10, shinH:  22, shinRx:  5,   // shin hanging from knee
  footW:    16, footH:   7, footRx:  3,   // foot at bottom of shin

  // ── Torso ──
  torsoX: -18, torsoY: -94, torsoW: 36, torsoH: 34, torsoRx: 7,
  pelvisX: -15, pelvisY: -56, pelvisW: 30, pelvisH:  6, pelvisRx: 3,
  // chest detail stripes (relative to torsoX/Y origin in upper-body group)
  stripe1Y: -86, stripeW: 24, stripeH: 4,
  stripe2Y: -78,

  // ── Arms ──
  shoulderOffX: 20,  // shoulders at ±20 from center
  shoulderY:   -88,  // shoulder joint Y
  uArmW: 10, uArmH: 24, uArmRx: 5,  // upper arm hanging from shoulder
  fArmW:  8, fArmH: 19, fArmRx: 4,  // forearm hanging from elbow

  // ── Head ──
  headY:   -112,   // head center Y
  headR:    13,    // head radius
  eyeOffX:   4.5,  // eyes at ±4.5 from head center
  eyeY:     -2,    // eye Y relative to head center
  eyeR:      2.5,
  // antenna
  antLen:   12,    // stem length above head
  antBallR:  3.5,

  // ── Shadow ──
  shadowRx: 21, shadowRy: 6,
};


// ─────────────────────────────────────────────────────────
class Robot {
  /**
   * @param {SVGElement} mountEl  — <g> to attach robot to
   * @param {object}     opts
   *   x      {number}  initial x in scene coords   (default 0)
   *   y      {number}  ground y in scene coords     (default 300)
   *   scale  {number}  uniform scale factor         (default 1)
   *   flip   {boolean} start facing left            (default false)
   */
  constructor(mountEl, opts = {}) {
    this.mount   = mountEl;
    this.x       = opts.x     ?? 0;
    this.y       = opts.y     ?? 300;
    this.scale   = opts.scale ?? 1;
    this.dir     = opts.flip  ? -1 : 1;   // 1 = right, -1 = left

    this._build();

    // Pre-bake pivot transform-origins so every rotation animation
    // automatically uses the joint as the pivot point.
    gsap.set([
      this._e('left-thigh'),   this._e('right-thigh'),
      this._e('left-shin'),    this._e('right-shin'),
      this._e('left-shoulder'),this._e('right-shoulder'),
      this._e('left-elbow'),   this._e('right-elbow'),
    ], { transformOrigin: '50% 0%' });

    gsap.set(this._e('head'), { transformOrigin: '50% 50%' });

    // Place robot at initial position
    this._applyTransform();
  }

  // ── Private: get element by shortname ──────────────────
  _e(name) {
    return this.root.querySelector(`#r-${name}`);
  }

  // ── Private: update root SVG transform ─────────────────
  _applyTransform() {
    // Use a raw SVG transform attribute instead of GSAP CSS transforms.
    // This guarantees scale is always applied around (0,0) = feet,
    // no bounding-box transform-origin surprises.
    //  translate(x, y)   — moves feet to scene position
    //  scale(sx, sy)     — scales upward from the feet
    const sx = this.scale * this.dir;
    const sy = this.scale;
    this.root.setAttribute(
      'transform',
      `translate(${this.x}, ${this.y}) scale(${sx}, ${sy})`
    );
  }

  // ── Private: build all SVG elements ────────────────────
  _build() {
    const NS  = 'http://www.w3.org/2000/svg';
    const mk  = (tag, attrs) => {
      const el = document.createElementNS(NS, tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      return el;
    };
    // Shorthand helpers
    const g  = (id, tx = 0, ty = 0) =>
      mk('g', { id, transform: `translate(${tx},${ty})` });
    const rect = (x, y, w, h, rx, fill) =>
      mk('rect', { x, y, width: w, height: h, rx, fill });
    const circ = (cx, cy, r, fill) =>
      mk('circle', { cx, cy, r, fill });

    // ── Root group ──────────────────────────────────────
    this.root = mk('g', { id: 'r-root', filter: 'url(#body-shadow)' });
    this.mount.appendChild(this.root);

    // ── Ground shadow ────────────────────────────────────
    this.root.appendChild(mk('ellipse', {
      id: 'r-shadow',
      cx: 0, cy: 4,
      rx: P.shadowRx, ry: P.shadowRy,
      fill: 'rgba(0,0,0,0.28)',
    }));

    // ── Legs ─────────────────────────────────────────────
    ['left', 'right'].forEach(side => {
      const sx = side === 'left' ? -P.hipOffX : P.hipOffX;

      // HIP GROUP — pivot is the hip joint
      const hipG = mk('g', {
        id: `r-${side}-thigh`,
        transform: `translate(${sx}, ${P.hipY})`,
      });

      // Thigh shape (hangs down from hip pivot at y=0)
      hipG.appendChild(rect(-P.thighW / 2, 0, P.thighW, P.thighH, P.thighRx, 'url(#grad-body)'));

      // Knee joint indicator dot
      hipG.appendChild(circ(0, P.thighH, 4, COLORS.joint));

      // KNEE GROUP — pivot is the knee joint (bottom of thigh)
      const kneeG = mk('g', {
        id: `r-${side}-shin`,
        transform: `translate(0, ${P.thighH})`,
      });
      kneeG.appendChild(rect(-P.shinW / 2, 0, P.shinW, P.shinH, P.shinRx, 'url(#grad-dark)'));

      // Foot (attached at bottom of shin, offset toward front per side)
      const footOffX = side === 'left' ? -(P.footW - P.shinW / 2) + 2 : -2;
      kneeG.appendChild(rect(footOffX, P.shinH - P.footH + 1, P.footW, P.footH, P.footRx, COLORS.foot));

      hipG.appendChild(kneeG);
      this.root.appendChild(hipG);
    });

    // ── Upper body group (for independent Y bobbing) ─────
    this.upperBody = mk('g', { id: 'r-upper-body' });
    this.root.appendChild(this.upperBody);

    // Pelvis band
    this.upperBody.appendChild(
      rect(P.pelvisX, P.pelvisY, P.pelvisW, P.pelvisH, P.pelvisRx, COLORS.dark)
    );

    // Torso
    this.upperBody.appendChild(
      rect(P.torsoX, P.torsoY, P.torsoW, P.torsoH, P.torsoRx, 'url(#grad-body)')
    );

    // Chest accent stripes (small panel detail)
    this.upperBody.appendChild(
      rect(P.torsoX + 6, P.stripe1Y, P.stripeW, P.stripeH, 2, COLORS.chest1)
    );
    this.upperBody.appendChild(
      rect(P.torsoX + 6, P.stripe2Y, P.stripeW, P.stripeH, 2, COLORS.chest2)
    );

    // ── Arms ─────────────────────────────────────────────
    ['left', 'right'].forEach(side => {
      const sx = side === 'left' ? -P.shoulderOffX : P.shoulderOffX;

      // Shoulder joint indicator dot (rendered BEHIND arm so it looks like a socket)
      this.upperBody.appendChild(circ(sx, P.shoulderY, 5.5, COLORS.joint));

      // SHOULDER GROUP — pivot is the shoulder joint
      const shoulderG = mk('g', {
        id:        `r-${side}-shoulder`,
        transform: `translate(${sx}, ${P.shoulderY})`,
      });

      // Upper arm (hangs down from shoulder)
      shoulderG.appendChild(
        rect(-P.uArmW / 2, 0, P.uArmW, P.uArmH, P.uArmRx, 'url(#grad-body)')
      );

      // Elbow joint dot
      shoulderG.appendChild(circ(0, P.uArmH, 3.5, COLORS.joint));

      // ELBOW GROUP — pivot is the elbow joint (bottom of upper arm)
      const elbowG = mk('g', {
        id:        `r-${side}-elbow`,
        transform: `translate(0, ${P.uArmH})`,
      });
      elbowG.appendChild(
        rect(-P.fArmW / 2, 0, P.fArmW, P.fArmH, P.fArmRx, 'url(#grad-dark)')
      );

      shoulderG.appendChild(elbowG);
      this.upperBody.appendChild(shoulderG);
    });

    // ── Head ─────────────────────────────────────────────
    const headG = mk('g', { id: 'r-head', transform: `translate(0, ${P.headY})` });

    // Head circle
    headG.appendChild(circ(0, 0, P.headR, 'url(#grad-body)'));

    // Eyes (with cyan glow)
    ['left', 'right'].forEach(side => {
      const ex = side === 'left' ? -P.eyeOffX : P.eyeOffX;
      const eye = circ(ex, P.eyeY, P.eyeR, COLORS.eye);
      eye.setAttribute('id', `r-eye-${side === 'left' ? 'l' : 'r'}`);
      eye.setAttribute('filter', 'url(#glow-cyan)');
      headG.appendChild(eye);
    });

    // Antenna stem
    headG.appendChild(mk('line', {
      x1: 0, y1: -P.headR,
      x2: 0, y2: -P.headR - P.antLen,
      stroke: COLORS.dark,
      'stroke-width': 2,
      'stroke-linecap': 'round',
    }));

    // Antenna ball (with cyan glow)
    const antBall = circ(0, -P.headR - P.antLen - P.antBallR, P.antBallR, COLORS.eye);
    antBall.setAttribute('id', 'r-antenna-ball');
    antBall.setAttribute('filter', 'url(#glow-cyan)');
    headG.appendChild(antBall);

    this.upperBody.appendChild(headG);
  }


  // ═══════════════════════════════════════════════════════
  //  ANIMATION METHODS
  //  Each method returns a gsap.timeline() so it can be
  //  sequenced with .add() in the scene choreography.
  // ═══════════════════════════════════════════════════════

  /**
   * Snap all limbs + upper body back to neutral pose.
   * duration: how long the reset takes (seconds)
   */
  reset(duration = 0.25) {
    const tl = gsap.timeline();
    // Reset limbs (pivot-anchored groups)
    tl.to(
      [
        this._e('left-thigh'),    this._e('right-thigh'),
        this._e('left-shin'),     this._e('right-shin'),
        this._e('left-shoulder'), this._e('right-shoulder'),
        this._e('left-elbow'),    this._e('right-elbow'),
        this._e('head'),
      ],
      { rotation: 0, duration, ease: 'power2.out' },
      0
    );
    // Reset upper-body translate + rotation separately
    tl.to(
      this._e('upper-body'),
      { x: 0, y: 0, rotation: 0, duration, ease: 'power2.out' },
      0
    );
    return tl;
  }

  /**
   * Idle: gentle body bob + head sway + antenna wobble.
   * Returns ONE master timeline — call .kill() on it to stop all sub-animations.
   * @param {number} rate  — speed multiplier (1 = normal)
   */
  idle(rate = 1) {
    const base    = 0.85 / rate;
    const master  = gsap.timeline();
    const body    = gsap.timeline({ repeat: -1 });
    const headTl  = gsap.timeline({ repeat: -1, delay: 0.15 });
    const antTl   = gsap.timeline({ repeat: -1, delay: 0.35 });
    const armTl   = gsap.timeline({ repeat: -1, delay: 0.5 });

    // Body bob — asymmetric timing (up faster, down slower → breathing feel)
    body.to(this._e('upper-body'), { y: -3.5, duration: base * 0.85, ease: 'sine.out' })
        .to(this._e('upper-body'), { y:  0.5, duration: base * 1.15, ease: 'sine.in' })
        .to(this._e('upper-body'), { y:  0,   duration: base * 0.3,  ease: 'sine.out' });

    // Head sway — wider on one side (personality)
    headTl.to(this._e('head'), { rotation:  5,  duration: base * 1.4, ease: 'sine.inOut' })
          .to(this._e('head'), { rotation: -3,  duration: base * 1.2, ease: 'sine.inOut' })
          .to(this._e('head'), { rotation:  0,  duration: base * 0.6, ease: 'sine.inOut' });

    // Antenna wobble — slightly different period than head for polyrhythm
    antTl.to(this._e('antenna-ball'), { y: -2.5, duration: base * 0.6, ease: 'sine.out' })
         .to(this._e('antenna-ball'), { y:  2,   duration: base * 0.8, ease: 'sine.inOut' })
         .to(this._e('antenna-ball'), { y:  0,   duration: base * 0.5, ease: 'sine.in' });

    // Subtle arm sway — very small, like arms hanging loose
    const lSh = this._e('left-shoulder');
    const rSh = this._e('right-shoulder');
    armTl.to(lSh, { rotation:  3, duration: base * 1.8, ease: 'sine.inOut' })
         .to(lSh, { rotation: -2, duration: base * 1.5, ease: 'sine.inOut' })
         .to(lSh, { rotation:  0, duration: base * 0.8, ease: 'sine.inOut' });
    // Right arm on slightly different timing
    armTl.to(rSh, { rotation: -2, duration: base * 1.6, ease: 'sine.inOut' }, 0.3)
         .to(rSh, { rotation:  3, duration: base * 1.4, ease: 'sine.inOut' }, base * 1.6 + 0.3)
         .to(rSh, { rotation:  0, duration: base * 0.9, ease: 'sine.inOut' }, base * 3.0 + 0.3);

    // Add all to master so .kill() on master stops everything
    master.add(body, 0).add(headTl, 0).add(antTl, 0).add(armTl, 0);
    return master;
  }

  /**
   * Walk to targetX at a given speed.
   * Implements:
   *  - Per-step body bob (up at mid-stride, down at foot-plant)
   *  - Torso lean in direction of travel
   *  - Head lag / follow-through
   *  - Opposite arm-leg counter-swing with elbow bend
   *  - Knee flex on plant leg
   *  - Ease-in at walk start, ease-out at walk end
   *
   * @param {number} targetX  — destination x in scene coords
   * @param {number} speed    — px per second (default 110)
   */
  walk(targetX, speed = 110) {
    const dist = Math.abs(targetX - this.x);
    if (dist < 1) return gsap.timeline();

    const duration = dist / speed;
    const newDir   = targetX > this.x ? 1 : -1;

    // Flip direction immediately
    if (newDir !== this.dir) {
      this.dir = newDir;
      this._applyTransform();
    }

    const self  = this;
    const proxy = { x: this.x };
    const tl    = gsap.timeline({
      onComplete() {
        self.x = targetX;
        self._applyTransform();
      },
    });

    // ── Translation with subtle ease-in / ease-out ──────
    tl.to(proxy, {
      x:        targetX,
      duration,
      ease:     'power1.inOut',       // gentle accel → decel
      onUpdate() {
        self.x = proxy.x;
        self._applyTransform();
      },
    }, 0);

    // ── Torso lean forward on entry, upright at end ─────
    const ub = this._e('upper-body');
    tl.to(ub, { rotation: 4, duration: 0.25, ease: 'power2.out' }, 0);
    tl.to(ub, { rotation: 0, duration: 0.30, ease: 'power2.inOut' }, duration - 0.30);

    // ── Head lag: arrives ~0.08s after body starts/stops ─
    const head = this._e('head');
    tl.to(head, { rotation: -6, duration: 0.15, ease: 'power2.out' }, 0.08);
    tl.to(head, { rotation:  0, duration: 0.20, ease: 'power2.inOut' }, duration - 0.25);

    // ── Step cycles ──────────────────────────────────────
    const stepDur = 0.36;     // slightly faster at this scale for snap
    const steps   = Math.ceil(duration / stepDur);
    for (let i = 0; i < steps; i++) {
      // Alternating: even = left-forward, odd = right-forward
      const leftFirst = i % 2 === 0;
      tl.add(this._stepCycle(leftFirst, stepDur), i * stepDur);
    }

    // ── Body bob (up at mid-stride, down at foot-plant) ─
    // Two bobs per step cycle
    for (let i = 0; i < steps; i++) {
      const t = i * stepDur;
      // Up at mid-stride (slight)
      tl.to(ub, { y: -3, duration: stepDur * 0.5, ease: 'sine.out' }, t);
      // Down at foot-plant
      tl.to(ub, { y: 1,  duration: stepDur * 0.5, ease: 'sine.in' }, t + stepDur * 0.5);
    }
    // Settle body back to 0 at end
    tl.to(ub, { y: 0, duration: 0.15, ease: 'power2.out' }, duration - 0.15);

    return tl;
  }

  /**
   * Single step: one leg swings forward, other pushes back.
   * Includes opposite arm swing, elbow bend, knee flex.
   * @param {boolean} leftFirst — true: left leg leads, false: right
   * @param {number}  dur       — full step duration (seconds)
   */
  _stepCycle(leftFirst = true, dur = 0.36) {
    // Refs
    const lTh = this._e('left-thigh'),  rTh = this._e('right-thigh');
    const lSn = this._e('left-shin'),   rSn = this._e('right-shin');
    const lSh = this._e('left-shoulder'), rSh = this._e('right-shoulder');
    const lEl = this._e('left-elbow'),    rEl = this._e('right-elbow');

    // Which limbs lead vs trail
    const leadTh  = leftFirst ? lTh : rTh;
    const trailTh = leftFirst ? rTh : lTh;
    const leadSn  = leftFirst ? lSn : rSn;
    const trailSn = leftFirst ? rSn : lSn;
    const leadArm = leftFirst ? rSh : lSh;  // opposite arm
    const trailArm= leftFirst ? lSh : rSh;
    const leadElb = leftFirst ? rEl : lEl;
    const trailElb= leftFirst ? lEl : rEl;

    // Slight randomness for organic feel (±12%)
    const jitter = () => 0.88 + Math.random() * 0.24;
    const half   = dur * 0.5;

    // Swing angles — lead leg forward (negative rotation lifts forward)
    const thighSwing = 25 * jitter();
    const trailSwing = 20 * jitter();
    const armSwing   = 18 * jitter();

    const tl = gsap.timeline();

    // ── CONTACT → PASSING  (first half of step) ─────────
    // Lead leg: swing forward
    tl.to(leadTh,  { rotation: -thighSwing, duration: half, ease: 'sine.out' }, 0);
    // Lead shin: slight knee flex as it reaches forward
    tl.to(leadSn,  { rotation: 8,   duration: half * 0.6, ease: 'sine.out' }, 0);
    tl.to(leadSn,  { rotation: -6,  duration: half * 0.4, ease: 'sine.in' }, half * 0.6);

    // Trail leg: extend backward
    tl.to(trailTh, { rotation: trailSwing, duration: half, ease: 'sine.out' }, 0);
    // Trail shin: push off (slight flex)
    tl.to(trailSn, { rotation: -10, duration: half, ease: 'sine.out' }, 0);

    // Arms counter-swing (opposite to legs, classic locomotion)
    tl.to(leadArm,  { rotation: -armSwing, duration: half, ease: 'sine.inOut' }, 0);
    tl.to(trailArm, { rotation:  armSwing, duration: half, ease: 'sine.inOut' }, 0);
    // Elbow bends on back-swing arm
    tl.to(trailElb, { rotation: 18 * jitter(), duration: half, ease: 'sine.out' }, 0);
    tl.to(leadElb,  { rotation: 5, duration: half, ease: 'sine.out' }, 0);

    // ── PASSING → NEXT CONTACT  (second half) ───────────
    // Lead leg passes through to plant
    tl.to(leadTh,   { rotation: 4,  duration: half, ease: 'sine.in' }, half);
    tl.to(leadSn,   { rotation: 12, duration: half * 0.7, ease: 'sine.out' }, half);
    tl.to(leadSn,   { rotation: 3,  duration: half * 0.3, ease: 'sine.in' }, half + half * 0.7);

    // Trail leg swings forward for next step
    tl.to(trailTh,  { rotation: -6, duration: half, ease: 'sine.in' }, half);
    tl.to(trailSn,  { rotation: 4,  duration: half, ease: 'sine.in' }, half);

    // Arms return toward neutral
    tl.to(leadArm,  { rotation: 6,  duration: half, ease: 'sine.inOut' }, half);
    tl.to(trailArm, { rotation: -6, duration: half, ease: 'sine.inOut' }, half);
    // Elbows relax
    tl.to(trailElb, { rotation: 3, duration: half, ease: 'sine.inOut' }, half);
    tl.to(leadElb,  { rotation: 2, duration: half, ease: 'sine.inOut' }, half);

    return tl;
  }

  /**
   * Jump: crouch → launch → tuck mid-air → land → bounce.
   * @param {number} height  — how high the robot jumps (px, default 55)
   * @param {number} type    — 0: normal  1: spin  2: wide
   */
  jump(height = 55, type = 0) {
    const origY  = this.y;
    const apex   = origY - height;
    const self   = this;
    const proxy  = { y: origY };

    const tl = gsap.timeline({
      onComplete: () => { self.y = origY; self._applyTransform(); },
    });

    const moveY = (toY, dur, ease) =>
      tl.to(proxy, {
        y: toY,
        duration: dur,
        ease,
        onUpdate() {
          self.y = proxy.y;
          self._applyTransform();
        },
      });

    // Crouch
    tl.to(this._e('upper-body'),                         { y: 5,  duration: 0.1, ease: 'power2.out' })
      .to([this._e('left-thigh'), this._e('right-thigh')], { rotation: 18, duration: 0.1, ease: 'power2.out' }, '<')
      .to([this._e('left-shin'),  this._e('right-shin')],   { rotation: 15, duration: 0.1, ease: 'power2.out' }, '<');

    // Launch upward
    moveY(apex, 0.28, 'power2.out');
    tl.to(this._e('upper-body'),                              { y: 0, duration: 0.25 }, '<')
      .to([this._e('left-thigh'), this._e('right-thigh')],   { rotation: 0, duration: 0.1 }, '<')
      .to([this._e('left-shin'),  this._e('right-shin')],    { rotation: 0, duration: 0.1 }, '<');

    // Mid-air pose by type
    if (type === 1) {
      // Spin arms
      tl.to([this._e('left-shoulder'), this._e('right-shoulder')],
        { rotation: 180, duration: 0.28, ease: 'none' }, '<0.05');
    } else if (type === 2) {
      // Wide legs
      tl.to([this._e('left-thigh'), this._e('right-thigh')],
        { rotation: -35, duration: 0.15, ease: 'power1.inOut' }, '<0.05')
        .to([this._e('left-shoulder'), this._e('right-shoulder')],
          { rotation: -90, duration: 0.15 }, '<');
    } else {
      // Normal tuck
      tl.to([this._e('left-thigh'), this._e('right-thigh')],
        { rotation: -28, duration: 0.15, ease: 'power1.inOut' })
        .to([this._e('left-shin'),  this._e('right-shin')],
          { rotation: 35, duration: 0.15 }, '<');
    }

    // Descend
    moveY(origY, 0.26, 'power2.in');
    tl.to([this._e('left-thigh'),    this._e('right-thigh')],
        { rotation: 0, duration: 0.18 }, '<')
      .to([this._e('left-shin'),     this._e('right-shin')],
        { rotation: 0, duration: 0.18 }, '<')
      .to([this._e('left-shoulder'), this._e('right-shoulder')],
        { rotation: 0, duration: 0.18 }, '<');

    // Land bounce
    tl.to(this._e('upper-body'), { y: 8, duration: 0.07, ease: 'power2.in' })
      .to(this._e('upper-body'), { y: 0, duration: 0.25, ease: 'elastic.out(1, 0.4)' });

    return tl;
  }

  /**
   * Wave right arm (or left arm if side='left').
   * Now with anticipation, overshoot, head turn, and body weight shift.
   * @param {number} repeats  — number of waves  (default 2)
   * @param {string} side     — 'right' | 'left'
   */
  wave(repeats = 2, side = 'right') {
    const arm   = this._e(`${side}-shoulder`);
    const elbow = this._e(`${side}-elbow`);
    const head  = this._e('head');
    const ub    = this._e('upper-body');
    const tl    = gsap.timeline();

    // Anticipation: slight counter-dip before raising arm
    tl.to(arm, { rotation: 8, duration: 0.08, ease: 'power2.in' });

    // Raise arm up with overshoot
    tl.to(arm,   { rotation: -125, duration: 0.28, ease: 'back.out(1.4)' });
    // Slight body lean away from waving arm
    tl.to(ub, { x: side === 'right' ? -2 : 2, duration: 0.2, ease: 'sine.out' }, '<0.05');
    // Head turns toward the wave
    tl.to(head, { rotation: side === 'right' ? 8 : -8, duration: 0.22, ease: 'power2.out' }, '<');

    // Wave the forearm — with accelerating then slowing arcs
    for (let i = 0; i < repeats; i++) {
      const speed = 0.12 + i * 0.02;  // slightly slower each wave
      tl.to(elbow, { rotation: 40,  duration: speed, ease: 'sine.out' })
        .to(elbow, { rotation: -15, duration: speed, ease: 'sine.in' });
    }

    // Lower arm back with follow-through
    tl.to(elbow, { rotation: 0, duration: 0.18, ease: 'power2.inOut' });
    tl.to(arm,   { rotation: 5, duration: 0.25, ease: 'power2.in' }, '<0.05');
    tl.to(arm,   { rotation: 0, duration: 0.15, ease: 'sine.out' });
    // Body + head back to neutral
    tl.to(ub,   { x: 0, duration: 0.2, ease: 'sine.inOut' }, '<');
    tl.to(head, { rotation: 0, duration: 0.2, ease: 'sine.inOut' }, '<');

    return tl;
  }

  /**
   * Full dance routine (~3s of moves) with organic flow.
   * Includes:
   *  - Wind-up anticipation before each move
   *  - Overlapping action (body arrives first, head follows)
   *  - Asymmetric timing (not everything on the same beat)
   *  - Weight-shift shimmies
   *  - Follow-through on arm drops
   */
  dance() {
    const lSh  = this._e('left-shoulder');
    const rSh  = this._e('right-shoulder');
    const lEl  = this._e('left-elbow');
    const rEl  = this._e('right-elbow');
    const ub   = this._e('upper-body');
    const lTh  = this._e('left-thigh');
    const rTh  = this._e('right-thigh');
    const lSn  = this._e('left-shin');
    const rSn  = this._e('right-shin');
    const head = this._e('head');

    const tl = gsap.timeline();

    // ── Anticipation dip ─────────────────────────────────
    tl.to(ub, { y: 4, duration: 0.1, ease: 'power2.in' });

    // ── Beat 1 — both arms pop up with overshoot ─────────
    tl.to(ub, { y: -2, duration: 0.12, ease: 'power3.out' })
      .to([lSh, rSh], { rotation: -110, duration: 0.2, ease: 'back.out(2.5)' }, '<')
      .to([lEl, rEl], { rotation: 25, duration: 0.18 }, '<0.04')
      // Head arrives 0.05s late (overlap)
      .to(head, { rotation: -5, y: -1.5, duration: 0.12, ease: 'power2.out' }, '<0.05');

    // ── Beat 2 — asymmetric shimmy right ─────────────────
    // Body leads, legs follow 0.03s later, head trails 0.06s
    tl.to(ub,   { x: 8, rotation: 3, duration: 0.16, ease: 'power2.out' })
      .to(lTh,  { rotation: -20, duration: 0.15, ease: 'sine.out' }, '<0.03')
      .to(rTh,  { rotation: 15, duration: 0.15, ease: 'sine.out' }, '<')
      .to(lSn,  { rotation: 8,  duration: 0.12 }, '<0.02')
      .to(head, { rotation: 12, y: 0, duration: 0.14, ease: 'power2.out' }, '<0.06');

    // ── Beat 3 — shimmy left (faster, more energy) ───────
    tl.to(ub,   { x: -8, rotation: -3, duration: 0.14, ease: 'power2.out' })
      .to(lTh,  { rotation: 15, duration: 0.13, ease: 'sine.out' }, '<0.03')
      .to(rTh,  { rotation: -20, duration: 0.13, ease: 'sine.out' }, '<')
      .to(rSn,  { rotation: 8, duration: 0.10 }, '<0.02')
      .to(head, { rotation: -12, duration: 0.12, ease: 'power2.out' }, '<0.05');

    // ── Beat 4 — shimmy right again, bounce ──────────────
    tl.to(ub,   { x: 6, rotation: 2, duration: 0.14, ease: 'power2.out' })
      .to(lTh,  { rotation: -12, duration: 0.12 }, '<0.02')
      .to(rTh,  { rotation: 10, duration: 0.12 }, '<')
      .to(head, { rotation: 8, duration: 0.12 }, '<0.04');

    // ── Beat 5 — snap to center + arms drop with follow-through ─
    tl.to(ub,   { x: 0, rotation: 0, duration: 0.16, ease: 'power2.inOut' })
      .to(head,  { rotation: 0, duration: 0.18, ease: 'power2.inOut' }, '<0.03')
      // Arms drop with overshoot (go past neutral then spring back)
      .to([lSh, rSh], { rotation: 12, duration: 0.18, ease: 'power3.in' }, '<')
      .to([lSh, rSh], { rotation: -4, duration: 0.12, ease: 'sine.out' })
      .to([lSh, rSh], { rotation: 0,  duration: 0.08 })
      .to([lEl, rEl], { rotation: 0, duration: 0.14, ease: 'power2.out' }, '<0.04')
      .to([lTh, rTh], { rotation: 0, duration: 0.16 }, '<')
      .to([lSn, rSn], { rotation: 0, duration: 0.14 }, '<');

    // ── Windmill (staggered start per arm for organic feel) ──
    tl.to(lSh, { rotation: -360, duration: 0.55, ease: 'power1.inOut' })
      .to(rSh, { rotation:  360, duration: 0.55, ease: 'power1.inOut' }, '<0.06')
      // Body leans back slightly during windmill
      .to(ub, { rotation: -3, duration: 0.25, ease: 'sine.out' }, '<')
      .to(ub, { rotation:  0, duration: 0.25, ease: 'sine.in' }, '<0.25')
      .add(() => { gsap.set([lSh, rSh], { rotation: 0 }); });

    // ── Mini jump with weight feel ─────────────────────
    const origY  = this.y;
    const self   = this;
    const proxy  = { y: origY };
    // Crouch before jump
    tl.to(ub, { y: 5, duration: 0.08, ease: 'power2.in' });
    tl.to(proxy, {
      y: origY - 28,
      duration: 0.22,
      ease: 'power2.out',
      onUpdate() { self.y = proxy.y; self._applyTransform(); },
    });
    tl.to(ub, { y: -2, duration: 0.15 }, '<');
    tl.to(proxy, {
      y: origY,
      duration: 0.26,
      ease: 'bounce.out',
      onUpdate() { self.y = proxy.y; self._applyTransform(); },
    });
    // Land settle
    tl.to(ub, { y: 6, duration: 0.06, ease: 'power2.in' })
      .to(ub, { y: 0, duration: 0.20, ease: 'elastic.out(1, 0.5)' });

    return tl;
  }

  /**
   * Sit down: legs fold forward, body sinks.
   * @param {number} sitDuration  — seconds to stay seated before standing (0=stay)
   */
  sit(sitDuration = 0) {
    const tl = gsap.timeline();

    tl.to([this._e('left-thigh'), this._e('right-thigh')],
        { rotation: -60, duration: 0.4, ease: 'power2.out' })
      .to([this._e('left-shin'),  this._e('right-shin')],
        { rotation:  80, duration: 0.4 }, '<')
      .to(this._e('upper-body'), { y: 14, duration: 0.4 }, '<');

    if (sitDuration > 0) {
      tl.to({}, { duration: sitDuration }); // hold
      tl.to([this._e('left-thigh'), this._e('right-thigh')],
          { rotation: 0, duration: 0.35, ease: 'power2.out' })
        .to([this._e('left-shin'),  this._e('right-shin')],
          { rotation: 0, duration: 0.35 }, '<')
        .to(this._e('upper-body'), { y: 0, duration: 0.35 }, '<');
    }

    return tl;
  }

  /**
   * Look in a direction by rotating head.
   * @param {string} dir  — 'left' | 'right' | 'up' | 'reset'
   */
  look(dir) {
    const angles = { left: -20, right: 20, up: -15, reset: 0 };
    return gsap.timeline().to(this._e('head'), {
      rotation: angles[dir] ?? 0,
      duration: 0.25,
      ease: 'power2.inOut',
    });
  }

  /**
   * Turn to face a direction.
   * @param {number} newDir  — 1 (right) or -1 (left)
   */
  turn(newDir) {
    if (newDir === this.dir) return gsap.timeline();
    const self  = this;
    const proxy = { sx: this.scale * this.dir };
    const tl    = gsap.timeline({
      onComplete: () => { self.dir = newDir; self._applyTransform(); },
    });
    // Squash to 0
    tl.to(proxy, {
      sx: 0,
      duration: 0.12,
      ease: 'power2.in',
      onUpdate() {
        self.root.setAttribute('transform',
          `translate(${self.x}, ${self.y}) scale(${proxy.sx}, ${self.scale})`);
      },
    })
    // Expand in new direction
    .to(proxy, {
      sx: self.scale * newDir,
      duration: 0.12,
      ease: 'power2.out',
      onUpdate() {
        self.root.setAttribute('transform',
          `translate(${self.x}, ${self.y}) scale(${proxy.sx}, ${self.scale})`);
      },
    });
    // Update dir immediately at midpoint (so the expand goes the right way)
    tl.add(() => { self.dir = newDir; }, 0.12);
    return tl;
  }

  /**
   * Surprised: arms fling up, body jerks back, with stagger and settle.
   */
  surprise() {
    const lSh  = this._e('left-shoulder');
    const rSh  = this._e('right-shoulder');
    const lEl  = this._e('left-elbow');
    const rEl  = this._e('right-elbow');
    const head = this._e('head');
    const ub   = this._e('upper-body');

    const tl = gsap.timeline();

    // Body snaps back, head leads with exaggerated recoil
    tl.to(head, { rotation: 12, y: -3, duration: 0.08, ease: 'power4.out' })
      .to(ub,   { y: -6, rotation: -4, duration: 0.12, ease: 'power4.out' }, '<0.02');

    // Arms fling up (staggered — left first for asymmetry)
    tl.to(lSh, { rotation: -145, duration: 0.12, ease: 'power4.out' }, 0)
      .to(rSh, { rotation: -125, duration: 0.14, ease: 'power4.out' }, 0.03)
      .to(lEl, { rotation: 20, duration: 0.10 }, '<')
      .to(rEl, { rotation: 30, duration: 0.12 }, '<');

    // Hold the surprise beat
    tl.to({}, { duration: 0.15 });

    // Settle: elastic recovery on arms, body sinks down
    tl.to(lSh, { rotation: 0, duration: 0.5, ease: 'elastic.out(1, 0.4)' })
      .to(rSh, { rotation: 0, duration: 0.55, ease: 'elastic.out(1, 0.35)' }, '<0.04')
      .to([lEl, rEl], { rotation: 0, duration: 0.3, ease: 'power2.out' }, '<')
      .to(ub,   { y: 2, rotation: 0, duration: 0.15, ease: 'power2.in' }, '<')
      .to(ub,   { y: 0, duration: 0.20, ease: 'sine.out' })
      .to(head, { rotation: 0, y: 0, duration: 0.25, ease: 'power2.inOut' }, '<');

    return tl;
  }

  /**
   * Thinking pose: one arm raised to chin, head tilted.
   * Now with body-weight shift, subtle micro-movements while thinking.
   * @param {number} duration  — how long to hold the pose (seconds)
   */
  think(duration = 1.2) {
    const rSh  = this._e('right-shoulder');
    const rEl  = this._e('right-elbow');
    const head = this._e('head');
    const ub   = this._e('upper-body');
    const tl   = gsap.timeline();

    // Shift weight to left (leaning into thought)
    tl.to(ub,   { x: -3, rotation: -2, duration: 0.3, ease: 'power2.out' })
      .to(rSh,  { rotation: -80, duration: 0.32, ease: 'back.out(1.3)' }, '<0.05')
      .to(rEl,  { rotation:  65, duration: 0.30, ease: 'power2.out' }, '<0.04')
      .to(head, { rotation: -14, duration: 0.28, ease: 'power2.out' }, '<0.08');

    // Subtle micro-nod while thinking (alive feel)
    if (duration > 0.4) {
      const midT = tl.duration();
      tl.to(head, { rotation: -10, duration: duration * 0.3, ease: 'sine.inOut' }, midT);
      tl.to(head, { rotation: -16, duration: duration * 0.3, ease: 'sine.inOut' }, midT + duration * 0.3);
      tl.to(head, { rotation: -12, duration: duration * 0.4, ease: 'sine.inOut' }, midT + duration * 0.6);
    } else {
      tl.to({}, { duration });
    }

    // Un-think (head leads exit, arm follows through)
    tl.to(head, { rotation: 0, duration: 0.22, ease: 'power2.inOut' })
      .to(rSh,  { rotation: 5, duration: 0.28, ease: 'power2.in' }, '<0.06')
      .to(rSh,  { rotation: 0, duration: 0.12, ease: 'sine.out' })
      .to(rEl,  { rotation: 0, duration: 0.25, ease: 'power2.inOut' }, '<')
      .to(ub,   { x: 0, rotation: 0, duration: 0.22, ease: 'sine.inOut' }, '<');

    return tl;
  }

  /**
   * Victory pose: jump + both arms up + organic wiggle + settle.
   */
  victory() {
    const lSh  = this._e('left-shoulder');
    const rSh  = this._e('right-shoulder');
    const ub   = this._e('upper-body');
    const head = this._e('head');
    const tl   = gsap.timeline();

    tl.add(this.jump(70, 1));

    // Arms shoot up with one slightly higher (asymmetric = organic)
    tl.to(lSh, { rotation: -130, duration: 0.18, ease: 'back.out(2)' }, '-=0.3');
    tl.to(rSh, { rotation: -115, duration: 0.20, ease: 'back.out(1.8)' }, '-=0.32');
    tl.to(head, { rotation: -8, y: -1, duration: 0.15 }, '<0.04');

    // Hold for applause
    tl.to({}, { duration: 0.3 });

    // Organic shimmy-wiggle (decelerating, like energy running out)
    tl.to(ub, { rotation: 10,  duration: 0.09, ease: 'power2.out' })
      .to(ub, { rotation: -9,  duration: 0.09 })
      .to(ub, { rotation: 7,   duration: 0.10 })
      .to(ub, { rotation: -5,  duration: 0.11 })
      .to(ub, { rotation: 2,   duration: 0.12, ease: 'sine.out' })
      .to(ub, { rotation: 0,   duration: 0.10, ease: 'sine.out' });

    // Head follows wiggle with slight delay
    tl.to(head, { rotation: 5, duration: 0.08 }, '-=0.60');
    tl.to(head, { rotation: -4, duration: 0.08 }, '-=0.50');
    tl.to(head, { rotation: 0, y: 0, duration: 0.15, ease: 'sine.out' }, '-=0.25');

    // Arms lower with follow-through (overshoot then settle)
    tl.to(lSh, { rotation: 8,  duration: 0.22, ease: 'power3.in' }, '-=0.15');
    tl.to(rSh, { rotation: 6,  duration: 0.24, ease: 'power3.in' }, '<0.03');
    tl.to([lSh, rSh], { rotation: -3, duration: 0.10, ease: 'sine.out' });
    tl.to([lSh, rSh], { rotation: 0,  duration: 0.08 });

    return tl;
  }
}
