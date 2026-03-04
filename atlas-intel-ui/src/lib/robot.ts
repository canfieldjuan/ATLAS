/**
 * Atlas Robot -- ESM port of animated-robot-logo/js/robot.js
 *
 * Pure SVG robot with GSAP animation methods.
 * Coordinate system: origin (0,0) = feet center at ground level.
 * Y is negative going UP.
 */

import gsap from 'gsap'

const NS = 'http://www.w3.org/2000/svg'

const COLORS = {
  body: '#6a8caf',
  dark: '#2e4a68',
  light: '#9abdd8',
  joint: '#1e3350',
  eye: '#00e5ff',
  foot: '#4d7090',
  chest1: '#7aaed0',
  chest2: '#1e3350',
}

const P = {
  hipOffX: 11, hipY: -52,
  thighW: 12, thighH: 22, thighRx: 5,
  shinW: 10, shinH: 22, shinRx: 5,
  footW: 16, footH: 7, footRx: 3,
  torsoX: -18, torsoY: -94, torsoW: 36, torsoH: 34, torsoRx: 7,
  pelvisX: -15, pelvisY: -56, pelvisW: 30, pelvisH: 6, pelvisRx: 3,
  stripe1Y: -86, stripeW: 24, stripeH: 4, stripe2Y: -78,
  shoulderOffX: 20, shoulderY: -88,
  uArmW: 10, uArmH: 24, uArmRx: 5,
  fArmW: 8, fArmH: 19, fArmRx: 4,
  headY: -112, headR: 13,
  eyeOffX: 4.5, eyeY: -2, eyeR: 2.5,
  antLen: 12, antBallR: 3.5,
  shadowRx: 21, shadowRy: 6,
}

function mk(tag: string, attrs: Record<string, string | number>): SVGElement {
  const el = document.createElementNS(NS, tag)
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)))
  return el
}

function rect(x: number, y: number, w: number, h: number, rx: number, fill: string) {
  return mk('rect', { x, y, width: w, height: h, rx, fill })
}

function circ(cx: number, cy: number, r: number, fill: string) {
  return mk('circle', { cx, cy, r, fill })
}

interface RobotOpts {
  x?: number
  y?: number
  scale?: number
  flip?: boolean
}

export class Robot {
  mount: SVGElement
  x: number
  y: number
  scale: number
  dir: number
  root!: SVGElement
  upperBody!: SVGElement
  destroyed = false
  private _activeTl: gsap.core.Timeline | null = null

  constructor(mountEl: SVGElement, opts: RobotOpts = {}) {
    this.mount = mountEl
    this.x = opts.x ?? 0
    this.y = opts.y ?? 300
    this.scale = opts.scale ?? 1
    this.dir = opts.flip ? -1 : 1

    this._build()

    gsap.set([
      this._e('left-thigh'), this._e('right-thigh'),
      this._e('left-shin'), this._e('right-shin'),
      this._e('left-shoulder'), this._e('right-shoulder'),
      this._e('left-elbow'), this._e('right-elbow'),
    ], { transformOrigin: '50% 0%' })

    gsap.set(this._e('head'), { transformOrigin: '50% 50%' })
    this._applyTransform()
  }

  _e(name: string) {
    return this.root.querySelector(`#r-${name}`) as SVGElement
  }

  _applyTransform() {
    const sx = this.scale * this.dir
    const sy = this.scale
    this.root.setAttribute('transform', `translate(${this.x}, ${this.y}) scale(${sx}, ${sy})`)
  }

  _build() {
    this.root = mk('g', { id: 'r-root', filter: 'url(#body-shadow)' })
    this.mount.appendChild(this.root)

    // Shadow
    this.root.appendChild(mk('ellipse', {
      id: 'r-shadow', cx: 0, cy: 4,
      rx: P.shadowRx, ry: P.shadowRy, fill: 'rgba(0,0,0,0.28)',
    }))

    // Legs
    for (const side of ['left', 'right']) {
      const sx = side === 'left' ? -P.hipOffX : P.hipOffX
      const hipG = mk('g', { id: `r-${side}-thigh`, transform: `translate(${sx}, ${P.hipY})` })
      hipG.appendChild(rect(-P.thighW / 2, 0, P.thighW, P.thighH, P.thighRx, 'url(#grad-body)'))
      hipG.appendChild(circ(0, P.thighH, 4, COLORS.joint))

      const kneeG = mk('g', { id: `r-${side}-shin`, transform: `translate(0, ${P.thighH})` })
      kneeG.appendChild(rect(-P.shinW / 2, 0, P.shinW, P.shinH, P.shinRx, 'url(#grad-dark)'))
      const footOffX = side === 'left' ? -(P.footW - P.shinW / 2) + 2 : -2
      kneeG.appendChild(rect(footOffX, P.shinH - P.footH + 1, P.footW, P.footH, P.footRx, COLORS.foot))

      hipG.appendChild(kneeG)
      this.root.appendChild(hipG)
    }

    // Upper body
    this.upperBody = mk('g', { id: 'r-upper-body' })
    this.root.appendChild(this.upperBody)

    this.upperBody.appendChild(rect(P.pelvisX, P.pelvisY, P.pelvisW, P.pelvisH, P.pelvisRx, COLORS.dark))
    this.upperBody.appendChild(rect(P.torsoX, P.torsoY, P.torsoW, P.torsoH, P.torsoRx, 'url(#grad-body)'))
    this.upperBody.appendChild(rect(P.torsoX + 6, P.stripe1Y, P.stripeW, P.stripeH, 2, COLORS.chest1))
    this.upperBody.appendChild(rect(P.torsoX + 6, P.stripe2Y, P.stripeW, P.stripeH, 2, COLORS.chest2))

    // Arms
    for (const side of ['left', 'right']) {
      const sx = side === 'left' ? -P.shoulderOffX : P.shoulderOffX
      this.upperBody.appendChild(circ(sx, P.shoulderY, 5.5, COLORS.joint))

      const shoulderG = mk('g', { id: `r-${side}-shoulder`, transform: `translate(${sx}, ${P.shoulderY})` })
      shoulderG.appendChild(rect(-P.uArmW / 2, 0, P.uArmW, P.uArmH, P.uArmRx, 'url(#grad-body)'))
      shoulderG.appendChild(circ(0, P.uArmH, 3.5, COLORS.joint))

      const elbowG = mk('g', { id: `r-${side}-elbow`, transform: `translate(0, ${P.uArmH})` })
      elbowG.appendChild(rect(-P.fArmW / 2, 0, P.fArmW, P.fArmH, P.fArmRx, 'url(#grad-dark)'))
      shoulderG.appendChild(elbowG)
      this.upperBody.appendChild(shoulderG)
    }

    // Head
    const headG = mk('g', { id: 'r-head', transform: `translate(0, ${P.headY})` })
    headG.appendChild(circ(0, 0, P.headR, 'url(#grad-body)'))

    for (const side of ['left', 'right']) {
      const ex = side === 'left' ? -P.eyeOffX : P.eyeOffX
      const eye = circ(ex, P.eyeY, P.eyeR, COLORS.eye)
      eye.setAttribute('id', `r-eye-${side === 'left' ? 'l' : 'r'}`)
      eye.setAttribute('filter', 'url(#glow-cyan)')
      headG.appendChild(eye)
    }

    headG.appendChild(mk('line', {
      x1: 0, y1: -P.headR, x2: 0, y2: -P.headR - P.antLen,
      stroke: COLORS.dark, 'stroke-width': 2, 'stroke-linecap': 'round',
    }))

    const antBall = circ(0, -P.headR - P.antLen - P.antBallR, P.antBallR, COLORS.eye)
    antBall.setAttribute('id', 'r-antenna-ball')
    antBall.setAttribute('filter', 'url(#glow-cyan)')
    headG.appendChild(antBall)

    this.upperBody.appendChild(headG)
  }

  // ── Animation methods ───────────────────────────────────

  reset(duration = 0.25) {
    const tl = gsap.timeline()
    tl.to([
      this._e('left-thigh'), this._e('right-thigh'),
      this._e('left-shin'), this._e('right-shin'),
      this._e('left-shoulder'), this._e('right-shoulder'),
      this._e('left-elbow'), this._e('right-elbow'),
      this._e('head'),
    ], { rotation: 0, duration, ease: 'power2.out' }, 0)
    tl.to(this._e('upper-body'), { x: 0, y: 0, rotation: 0, duration, ease: 'power2.out' }, 0)
    return tl
  }

  idle(rate = 1) {
    const base = 0.85 / rate
    const master = gsap.timeline()
    const body = gsap.timeline({ repeat: -1 })
    const headTl = gsap.timeline({ repeat: -1, delay: 0.15 })
    const antTl = gsap.timeline({ repeat: -1, delay: 0.35 })
    const armTl = gsap.timeline({ repeat: -1, delay: 0.5 })

    body.to(this._e('upper-body'), { y: -3.5, duration: base * 0.85, ease: 'sine.out' })
      .to(this._e('upper-body'), { y: 0.5, duration: base * 1.15, ease: 'sine.in' })
      .to(this._e('upper-body'), { y: 0, duration: base * 0.3, ease: 'sine.out' })

    headTl.to(this._e('head'), { rotation: 5, duration: base * 1.4, ease: 'sine.inOut' })
      .to(this._e('head'), { rotation: -3, duration: base * 1.2, ease: 'sine.inOut' })
      .to(this._e('head'), { rotation: 0, duration: base * 0.6, ease: 'sine.inOut' })

    antTl.to(this._e('antenna-ball'), { y: -2.5, duration: base * 0.6, ease: 'sine.out' })
      .to(this._e('antenna-ball'), { y: 2, duration: base * 0.8, ease: 'sine.inOut' })
      .to(this._e('antenna-ball'), { y: 0, duration: base * 0.5, ease: 'sine.in' })

    const lSh = this._e('left-shoulder')
    const rSh = this._e('right-shoulder')
    armTl.to(lSh, { rotation: 3, duration: base * 1.8, ease: 'sine.inOut' })
      .to(lSh, { rotation: -2, duration: base * 1.5, ease: 'sine.inOut' })
      .to(lSh, { rotation: 0, duration: base * 0.8, ease: 'sine.inOut' })
    armTl.to(rSh, { rotation: -2, duration: base * 1.6, ease: 'sine.inOut' }, 0.3)
      .to(rSh, { rotation: 3, duration: base * 1.4, ease: 'sine.inOut' }, base * 1.6 + 0.3)
      .to(rSh, { rotation: 0, duration: base * 0.9, ease: 'sine.inOut' }, base * 3.0 + 0.3)

    master.add(body, 0).add(headTl, 0).add(antTl, 0).add(armTl, 0)
    return master
  }

  walk(targetX: number, speed = 110) {
    const dist = Math.abs(targetX - this.x)
    if (dist < 1) return gsap.timeline()

    const duration = dist / speed
    const newDir = targetX > this.x ? 1 : -1
    if (newDir !== this.dir) { this.dir = newDir; this._applyTransform() }

    const self = this
    const proxy = { x: this.x }
    const tl = gsap.timeline({ onComplete() { self.x = targetX; self._applyTransform() } })

    tl.to(proxy, {
      x: targetX, duration, ease: 'power1.inOut',
      onUpdate() { self.x = proxy.x; self._applyTransform() },
    }, 0)

    const ub = this._e('upper-body')
    tl.to(ub, { rotation: 4, duration: 0.25, ease: 'power2.out' }, 0)
    tl.to(ub, { rotation: 0, duration: 0.30, ease: 'power2.inOut' }, duration - 0.30)

    const head = this._e('head')
    tl.to(head, { rotation: -6, duration: 0.15, ease: 'power2.out' }, 0.08)
    tl.to(head, { rotation: 0, duration: 0.20, ease: 'power2.inOut' }, duration - 0.25)

    const stepDur = 0.36
    const steps = Math.ceil(duration / stepDur)
    for (let i = 0; i < steps; i++) {
      tl.add(this._stepCycle(i % 2 === 0, stepDur), i * stepDur)
    }

    for (let i = 0; i < steps; i++) {
      const t = i * stepDur
      tl.to(ub, { y: -3, duration: stepDur * 0.5, ease: 'sine.out' }, t)
      tl.to(ub, { y: 1, duration: stepDur * 0.5, ease: 'sine.in' }, t + stepDur * 0.5)
    }
    tl.to(ub, { y: 0, duration: 0.15, ease: 'power2.out' }, duration - 0.15)

    return tl
  }

  _stepCycle(leftFirst = true, dur = 0.36) {
    const lTh = this._e('left-thigh'), rTh = this._e('right-thigh')
    const lSn = this._e('left-shin'), rSn = this._e('right-shin')
    const lSh = this._e('left-shoulder'), rSh = this._e('right-shoulder')
    const lEl = this._e('left-elbow'), rEl = this._e('right-elbow')

    const leadTh = leftFirst ? lTh : rTh, trailTh = leftFirst ? rTh : lTh
    const leadSn = leftFirst ? lSn : rSn, trailSn = leftFirst ? rSn : lSn
    const leadArm = leftFirst ? rSh : lSh, trailArm = leftFirst ? lSh : rSh
    const leadElb = leftFirst ? rEl : lEl, trailElb = leftFirst ? lEl : rEl

    const jitter = () => 0.88 + Math.random() * 0.24
    const half = dur * 0.5
    const thighSwing = 25 * jitter(), trailSwing = 20 * jitter(), armSwing = 18 * jitter()

    const tl = gsap.timeline()

    tl.to(leadTh, { rotation: -thighSwing, duration: half, ease: 'sine.out' }, 0)
    tl.to(leadSn, { rotation: 8, duration: half * 0.6, ease: 'sine.out' }, 0)
    tl.to(leadSn, { rotation: -6, duration: half * 0.4, ease: 'sine.in' }, half * 0.6)
    tl.to(trailTh, { rotation: trailSwing, duration: half, ease: 'sine.out' }, 0)
    tl.to(trailSn, { rotation: -10, duration: half, ease: 'sine.out' }, 0)
    tl.to(leadArm, { rotation: -armSwing, duration: half, ease: 'sine.inOut' }, 0)
    tl.to(trailArm, { rotation: armSwing, duration: half, ease: 'sine.inOut' }, 0)
    tl.to(trailElb, { rotation: 18 * jitter(), duration: half, ease: 'sine.out' }, 0)
    tl.to(leadElb, { rotation: 5, duration: half, ease: 'sine.out' }, 0)

    tl.to(leadTh, { rotation: 4, duration: half, ease: 'sine.in' }, half)
    tl.to(leadSn, { rotation: 12, duration: half * 0.7, ease: 'sine.out' }, half)
    tl.to(leadSn, { rotation: 3, duration: half * 0.3, ease: 'sine.in' }, half + half * 0.7)
    tl.to(trailTh, { rotation: -6, duration: half, ease: 'sine.in' }, half)
    tl.to(trailSn, { rotation: 4, duration: half, ease: 'sine.in' }, half)
    tl.to(leadArm, { rotation: 6, duration: half, ease: 'sine.inOut' }, half)
    tl.to(trailArm, { rotation: -6, duration: half, ease: 'sine.inOut' }, half)
    tl.to(trailElb, { rotation: 3, duration: half, ease: 'sine.inOut' }, half)
    tl.to(leadElb, { rotation: 2, duration: half, ease: 'sine.inOut' }, half)

    return tl
  }

  wave(repeats = 2, side: 'right' | 'left' = 'right') {
    const arm = this._e(`${side}-shoulder`)
    const elbow = this._e(`${side}-elbow`)
    const head = this._e('head')
    const ub = this._e('upper-body')
    const tl = gsap.timeline()

    tl.to(arm, { rotation: 8, duration: 0.08, ease: 'power2.in' })
    tl.to(arm, { rotation: -125, duration: 0.28, ease: 'back.out(1.4)' })
    tl.to(ub, { x: side === 'right' ? -2 : 2, duration: 0.2, ease: 'sine.out' }, '<0.05')
    tl.to(head, { rotation: side === 'right' ? 8 : -8, duration: 0.22, ease: 'power2.out' }, '<')

    for (let i = 0; i < repeats; i++) {
      const speed = 0.12 + i * 0.02
      tl.to(elbow, { rotation: 40, duration: speed, ease: 'sine.out' })
        .to(elbow, { rotation: -15, duration: speed, ease: 'sine.in' })
    }

    tl.to(elbow, { rotation: 0, duration: 0.18, ease: 'power2.inOut' })
    tl.to(arm, { rotation: 5, duration: 0.25, ease: 'power2.in' }, '<0.05')
    tl.to(arm, { rotation: 0, duration: 0.15, ease: 'sine.out' })
    tl.to(ub, { x: 0, duration: 0.2, ease: 'sine.inOut' }, '<')
    tl.to(head, { rotation: 0, duration: 0.2, ease: 'sine.inOut' }, '<')

    return tl
  }

  /** Track the active timeline so destroy() can kill it. */
  trackTimeline(tl: gsap.core.Timeline) {
    this._activeTl?.kill()
    this._activeTl = tl
    return tl
  }

  destroy() {
    this.destroyed = true
    this._activeTl?.kill()
    this._activeTl = null
    gsap.killTweensOf(this.root)
    gsap.killTweensOf(this.root.querySelectorAll('*'))
    this.root.remove()
  }
}

/** SVG <defs> required by the Robot class */
export const ROBOT_DEFS = `
  <linearGradient id="grad-body" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#9abdd8"/>
    <stop offset="100%" stop-color="#6a8caf"/>
  </linearGradient>
  <linearGradient id="grad-dark" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#3a6080"/>
    <stop offset="100%" stop-color="#2e4a68"/>
  </linearGradient>
  <filter id="glow-cyan" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur stdDeviation="1.5" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="body-shadow" x="-10%" y="-10%" width="120%" height="130%">
    <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.3)"/>
  </filter>
`
