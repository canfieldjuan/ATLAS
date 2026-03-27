"use client";

import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { Robot, ROBOT_DEFS } from '@/lib/robot'

const GROUND_Y = 362
const ROBOT_SCALE = 0.92
const ROBOT_START = -80

const LETTERS = { A1: 192, T: 316, L: 430, A2: 550, S: 672 }
const JUMP_POINTS = [
  { x: LETTERS.A1, height: 40, type: 0 },
  { x: LETTERS.T, height: 48, type: 0 },
  { x: LETTERS.L, height: 36, type: 0 },
  { x: LETTERS.A2, height: 44, type: 2 },
  { x: LETTERS.S, height: 52, type: 1 },
]

function idlePause(robot: Robot, duration: number) {
  const tl = gsap.timeline()
  let idleTl: gsap.core.Timeline | undefined
  tl.add(() => { idleTl = robot.idle() })
  tl.to({}, { duration })
  tl.add(() => { if (idleTl) idleTl.kill() })
  tl.add(robot.reset(0.25))
  return tl
}

function buildCrossing(robot: Robot) {
  const tl = gsap.timeline()
  JUMP_POINTS.forEach(({ x, height, type }) => {
    tl.add(robot.walk(x, 105))
    tl.add(robot.jump(height, type))
  })
  tl.add(robot.walk(LETTERS.S + 130, 105))
  return tl
}

function runChoreography(robot: Robot) {
  if (robot.destroyed) return
  const tl = gsap.timeline({ onComplete: () => runChoreography(robot) })
  robot.trackTimeline(tl)

  tl.add(robot.walk(LETTERS.A1 - 60, 130))
  tl.add(robot.look('right'))
  tl.add(robot.think(0.8))
  tl.add(buildCrossing(robot))
  tl.add(robot.look('reset'))
  tl.add(robot.turn(-1))
  tl.add(robot.walk(450, 100))
  tl.add(robot.dance())
  tl.add(robot.dance())
  tl.add(robot.victory())
  tl.add(robot.sit(1.0))
  tl.add(robot.turn(1))
  tl.add(robot.wave(3))
  tl.add(robot.reset())
  tl.add(idlePause(robot, 1.6))
  tl.add(robot.turn(-1))
  tl.add(robot.walk(ROBOT_START, 160))
}

interface AtlasHeroSceneProps {
  title?: string
  tagline?: string
}

/**
 * Full animated Atlas hero scene -- robot walks across the ATLAS logo,
 * jumps over letters, dances, waves, and loops. Direct port of
 * animated-robot-logo with React lifecycle management.
 */
export default function AtlasHeroScene({
  title = 'ATLAS',
  tagline = 'INTELLIGENCE PLATFORM',
}: AtlasHeroSceneProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const robotRef = useRef<Robot | null>(null)

  useEffect(() => {
    const svg = svgRef.current
    if (!svg || robotRef.current) return

    const mountG = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    mountG.setAttribute('id', 'robot-mount')
    svg.appendChild(mountG)

    const robot = new Robot(mountG, {
      x: ROBOT_START,
      y: GROUND_Y,
      scale: ROBOT_SCALE,
    })
    robotRef.current = robot

    // Star twinkle
    const stars = svg.querySelectorAll('[data-stars] circle')
    stars.forEach((star, i) => {
      gsap.to(star, {
        opacity: Math.random() * 0.4 + 0.1,
        duration: Math.random() * 2 + 1.5,
        repeat: -1,
        yoyo: true,
        delay: i * 0.15,
        ease: 'sine.inOut',
      })
    })

    runChoreography(robot)

    return () => {
      robot.destroy()
      robotRef.current = null
    }
  }, [])

  return (
    <div className="w-full max-w-[900px] mx-auto">
      <svg
        ref={svgRef}
        viewBox="0 0 900 440"
        className="w-full h-auto"
        aria-label="Animated Atlas robot scene"
      >
        <defs dangerouslySetInnerHTML={{ __html: ROBOT_DEFS + `
          <filter id="logo-glow" x="-10%" y="-40%" width="120%" height="180%">
            <feGaussianBlur stdDeviation="7" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <linearGradient id="grad-logo" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#d8ecff"/>
            <stop offset="55%" stop-color="#7aadd4"/>
            <stop offset="100%" stop-color="#2a4a70"/>
          </linearGradient>
        `}} />

        {/* Background stars */}
        <g data-stars="" opacity="0.45">
          <circle cx="42" cy="28" r="1" fill="white"/>
          <circle cx="127" cy="55" r="1.5" fill="white"/>
          <circle cx="210" cy="18" r="1" fill="white"/>
          <circle cx="320" cy="42" r="0.8" fill="white"/>
          <circle cx="480" cy="12" r="1.2" fill="white"/>
          <circle cx="630" cy="35" r="1" fill="white"/>
          <circle cx="755" cy="20" r="1.5" fill="white"/>
          <circle cx="858" cy="50" r="1" fill="white"/>
          <circle cx="70" cy="80" r="0.8" fill="white"/>
          <circle cx="180" cy="95" r="1" fill="white"/>
          <circle cx="540" cy="70" r="1.2" fill="white"/>
          <circle cx="780" cy="88" r="0.8" fill="white"/>
          <circle cx="880" cy="110" r="1" fill="white"/>
          <circle cx="25" cy="140" r="1.5" fill="white"/>
          <circle cx="840" cy="155" r="0.8" fill="white"/>
          <circle cx="350" cy="30" r="2" fill="white" opacity="0.9"/>
          <circle cx="690" cy="58" r="2" fill="white" opacity="0.9"/>
          <circle cx="150" cy="130" r="1.8" fill="white" opacity="0.7"/>
        </g>

        {/* Floor line */}
        <line x1="80" y1="388" x2="820" y2="388"
              stroke="rgba(80,140,210,0.12)" strokeWidth="1"/>

        {/* Logo text */}
        <text
          x="450" y="383"
          textAnchor="middle"
          fontFamily="Impact, 'Arial Narrow', Arial, sans-serif"
          fontSize="168"
          fontWeight="900"
          letterSpacing="12"
          fill="url(#grad-logo)"
          filter="url(#logo-glow)"
        >{title}</text>

        {/* Tagline */}
        <text
          x="450" y="418"
          textAnchor="middle"
          fontFamily="'Courier New', Courier, monospace"
          fontSize="13"
          letterSpacing="6"
          fill="rgba(80,140,210,0.55)"
        >{tagline}</text>
      </svg>
    </div>
  )
}
