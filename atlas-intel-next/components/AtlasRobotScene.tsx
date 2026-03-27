"use client";

import { useEffect, useRef } from 'react'
import { Robot, ROBOT_DEFS } from '@/lib/robot'

/**
 * Animated Atlas robot for the blog hero section.
 * Choreography: walk in from left, wave, settle into idle loop.
 */
export default function AtlasRobotScene() {
  const svgRef = useRef<SVGSVGElement>(null)
  const robotRef = useRef<Robot | null>(null)

  useEffect(() => {
    const svg = svgRef.current
    if (!svg || robotRef.current) return

    const mountG = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    svg.appendChild(mountG)

    const robot = new Robot(mountG, { x: -60, y: 170, scale: 0.9, flip: false })
    robotRef.current = robot

    // Choreography: walk in -> wave -> idle (guarded against unmount)
    const walkTl = robot.walk(120, 100)
    robot.trackTimeline(walkTl)
    walkTl.eventCallback('onComplete', () => {
      if (robot.destroyed) return
      const waveTl = robot.wave(2)
      robot.trackTimeline(waveTl)
      waveTl.eventCallback('onComplete', () => {
        if (robot.destroyed) return
        const resetTl = robot.reset(0.3)
        robot.trackTimeline(resetTl)
        resetTl.eventCallback('onComplete', () => {
          if (robot.destroyed) return
          robot.trackTimeline(robot.idle(0.8))
        })
      })
    })

    return () => {
      robot.destroy()
      robotRef.current = null
    }
  }, [])

  return (
    <svg
      ref={svgRef}
      viewBox="0 0 240 200"
      className="w-40 h-32 sm:w-52 sm:h-40 mx-auto"
      aria-label="Animated Atlas robot"
    >
      <defs dangerouslySetInnerHTML={{ __html: ROBOT_DEFS }} />
    </svg>
  )
}
