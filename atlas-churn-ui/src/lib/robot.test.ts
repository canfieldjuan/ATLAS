import { cleanup } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const gsapMock = vi.hoisted(() => {
  const timelines: any[] = []
  return {
    set: vi.fn(),
    killTweensOf: vi.fn(),
    timeline: vi.fn(() => {
      const tl: any = {
        to: vi.fn(),
        add: vi.fn(),
        duration: vi.fn(),
        kill: vi.fn(),
      }
      tl.to.mockReturnValue(tl)
      tl.add.mockReturnValue(tl)
      tl.duration.mockReturnValue(0)
      timelines.push(tl)
      return tl
    }),
    __timelines: timelines,
  }
})

vi.mock('gsap', () => ({
  default: gsapMock,
}))

import { ROBOT_DEFS, Robot } from './robot'

describe('robot', () => {
  beforeEach(() => {
    cleanup()
    document.body.innerHTML = ''
    vi.clearAllMocks()
    gsapMock.__timelines.length = 0
  })

  it('builds the SVG robot with the expected transform and defs', () => {
    const mount = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    document.body.appendChild(mount)

    const robot = new Robot(mount, { x: 10, y: 20, scale: 2, flip: true })

    expect(robot.root.getAttribute('transform')).toBe('translate(10, 20) scale(-2, 2)')
    expect(robot.root.querySelector('#r-head')).not.toBeNull()
    expect(robot.root.querySelector('#r-left-thigh')).not.toBeNull()
    expect(robot.root.querySelector('#r-right-shoulder')).not.toBeNull()
    expect(gsapMock.set).toHaveBeenCalledTimes(2)
    expect(ROBOT_DEFS).toContain('id="grad-body"')
    expect(ROBOT_DEFS).toContain('id="body-shadow"')
  })

  it('tracks and kills the active timeline on destroy', () => {
    const mount = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    document.body.appendChild(mount)

    const robot = new Robot(mount)
    const priorTimeline = {
      kill: vi.fn(),
    }

    robot.trackTimeline(priorTimeline as any)
    robot.destroy()

    expect(robot.destroyed).toBe(true)
    expect(priorTimeline.kill).toHaveBeenCalledTimes(1)
    expect(gsapMock.killTweensOf).toHaveBeenCalledTimes(2)
    expect(mount.querySelector('#r-root')).toBeNull()
  })
})
