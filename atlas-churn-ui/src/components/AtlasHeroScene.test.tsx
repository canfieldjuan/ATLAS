import { cleanup, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import AtlasHeroScene from './AtlasHeroScene'

const gsapMock = vi.hoisted(() => ({
  timeline: vi.fn(() => {
    const timeline = {
      add: vi.fn(() => timeline),
      to: vi.fn(() => timeline),
      kill: vi.fn(),
    }
    return timeline
  }),
  to: vi.fn(),
}))

const robotModule = vi.hoisted(() => {
  const instances: Array<Record<string, unknown>> = []

  const makeTimeline = () => {
    const timeline = {
      add: vi.fn(() => timeline),
      to: vi.fn(() => timeline),
      kill: vi.fn(),
      eventCallback: vi.fn(() => timeline),
    }
    return timeline
  }

  const Robot = vi.fn().mockImplementation(() => {
    const instance = {
      destroyed: false,
      walk: vi.fn(() => makeTimeline()),
      look: vi.fn(() => makeTimeline()),
      think: vi.fn(() => makeTimeline()),
      jump: vi.fn(() => makeTimeline()),
      turn: vi.fn(() => makeTimeline()),
      dance: vi.fn(() => makeTimeline()),
      victory: vi.fn(() => makeTimeline()),
      sit: vi.fn(() => makeTimeline()),
      wave: vi.fn(() => makeTimeline()),
      reset: vi.fn(() => makeTimeline()),
      idle: vi.fn(() => makeTimeline()),
      trackTimeline: vi.fn(),
      destroy: vi.fn(),
    }
    instances.push(instance)
    return instance
  })

  return {
    Robot,
    ROBOT_DEFS: '<g id="robot-defs"></g>',
    instances,
  }
})

vi.mock('gsap', () => ({
  default: gsapMock,
}))

vi.mock('../lib/robot', () => ({
  Robot: robotModule.Robot,
  ROBOT_DEFS: robotModule.ROBOT_DEFS,
}))

describe('AtlasHeroScene', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    robotModule.instances.length = 0
  })

  it('renders the configured title and tagline and starts the robot choreography', () => {
    render(<AtlasHeroScene title="CHURN" tagline="RETENTION PLATFORM" />)

    expect(screen.getByLabelText('Animated Atlas robot scene')).toBeInTheDocument()
    expect(screen.getByText('CHURN')).toBeInTheDocument()
    expect(screen.getByText('RETENTION PLATFORM')).toBeInTheDocument()
    expect(robotModule.Robot).toHaveBeenCalledTimes(1)
    expect(robotModule.Robot).toHaveBeenCalledWith(
      expect.any(SVGGElement),
      expect.objectContaining({ x: -80, y: 362, scale: 0.92 }),
    )
    expect(gsapMock.to).toHaveBeenCalledTimes(18)
  })

  it('destroys the robot instance on unmount', () => {
    const { unmount } = render(<AtlasHeroScene />)

    const instance = robotModule.instances[0] as { destroy: ReturnType<typeof vi.fn> }
    unmount()

    expect(instance.destroy).toHaveBeenCalledTimes(1)
  })
})
