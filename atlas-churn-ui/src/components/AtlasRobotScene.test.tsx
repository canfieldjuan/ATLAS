import { cleanup, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import AtlasRobotScene from './AtlasRobotScene'

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

vi.mock('../lib/robot', () => ({
  Robot: robotModule.Robot,
  ROBOT_DEFS: robotModule.ROBOT_DEFS,
}))

describe('AtlasRobotScene', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    robotModule.instances.length = 0
  })

  it('renders the animated robot shell and starts the entry choreography', () => {
    render(<AtlasRobotScene />)

    expect(screen.getByLabelText('Animated Atlas robot')).toBeInTheDocument()
    expect(robotModule.Robot).toHaveBeenCalledWith(
      expect.any(SVGGElement),
      expect.objectContaining({ x: -60, y: 170, scale: 0.9, flip: false }),
    )

    const instance = robotModule.instances[0] as {
      walk: ReturnType<typeof vi.fn>
      trackTimeline: ReturnType<typeof vi.fn>
    }
    expect(instance.walk).toHaveBeenCalledWith(120, 100)
    expect(instance.trackTimeline).toHaveBeenCalled()
  })

  it('destroys the robot on unmount', () => {
    const { unmount } = render(<AtlasRobotScene />)

    const instance = robotModule.instances[0] as { destroy: ReturnType<typeof vi.fn> }
    unmount()

    expect(instance.destroy).toHaveBeenCalledTimes(1)
  })
})
