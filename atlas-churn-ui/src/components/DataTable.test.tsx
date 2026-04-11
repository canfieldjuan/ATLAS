import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import DataTable, { type Column } from './DataTable'

type Row = {
  id: string
  name: string
  score: number
}

const columns: Column<Row>[] = [
  {
    key: 'name',
    header: 'Name',
    render: (row) => row.name,
    sortable: true,
    sortValue: (row) => row.name,
  },
  {
    key: 'score',
    header: 'Score',
    render: (row) => String(row.score),
    sortable: true,
    sortValue: (row) => row.score,
  },
]

describe('DataTable', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders the empty state and fires the empty action', async () => {
    const user = userEvent.setup()
    const onClick = vi.fn()

    render(
      <DataTable
        columns={columns}
        data={[]}
        emptyMessage="No rows available"
        emptyAction={{ label: 'Add row', onClick }}
      />,
    )

    expect(screen.getByText('No rows available')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Add row' }))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('sorts rows and exposes aria-sort on sortable headers', async () => {
    const user = userEvent.setup()

    render(
      <DataTable
        columns={columns}
        data={[
          { id: '2', name: 'Bravo', score: 20 },
          { id: '1', name: 'Alpha', score: 10 },
        ]}
      />,
    )

    const scoreHeader = screen.getByText('Score').closest('th')
    const nameHeader = screen.getByText('Name').closest('th')

    expect(scoreHeader).toHaveAttribute('aria-sort', 'none')
    expect(nameHeader).toHaveAttribute('aria-sort', 'none')

    await user.click(screen.getByText('Name'))
    expect(nameHeader).toHaveAttribute('aria-sort', 'descending')
    expect(screen.getAllByRole('row')[1]).toHaveTextContent('Bravo')

    await user.click(screen.getByText('Name'))
    expect(nameHeader).toHaveAttribute('aria-sort', 'ascending')
    expect(screen.getAllByRole('row')[1]).toHaveTextContent('Alpha')
  })

  it('paginates rows, uses a single range separator, and forwards row clicks', async () => {
    const user = userEvent.setup()
    const onRowClick = vi.fn()
    const rows = Array.from({ length: 30 }, (_, index) => ({
      id: String(index + 1),
      name: `Vendor ${index + 1}`,
      score: index + 1,
    }))

    render(
      <DataTable
        columns={columns}
        data={rows}
        onRowClick={onRowClick}
      />,
    )

    expect(screen.getByText('1-25 of 30')).toBeInTheDocument()
    await user.click(screen.getByText('Vendor 1'))
    expect(onRowClick).toHaveBeenCalledWith(rows[0])

    await user.click(screen.getByRole('button', { name: 'Next page' }))
    expect(screen.getByText('26-30 of 30')).toBeInTheDocument()
    expect(screen.getByText('Vendor 26')).toBeInTheDocument()
  })
})
