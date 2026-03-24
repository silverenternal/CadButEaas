import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { CadViewer } from '@/components/cad-viewer'

// Mock @mlightcad/cad-simple-viewer
vi.mock('@mlightcad/cad-simple-viewer', () => {
  return {
    AcTrView2d: vi.fn().mockImplementation(() => ({
      isDirty: false,
    })),
    AcApDocument: vi.fn().mockImplementation(() => ({
      openUri: vi.fn().mockResolvedValue(true),
      openDocument: vi.fn().mockResolvedValue(true),
      database: {},
      uri: undefined,
      docTitle: 'Untitled',
    })),
    AcEdOpenMode: {
      Read: 0,
      Write: 1,
    },
  }
})

describe('CadViewer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  const TEST_ID = 'cad-viewer-container'

  it('should render container', async () => {
    render(<CadViewer file={null} data-testid={TEST_ID} />)
    const container = await screen.findByTestId(TEST_ID)
    expect(container).toBeInTheDocument()
  })

  it('should apply custom className', async () => {
    render(<CadViewer file={null} className="custom-class" data-testid={TEST_ID} />)
    const container = await screen.findByTestId(TEST_ID)
    expect(container).toHaveClass('custom-class')
  })

  it('should use default options when no options provided', async () => {
    render(<CadViewer file={null} data-testid={TEST_ID} />)
    const container = await screen.findByTestId(TEST_ID)
    expect(container).toBeInTheDocument()
  })
})
