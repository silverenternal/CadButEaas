import { describe, it, expect, vi } from 'vitest'
import { ApiError, NetworkError, TimeoutError, ValidationError } from '@/services/api-client'

vi.mock('@/services/api-client', async () => {
  const actual = await vi.importActual('@/services/api-client')
  return {
    ...actual,
    default: {
      getBaseUrl: () => 'http://test-api.com',
    },
  }
})

describe('API Client', () => {
  describe('ApiError', () => {
    it('should create ApiError with correct properties', () => {
      const errorData = {
        request_id: 'req-123',
        status: 'FAILURE' as const,
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Invalid input',
          details: { field: 'name' },
          retryable: false,
          suggestion: 'Check your input',
        },
        latency_ms: 100,
      }

      const error = new ApiError(errorData)

      expect(error.name).toBe('ApiError')
      expect(error.message).toBe('Invalid input')
      expect(error.code).toBe('VALIDATION_ERROR')
      expect(error.retryable).toBe(false)
      expect(error.suggestion).toBe('Check your input')
    })

    it('should default retryable to false', () => {
      const errorData = {
        request_id: 'req-123',
        status: 'FAILURE' as const,
        error: {
          code: 'UNKNOWN_ERROR',
          message: 'Something went wrong',
        },
        latency_ms: 100,
      }

      const error = new ApiError(errorData)
      expect(error.retryable).toBe(false)
    })
  })

  describe('NetworkError', () => {
    it('should create NetworkError with message', () => {
      const error = new NetworkError('Connection failed')
      expect(error.name).toBe('NetworkError')
      expect(error.message).toBe('Connection failed')
    })
  })

  describe('TimeoutError', () => {
    it('should create TimeoutError with default message', () => {
      const error = new TimeoutError()
      expect(error.name).toBe('TimeoutError')
      expect(error.message).toBe('请求超时')
    })
  })

  describe('ValidationError', () => {
    it('should create ValidationError with field', () => {
      const error = new ValidationError('Invalid email', 'email')
      expect(error.name).toBe('ValidationError')
      expect(error.message).toBe('Invalid email')
      expect(error.field).toBe('email')
    })

    it('should create ValidationError without field', () => {
      const error = new ValidationError('Invalid input')
      expect(error.name).toBe('ValidationError')
      expect(error.message).toBe('Invalid input')
      expect(error.field).toBeUndefined()
    })
  })
})
