import { QueryClient } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // 重试策略
      retry: (failureCount, error) => {
        // 不重试 4xx 错误
        if (error instanceof Error && error.message.includes('4')) {
          return false
        }
        return failureCount < 3
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),

      // 缓存策略
      staleTime: 5 * 60 * 1000, // 5 分钟
      gcTime: 10 * 60 * 1000, // 10 分钟后 GC
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,

      // 超时
      networkMode: 'online',
    },
    mutations: {
      retry: 1,
      networkMode: 'always',
    },
  },
})
