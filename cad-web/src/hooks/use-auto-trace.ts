import { useMutation } from '@tanstack/react-query'
import { interactionService } from '@/services/interaction-service'
import { useCanvasStore } from '@/stores/canvas-store'
import { toast } from 'sonner'

export function useAutoTrace() {
  const { setTraceResult } = useCanvasStore()

  const mutation = useMutation({
    mutationFn: async (edgeId: number) => {
      return interactionService.autoTrace({ edge_id: edgeId })
    },

    onSuccess: (data) => {
      if (data.success) {
        setTraceResult(data.loop_points)
        toast.success('追踪成功')
      } else {
        toast.warning(data.message || '未找到闭合环')
        setTraceResult(null)
      }
    },

    onError: (error: Error) => {
      toast.error(error.message || '追踪失败')
    },
  })

  return {
    autoTrace: mutation.mutateAsync,
    isTracing: mutation.isPending,
  }
}
