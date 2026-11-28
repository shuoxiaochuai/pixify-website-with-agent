import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { isTerminal } from '@/api/runs'
import {
  sessionsApi,
  type SessionCreateInput,
  type SessionDetail,
  type SessionPatchInput,
} from '@/api/sessions'
import { errorMessage } from '@/hooks/useAuth'
import { useRun } from '@/hooks/useRun'
import { offersUndo, toolLabel } from '@/lib/tools'
import { toast } from '@/stores/toasts'

const LIST_KEY = ['sessions']
const detailKey = (id: string) => ['session', id]
const historyKey = (id: string) => ['session', id, 'history']

export function useSessions() {
  return useQuery({ queryKey: LIST_KEY, queryFn: sessionsApi.list })
}

export function useSession(id: string | null) {
  return useQuery({
    queryKey: detailKey(id ?? ''),
    queryFn: () => sessionsApi.get(id as string),
    enabled: Boolean(id),
  })
}

export function useSessionHistory(id: string | null) {
  return useQuery({
    queryKey: historyKey(id ?? ''),
    queryFn: () => sessionsApi.history(id as string),
    enabled: Boolean(id),
  })
}

function useCacheSession(id: string) {
  const queryClient = useQueryClient()
  return (detail: SessionDetail) => {
    queryClient.setQueryData(detailKey(id), detail)
    void queryClient.invalidateQueries({ queryKey: LIST_KEY })
    void queryClient.invalidateQueries({ queryKey: historyKey(id) })
  }
}

export function useCreateSession() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (input: SessionCreateInput) => sessionsApi.create(input),
    onSuccess: (detail) => {
      queryClient.setQueryData(detailKey(detail.id), detail)
      void queryClient.invalidateQueries({ queryKey: LIST_KEY })
    },
  })
}

export function usePatchSession(id: string) {
  const cache = useCacheSession(id)
  return useMutation({
    mutationFn: (input: SessionPatchInput) => sessionsApi.patch(id, input),
    onSuccess: cache,
  })
}

/**
 * 历史列表里的重命名。只改标题，所以不必重取历史记录与文档，
 * 详情缓存就地换标题，列表整体重取一次。
 */
export function useRenameSession() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) =>
      sessionsApi.patch(id, { title }),
    onSuccess: (detail) => {
      queryClient.setQueryData<SessionDetail>(detailKey(detail.id), (current) =>
        current ? { ...current, title: detail.title } : detail,
      )
      void queryClient.invalidateQueries({ queryKey: LIST_KEY })
      toast('已重命名', 'ok')
    },
    onError: (error) => toast(errorMessage(error), 'danger'),
  })
}

/**
 * 删掉整条会话。后端是硬删除，没有撤销入口，所以调用方要先拿到确认。
 */
export function useDeleteSession() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => sessionsApi.remove(id),
    onSuccess: (_detail, id) => {
      // ['session', id] 是前缀，详情、消息流、编辑记录的缓存一起清掉，
      // 否则切回同一个地址会先闪出已经不存在的内容
      queryClient.removeQueries({ queryKey: detailKey(id) })
      void queryClient.invalidateQueries({ queryKey: LIST_KEY })
      toast('对话已删除', 'ok')
    },
    onError: (error) => toast(errorMessage(error), 'danger'),
  })
}

/**
 * 从图片墙摘掉一张。只删会话与图的关联行，素材本体还在素材库，
 * 画布上正在用的那张会被后端拒掉（409），拒绝理由直接由 toast 播报。
 */
export function useDetachWallAsset(id: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (assetId: string) => sessionsApi.detach(id, assetId),
    onSuccess: () => {
      // 只重取详情：墙就存在 session.assets 里，不去打扰历史记录和消息流
      void queryClient.invalidateQueries({ queryKey: detailKey(id), exact: true })
      toast('已从图片墙移除', 'ok')
    },
    onError: (error) => toast(errorMessage(error), 'danger'),
  })
}

export function useExportPack(id: string) {
  return useMutation({
    mutationFn: (assetIds: string[]) => sessionsApi.export(id, assetIds),
    onError: (error) => toast(errorMessage(error), 'danger'),
  })
}

export function useSessionTools(id: string) {
  const queryClient = useQueryClient()
  const cache = useCacheSession(id)
  // 记下自己发起的任务（run id + 工具），完成后供界面回填结果
  const [watched, setWatched] = useState<{ id: string; tool: string } | null>(null)
  const live = useRun(watched?.id ?? null)

  const invoke = useMutation({
    mutationFn: ({ tool, params }: { tool: string; params?: Record<string, unknown> }) =>
      sessionsApi.invoke(id, tool, params),
    onSuccess: (body) => {
      cache(body.session)
      if (!isTerminal(body.run.status)) {
        setWatched({ id: body.run.id, tool: body.run.tool })
      }
    },
    onError: (error) => toast(errorMessage(error), 'danger'),
  })

  const historyLock = useRef(false)
  const unlockHistory = () => {
    historyLock.current = false
  }

  const undo = useMutation({
    mutationFn: () => sessionsApi.undo(id),
    onSuccess: cache,
    onError: (error) => toast(errorMessage(error), 'danger'),
    onSettled: unlockHistory,
  })
  const redo = useMutation({
    mutationFn: () => sessionsApi.redo(id),
    onSuccess: cache,
    onError: (error) => toast(errorMessage(error), 'danger'),
    onSettled: unlockHistory,
  })

  const waiting = Boolean(watched && (!live.status || !isTerminal(live.status)))
  const busy = invoke.isPending || undo.isPending || redo.isPending || waiting

  // 提示里的撤销要在完成那一刻才可用，用 ref 拿最新的 mutation，避免把它塞进 effect 依赖
  const undoRef = useRef(() => undo.mutate())
  undoRef.current = () => undo.mutate()
  const pendingTool = watched?.tool
  // 一个任务只播报一次，后续依赖变化不会再弹提示
  const announced = useRef<string | null>(null)

  useEffect(() => {
    if (!watched || !live.status || !isTerminal(live.status)) return
    const runId = watched.id
    if (announced.current === runId) return
    announced.current = runId
    void queryClient.invalidateQueries({ queryKey: detailKey(id) })
    void queryClient.invalidateQueries({ queryKey: historyKey(id) })
    if (live.status === 'failed') {
      toast(live.error || '处理失败', 'danger')
      return
    }
    const label = toolLabel(pendingTool)
    toast(
      label ? `${label}完成` : live.stage || '已完成',
      'ok',
      offersUndo(pendingTool)
        ? { action: { label: '撤销', run: () => undoRef.current() } }
        : undefined,
    )
  }, [watched, live.status, live.error, live.stage, pendingTool, id, queryClient])

  /**
   * 当前在跑的是不是这一次调用。工具层同时只跑一个任务，界面据此只给发起的那个按钮
   * 换进度文案，其余按钮虽然一样禁用，但不会看起来像是也在生成。
   */
  const isRunning = (tool: string, match?: (params: Record<string, unknown>) => boolean) => {
    const pending = invoke.isPending || waiting ? invoke.variables : undefined
    if (pending?.tool !== tool) return false
    return match ? match(pending.params ?? {}) : true
  }

  const runHistory = (action: () => void) => {
    if (historyLock.current || busy) return
    historyLock.current = true
    action()
  }

  // 最近一次排队任务的最终状态与结果，病灶面板据此回填自动算出的阈值
  const lastRun =
    watched && live.status
      ? { id: watched.id, tool: watched.tool, status: live.status, result: live.result }
      : null

  return {
    invoke: (tool: string, params?: Record<string, unknown>) => invoke.mutate({ tool, params }),
    undo: () => runHistory(() => undo.mutate()),
    redo: () => runHistory(() => redo.mutate()),
    busy,
    isRunning,
    lastRun,
    pending: waiting,
    pendingStage: waiting ? live.stage : '',
    pendingProgress: waiting ? live.progress : 0,
  }
}

export type SessionTools = ReturnType<typeof useSessionTools>
