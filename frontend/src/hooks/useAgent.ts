import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { agentApi, type Turn, type TurnEvent, type TurnKind } from '@/api/agent'
import { isTerminal } from '@/api/runs'

const turnsKey = (sessionId: string) => ['session', sessionId, 'messages']

function useRefreshTurn(sessionId: string) {
  const queryClient = useQueryClient()
  return () => {
    void queryClient.invalidateQueries({ queryKey: turnsKey(sessionId) })
    void queryClient.invalidateQueries({ queryKey: ['session', sessionId] })
    void queryClient.invalidateQueries({ queryKey: ['session', sessionId, 'history'] })
  }
}

export function useTurns(sessionId: string) {
  const queryClient = useQueryClient()
  const query = useQuery({
    queryKey: turnsKey(sessionId),
    queryFn: () => agentApi.turns(sessionId),
    // 第一步 SSE 结束时下一步可能还没写进计划，整轮 running 时接着拉，避免卡片停在「等待中」
    refetchInterval: (current) =>
      current.state.data?.at(-1)?.status === 'running' ? 800 : false,
  })

  const latest = query.data?.at(-1)?.status
  const previous = useRef(latest)
  useEffect(() => {
    const wasRunning = previous.current === 'running'
    previous.current = latest
    if (!wasRunning || !latest || !isTerminal(latest)) return
    void queryClient.invalidateQueries({ queryKey: ['session', sessionId] })
    void queryClient.invalidateQueries({ queryKey: ['session', sessionId, 'history'] })
  }, [latest, sessionId, queryClient])

  useEffect(() => {
    previous.current = undefined
  }, [sessionId])

  return query
}

export function useSendMessage(sessionId: string) {
  const queryClient = useQueryClient()
  const refresh = useRefreshTurn(sessionId)
  return useMutation({
    mutationFn: (text: string) => agentApi.send(sessionId, text),
    onSuccess: (turn) => {
      // POST 只回一条进行中的空轮次，不等 refetch 就把用户那句话垫进列表
      queryClient.setQueryData<Turn[]>(turnsKey(sessionId), (current) =>
        current && !current.some((item) => item.id === turn.id) ? [...current, turn] : current,
      )
      refresh()
    },
  })
}

// 一轮指令跑到哪了：阶段文案、已报出的步骤名、正文增量、以及分流结果
export type TurnLive = {
  stages: string[]
  steps: (string | undefined)[]
  text: string
  kind: TurnKind | null
}

const IDLE: TurnLive = { stages: [], steps: [], text: '', kind: null }

function fold(current: TurnLive, event: TurnEvent): TurnLive {
  switch (event.type) {
    case 'stage':
      // 重排等环节可能重复同一句话，连着两条相同的只占一行
      return current.stages.at(-1) === event.text
        ? current
        : { ...current, stages: [...current.stages, event.text] }
    case 'kind':
      return { ...current, kind: event.value }
    case 'plan': {
      const steps = [...current.steps]
      steps[event.index - 1] = event.label
      return { ...current, steps }
    }
    case 'token':
      return { ...current, text: current.text + event.text }
    default:
      // snapshot 与 done 之后的事交给落库结果，这里不拿它覆盖已经流出来的正文
      return current
  }
}

/**
 * 订阅一轮指令的过程事件。
 *
 * 后端跑完才一次落库，中间这几秒画布上什么都看不到；这里把阶段、步骤名与正文增量接回来。
 * 传 null 表示这一轮不需要看过程（已结束或不是最新一轮），不建连接。
 */
export function useTurnStream(turnId: string | null, sessionId: string) {
  const queryClient = useQueryClient()
  const [frame, setFrame] = useState<{ id: string; live: TurnLive } | null>(null)

  useEffect(() => {
    if (!turnId) return

    const source = new EventSource(`/events/turns/${turnId}`)
    source.onmessage = (event) => {
      const payload = JSON.parse(event.data) as TurnEvent
      const over =
        payload.type === 'done' ||
        payload.type === 'failed' ||
        (payload.type === 'snapshot' && isTerminal(payload.turn.status))
      if (!over) {
        // 挂在这一轮上的残留帧不能泄到下一轮去
        setFrame((current) => ({
          id: turnId,
          live: fold(current?.id === turnId ? current.live : IDLE, payload),
        }))
        return
      }
      // 过程事件只是提前展示，以落库的那一轮为准；不关掉的话服务端断流后会反复重连
      source.close()
      void queryClient.invalidateQueries({ queryKey: turnsKey(sessionId) })
      void queryClient.invalidateQueries({ queryKey: ['session', sessionId] })
    }

    return () => source.close()
  }, [turnId, sessionId, queryClient])

  return frame?.id === turnId ? frame.live : IDLE
}

export function usePlanActions(sessionId: string) {
  const refresh = useRefreshTurn(sessionId)
  const confirm = useMutation({
    mutationFn: (turnId: string) => agentApi.confirm(sessionId, turnId),
    onSuccess: refresh,
  })
  const cancel = useMutation({
    mutationFn: (turnId: string) => agentApi.cancel(sessionId, turnId),
    onSuccess: refresh,
  })
  const retry = useMutation({
    mutationFn: (turnId: string) => agentApi.retry(sessionId, turnId),
    onSuccess: refresh,
  })
  return {
    confirm: (turnId: string) => confirm.mutate(turnId),
    cancel: (turnId: string) => cancel.mutate(turnId),
    retry: (turnId: string) => retry.mutate(turnId),
    busy: confirm.isPending || cancel.isPending || retry.isPending,
  }
}
