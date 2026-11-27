import { api } from '@/api/client'
import type { RunStatus } from '@/api/runs'

export type StepStatus =
  | 'pending'
  | 'waiting'
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'canceled'

export type PlanStep = {
  id: string
  tool: string
  label: string
  depends_on: string[]
  run_id: string | null
  status: StepStatus
}

// 一轮指令走的通道：edit 会产计划并动画布，chat 只出一句话，
// mixed 是同一句里既有提问又有修图动作——正文先答问题，步骤照旧属于修图
export type TurnKind = 'edit' | 'chat' | 'mixed'

export type Turn = {
  id: string
  revision: number
  goal: string
  reply: string
  // 可选：后端没重启或命中旧缓存时该字段缺失，不能因此炸掉渲染
  kind?: TurnKind
  status: RunStatus
  error: string | null
  created_at: string
  steps: PlanStep[]
}

// 一轮指令的过程事件。后台跑完才落库，这几秒的界面全靠它们撑着
export type TurnEvent =
  | { type: 'stage'; text: string }
  | { type: 'kind'; value: TurnKind }
  | { type: 'plan'; index: number; label: string }
  | { type: 'token'; text: string }
  | { type: 'done'; status: RunStatus; kind: TurnKind }
  | { type: 'failed'; error?: string }
  | { type: 'snapshot'; turn: Turn }

export const agentApi = {
  turns: (sessionId: string) => api.get<Turn[]>(`/sessions/${sessionId}/messages`),
  send: (sessionId: string, text: string) =>
    api.post<Turn>(`/sessions/${sessionId}/messages`, { text }),
  confirm: (sessionId: string, turnId: string) =>
    api.post<Turn>(`/sessions/${sessionId}/messages/${turnId}/confirm`),
  cancel: (sessionId: string, turnId: string) =>
    api.post<Turn>(`/sessions/${sessionId}/messages/${turnId}/cancel`),
  retry: (sessionId: string, turnId: string) =>
    api.post<Turn>(`/sessions/${sessionId}/messages/${turnId}/retry`),
}