import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'

import type { PlanStep, StepStatus, Turn } from '@/api/agent'
import { isTerminal } from '@/api/runs'
import Button from '@/components/ui/Button'
import ProgressBar from '@/components/ui/ProgressBar'
import TrashIcon from '@/components/ui/TrashIcon'
import {
  useDeleteTurn,
  usePlanActions,
  useTurnStream,
  useTurns,
  type TurnLive,
} from '@/hooks/useAgent'
import { useRun } from '@/hooks/useRun'

const STEP_LABEL: Record<StepStatus, string> = {
  pending: '等待中',
  waiting: '待确认',
  queued: '排队中',
  running: '进行中',
  succeeded: '已完成',
  failed: '未完成',
  canceled: '已取消',
}

export default function AgentConversation({ sessionId }: { sessionId: string }) {
  const { data: turns = [], isPending } = useTurns(sessionId)
  const deleteTurn = useDeleteTurn(sessionId)
  // 一次只让一条处在待确认删除状态，确认条直接顶掉那一轮，不弹窗
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  const end = useRef<HTMLDivElement>(null)
  const latest = turns.at(-1)
  // 后台跑完才落库，最新一轮既没答复也没计划的那几秒才值得看过程
  const liveId =
    latest && !isTerminal(latest.status) && !latest.reply && !latest.steps.length
      ? latest.id
      : null
  const live = useTurnStream(liveId, sessionId)
  // 首屏就带着进行中的轮次：说明是刷新或重新进入，后端会从检查点/已落库计划续跑
  const seeded = useRef(false)
  const [restoring, setRestoring] = useState(false)
  useEffect(() => {
    if (isPending || seeded.current) return
    seeded.current = true
    setRestoring(turns.some((turn) => turn.status === 'running'))
  }, [isPending, turns])
  useEffect(() => {
    if (restoring && !turns.some((turn) => turn.status === 'running')) setRestoring(false)
  }, [restoring, turns])
  // 阶段与正文是一条条长出来的，轮次状态却没变，光靠 turns.length 滚不动
  const growth = liveId ? `${live.stages.length}.${live.steps.length}.${live.text.length}` : ''

  useEffect(() => {
    end.current?.scrollIntoView({ block: 'end' })
  }, [turns.length, latest?.status, growth])

  if (isPending) {
    return <p className="text-faint min-h-0 flex-1 px-4 py-4 text-xs">加载中…</p>
  }

if (turns.length === 0) {
    return (
      <p className="text-faint min-h-0 flex-1 px-4 py-4 text-xs leading-relaxed">
        用一句话说明要怎么改，例如「水平翻转」或「去背景再调亮一点」。多步计划会先列出再执行。
        顺带问点别的也行，我只简短答两句。
      </p>
    )
  }

  return (
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-3 py-3">
      {restoring && (
        <p className="text-faint px-1 text-[10px] leading-relaxed">
          上次还有未完成的执行，正在同步进度；刷新或重进页面不会打断它。
        </p>
      )}
      {turns.map((turn) => (
        <TurnBlock
          key={turn.id}
          turn={turn}
          sessionId={sessionId}
          live={turn.id === liveId ? live : null}
          pendingDelete={turn.id === pendingDelete}
          deleting={deleteTurn.isPending}
          onAskDelete={() => setPendingDelete(turn.id)}
          onCancelDelete={() => setPendingDelete(null)}
          onDelete={() =>
            deleteTurn.mutate(turn.id, { onSuccess: () => setPendingDelete(null) })
          }
        />
      ))}
      <div ref={end} />
    </div>
  )
}

function TurnBlock({
  turn,
  sessionId,
  live,
  pendingDelete,
  deleting,
  onAskDelete,
  onCancelDelete,
  onDelete,
}: {
  turn: Turn
  sessionId: string
  live: TurnLive | null
  pendingDelete: boolean
  deleting: boolean
  onAskDelete: () => void
  onCancelDelete: () => void
  onDelete: () => void
}) {
  if (pendingDelete) {
    return (
      <div className="border-line bg-soft animate-fade-in flex flex-col gap-1.5 rounded-chip border px-2.5 py-2">
        <p className="text-muted text-[11px] leading-snug">
          删除这条对话？还在跑的任务会一并取消；已落定的画布改动不回退，要回退用撤销。
        </p>
        <div className="flex justify-end gap-1">
          <Button size="sm" disabled={deleting} onClick={onCancelDelete}>
            取消
          </Button>
          <Button size="sm" variant="solid" disabled={deleting} onClick={onDelete}>
            {deleting ? '删除中' : '确认删除'}
          </Button>
        </div>
      </div>
    )
  }

  // 闲聊轮没有步骤、没有确认，也不该长得像一次执行结果：分流一确定就换样式
  const chatting = (live?.kind ?? turn.kind) === 'chat'
  const reply = turn.reply || live?.text || ''

  return (
    <div className="group relative space-y-2">
      <p className="bg-brand-soft text-brand-strong ml-6 rounded-[12px] px-3 py-2 text-xs leading-relaxed">
        {turn.goal}
      </p>
      {/* 悬停才显现，不跟正文抢视线；键盘聚焦到时也要看得见 */}
      <span className="absolute right-0 top-0 z-10 opacity-0 transition-opacity duration-150 group-hover:opacity-100 focus-within:opacity-100">
        <Button size="sm" variant="icon" title="删除这条对话" onClick={onAskDelete}>
          <TrashIcon />
        </Button>
      </span>

      {live && <Thinking stages={live.stages} steps={live.steps} />}

      {turn.error ? (
        <p className="text-danger mr-6 text-xs leading-relaxed">{turn.error}</p>
      ) : reply ? (
        chatting ? (
          <p className="border-line text-faint mr-6 rounded-[12px] border border-dashed px-3 py-2 text-xs leading-relaxed">
            {reply}
          </p>
        ) : (
          // 混合轮是「文字答案 + 修图那段的说法」两段，按行分开渲染
          <p className="text-ink mr-6 whitespace-pre-line text-xs leading-relaxed">{reply}</p>
        )
      ) : null}

      {turn.steps.map((step, index) => (
        <StepCard key={step.id} index={index + 1} step={step} sessionId={sessionId} />
      ))}

      {/* 没有步骤就没有可确认、可取消、可重试的东西：规划阶段就失败的那一轮只能看错误文字 */}
      {chatting || !turn.steps.length ? null : <PlanActions turn={turn} sessionId={sessionId} />}
    </div>
  )
}

/**
 * 伪思考：只说已经确定会发生的事——在看意思、在安排第几步、这一步做什么。
 *
 * 模型的原文不外泄，也不开思考模式多花 token，步骤名是工具调用刚起头时就报出来的。
 */
function Thinking({ stages, steps }: { stages: string[]; steps: (string | undefined)[] }) {
  const named = steps.filter((label): label is string => Boolean(label))
  if (!stages.length && !named.length) return null

  return (
    <div className="border-line mr-6 space-y-1 rounded-[12px] border border-dashed px-3 py-2">
      {stages.map((line, index) => (
        <p
          key={`${index}-${line}`}
          className="text-faint flex items-center gap-1.5 text-[10px] leading-relaxed"
        >
          <span
            className={`size-1 shrink-0 rounded-full ${
              index === stages.length - 1 && !named.length ? 'bg-brand animate-pulse' : 'bg-line-strong'
            }`}
          />
          {line}
        </p>
      ))}
      {named.map((label, index) => (
        <p
          key={`${index}-${label}`}
          className="animate-fade-in text-faint pl-4 text-[10px] leading-relaxed"
        >
          {index + 1}. {label}
        </p>
      ))}
    </div>
  )
}

function PlanActions({ turn, sessionId }: { turn: Turn; sessionId: string }) {
  const actions = usePlanActions(sessionId)
  if (turn.status === 'queued') {
    return (
      <div className="mr-6 flex gap-1.5">
        <Action disabled={actions.busy} onClick={() => actions.confirm(turn.id)}>
          执行计划
        </Action>
        <Action disabled={actions.busy} onClick={() => actions.cancel(turn.id)}>
          取消
        </Action>
      </div>
    )
  }
  if (turn.status === 'running') {
    return (
      <div className="mr-6">
        {/* 后端现在连正在跑的任务一起停，不用硬等读条跑完 */}
        <Action disabled={actions.busy} onClick={() => actions.cancel(turn.id)}>
          取消任务
        </Action>
      </div>
    )
  }
  if (turn.status === 'failed') {
    return (
      <div className="mr-6">
        <Action disabled={actions.busy} onClick={() => actions.retry(turn.id)}>
          重试失败步骤
        </Action>
      </div>
    )
  }
  return null
}

function Action({
  children,
  disabled,
  onClick,
}: {
  children: ReactNode
  disabled?: boolean
  onClick: () => void
}) {
  return (
    <Button variant="outline" size="sm" disabled={disabled} onClick={onClick}>
      {children}
    </Button>
  )
}

function StepCard({
  index,
  step,
  sessionId,
}: {
  index: number
  step: PlanStep
  sessionId: string
}) {
  const queryClient = useQueryClient()
  const { status, progress, stage, error } = useRun(step.run_id)
  const watched = useRef(false)
  const current = status ?? step.status
  const running = current === 'queued' || current === 'running'

  useEffect(() => {
    if (status === 'queued' || status === 'running') watched.current = true
    if (!watched.current || !status || !isTerminal(status)) return
    watched.current = false
    void queryClient.invalidateQueries({ queryKey: ['session', sessionId] })
    void queryClient.invalidateQueries({ queryKey: ['session', sessionId, 'history'] })
    void queryClient.invalidateQueries({ queryKey: ['session', sessionId, 'messages'] })
  }, [status, sessionId, queryClient])

  return (
    <div className="border-line mr-6 rounded-[12px] border px-3 py-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-ink text-xs font-medium">
          {index}. {step.label}
        </span>
        <span className="text-faint shrink-0 text-[10px] tabular-nums">
          {running ? `${progress}%` : STEP_LABEL[current]}
        </span>
      </div>

      {running && (
        <>
          <ProgressBar value={progress} className="mt-2 h-0.5 rounded-full" />
          <p className="text-faint mt-1.5 text-[10px]">{stage}</p>
        </>
      )}

      {error && <p className="text-danger mt-1.5 text-[10px] leading-relaxed">{error}</p>}
    </div>
  )
}
