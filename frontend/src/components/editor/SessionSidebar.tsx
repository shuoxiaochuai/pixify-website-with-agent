import { useState } from 'react'
import { Link, NavLink, useNavigate } from 'react-router-dom'

import AgentConversation from '@/components/editor/AgentConversation'
import MessageComposer from '@/components/editor/MessageComposer'
import Button from '@/components/ui/Button'
import { buttonClass } from '@/components/ui/buttonStyles'
import PencilIcon from '@/components/ui/PencilIcon'
import TrashIcon from '@/components/ui/TrashIcon'
import { useSendMessage, useTurns } from '@/hooks/useAgent'
import { errorMessage } from '@/hooks/useAuth'
import { useDeleteSession, useRenameSession, useSessions } from '@/hooks/useSessions'
import { formatDateTime } from '@/lib/format'

export default function SessionSidebar({ activeId }: { activeId: string }) {
  // 历史对话默认收起，尽量把纵向空间留给当前对话
  const [historyOpen, setHistoryOpen] = useState(false)

  return (
    <aside className="border-line bg-paper flex w-72 shrink-0 flex-col border-r">
      <div className="border-line flex items-center gap-2 border-b p-3">
        <Link
          to="/create"
          className={`${buttonClass({ variant: 'outline' })} flex-1 py-2 text-center text-sm`}
        >
          新对话
        </Link>
        <Button
          active={historyOpen}
          className="px-3 py-2"
          onClick={() => setHistoryOpen((open) => !open)}
        >
          历史
        </Button>
      </div>

      {historyOpen && <SessionList activeId={activeId} />}

      {activeId ? (
        <Conversation sessionId={activeId} />
      ) : (
        <p className="text-faint min-h-0 flex-1 px-4 py-4 text-xs leading-relaxed">
          打开一个历史对话，或回到创作页开始新的一张。
        </p>
      )}
    </aside>
  )
}

function Conversation({ sessionId }: { sessionId: string }) {
  const send = useSendMessage(sessionId)
  const { data: turns = [] } = useTurns(sessionId)
  const planActive = turns.some((turn) => turn.status === 'queued' || turn.status === 'running')

  return (
    <>
      <AgentConversation sessionId={sessionId} />
      <MessageComposer
        pending={send.isPending || planActive}
        error={send.isError ? errorMessage(send.error) : null}
        placeholder={
          planActive
            ? '当前计划进行中，确认、取消或等它结束后再发'
            : '说明要怎么改，例如：去背景再水平翻转。回车发送'
        }
        onSend={(text) => send.mutate(text)}
      />
    </>
  )
}

function SessionList({ activeId }: { activeId: string }) {
  const { data: sessions = [], isPending } = useSessions()
  const navigate = useNavigate()
  const remove = useDeleteSession()
  const rename = useRenameSession()
  // 一次只让一条处在待确认状态，确认条直接顶掉这一行，不弹窗
  const [pendingId, setPendingId] = useState<string | null>(null)
  // 未命名的会话打开时留空输入框，直接键入名字即可
  const [editing, setEditing] = useState<{ id: string; draft: string } | null>(null)

  if (isPending) {
    return <p className="text-faint border-line border-b px-4 py-3 text-xs">加载中…</p>
  }

  if (sessions.length === 0) {
    return <p className="text-faint border-line border-b px-4 py-3 text-xs">还没有会话</p>
  }

  const confirmDelete = (id: string) => {
    remove.mutate(id, {
      onSuccess: () => {
        setPendingId(null)
        // 删的正是当前开着的那条，画布已经没有归属，退回创作页
        if (id === activeId) navigate('/create')
      },
    })
  }

  const saveRename = () => {
    if (!editing) return
    const title = editing.draft.trim()
    if (!title) return
    rename.mutate({ id: editing.id, title }, { onSuccess: () => setEditing(null) })
  }

  return (
    <ul className="border-line scrollbar-slim animate-fade-in max-h-52 shrink-0 space-y-0.5 overflow-y-auto border-b p-2">
      {sessions.map((session) => (
        <li key={session.id} className="group relative">
          {editing && editing.id === session.id ? (
            <div className="border-line bg-soft animate-fade-in rounded-chip flex flex-col gap-1.5 border px-2.5 py-2">
              <input
                autoFocus
                value={editing.draft}
                maxLength={80}
                placeholder="给这条对话起个名字"
                aria-label="对话标题"
                className="border-line text-ink placeholder:text-faint rounded-control focus:border-line-strong w-full border px-2 py-1 text-xs outline-none"
                onChange={(event) => {
                  const draft = event.target.value
                  setEditing((current) => (current ? { ...current, draft } : current))
                }}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') saveRename()
                  if (event.key === 'Escape') setEditing(null)
                }}
              />
              <div className="flex justify-end gap-1">
                <Button size="sm" disabled={rename.isPending} onClick={() => setEditing(null)}>
                  取消
                </Button>
                <Button
                  size="sm"
                  variant="solid"
                  disabled={rename.isPending || !editing.draft.trim()}
                  onClick={saveRename}
                >
                  {rename.isPending ? '保存中' : '保存'}
                </Button>
              </div>
            </div>
          ) : session.id === pendingId ? (
            <div className="border-line bg-soft animate-fade-in flex flex-col gap-1.5 rounded-chip border px-2.5 py-2">
              <p className="text-muted text-[11px] leading-snug">
                删除「{session.title}」？对话与编辑记录会一起清掉，删了找不回来（图仍留在素材库）。
              </p>
              <div className="flex justify-end gap-1">
                <Button size="sm" disabled={remove.isPending} onClick={() => setPendingId(null)}>
                  取消
                </Button>
                <Button
                  size="sm"
                  variant="solid"
                  disabled={remove.isPending}
                  onClick={() => confirmDelete(session.id)}
                >
                  {remove.isPending ? '删除中' : '确认删除'}
                </Button>
              </div>
            </div>
          ) : (
            <>
              <NavLink
                to={`/editor/${session.id}`}
                className={`rounded-chip block px-2.5 pr-14 py-2 transition-colors duration-150 ${
                  session.id === activeId
                    ? 'bg-brand-soft text-brand-strong'
                    : 'text-muted hover:bg-soft hover:text-ink'
                }`}
              >
                <span className="block truncate text-xs font-medium">{session.title}</span>
                <span className="text-faint block text-[10px] tabular-nums">
                  {formatDateTime(session.updated_at)}
                </span>
              </NavLink>
              <span className="absolute top-1/2 right-1 flex -translate-y-1/2 items-center gap-0.5">
                <Button
                  size="sm"
                  variant="icon"
                  title="重命名这条对话"
                  onClick={() => {
                    setPendingId(null)
                    setEditing({
                      id: session.id,
                      draft: session.title === '未命名会话' ? '' : session.title,
                    })
                  }}
                >
                  <PencilIcon />
                </Button>
                <Button
                  size="sm"
                  variant="icon"
                  title="删除这条对话"
                  onClick={() => {
                    setEditing(null)
                    setPendingId(session.id)
                  }}
                >
                  <TrashIcon />
                </Button>
              </span>
            </>
          )}
        </li>
      ))}
    </ul>
  )
}
