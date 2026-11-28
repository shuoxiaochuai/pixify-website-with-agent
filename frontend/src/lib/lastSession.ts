/**
 * 记住上一次正在编辑的会话，重新进入工作台时直接回到它。
 *
 * 按用户名分开存：同一浏览器换账号登录不会误开别人的会话；
 * 存储不可用（隐私模式、配额满）时静默退回创作页，不阻塞任何入口。
 */
const KEY = 'retouch:last-session'

type Stored = { user: string; id: string }

function read(): Stored | null {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return null
    const { user, id } = JSON.parse(raw) as Partial<Stored>
    if (typeof user !== 'string' || typeof id !== 'string' || !user || !id) return null
    return { user, id }
  } catch {
    return null
  }
}

export function rememberSession(user: string | null | undefined, id: string) {
  if (!user) return
  try {
    localStorage.setItem(KEY, JSON.stringify({ user, id }))
  } catch {
    // 记不住就退化为默认入口，不影响编辑本身
  }
}

/** 只清掉与这条会话对应的记录，避免误伤其他标签页刚记下的新会话。 */
export function forgetSession(id: string) {
  if (read()?.id !== id) return
  try {
    localStorage.removeItem(KEY)
  } catch {
    // 清不掉也没关系：编辑页对失效会话有兜底提示
  }
}

/** 工作台入口：上次编辑的会话属于当前账号就用它，否则回创作页。 */
export function workbenchEntry(user: string | null | undefined): string {
  const stored = read()
  return stored && user && stored.user === user ? `/editor/${stored.id}` : '/create'
}