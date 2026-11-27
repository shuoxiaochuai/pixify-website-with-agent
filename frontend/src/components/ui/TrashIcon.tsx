/** 垃圾桶图标。会话列表与对话条目的删除按钮共用同一个形状。 */
export default function TrashIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path d="M2.5 4h11" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <path d="M6 2.5h4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <path
        d="M4 4l.7 8.3a1.3 1.3 0 0 0 1.3 1.2h4a1.3 1.3 0 0 0 1.3-1.2L12 4"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
    </svg>
  )
}
