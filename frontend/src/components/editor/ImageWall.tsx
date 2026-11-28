import type { Asset, AssetKind } from '@/api/assets'

const KIND_LABELS: Record<AssetKind, string> = {
  original: '原图',
  generated: '生成',
  subject: '主体',
  background: '背景',
  mask: '遮罩',
  marketing: '衍生',
  export: '导出',
}

/** 会话内全部图片，含未采用的候选。点击切换画布当前图，不覆盖任何已有结果。 */
export default function ImageWall({
  assets,
  currentId,
  usedIds,
  disabled,
  onPick,
  onDetach,
}: {
  assets: Asset[]
  currentId: string
  /** 正摆在画布上的那些，包成图层的那张，摘掉画布就空一块，所以不给叉 */
  usedIds: Set<string>
  disabled: boolean
  onPick: (assetId: string) => void
  onDetach: (assetId: string) => void
}) {
  return (
    <div className="border-line bg-paper shrink-0 border-t">
      <div className="scrollbar-slim flex scroll-px-4 items-center gap-2 overflow-x-auto px-4 py-3">
        {assets
          .filter((asset) => asset.kind !== 'mask')
          .filter(
            (asset) =>
              asset.id === currentId || (asset.kind !== 'subject' && asset.kind !== 'background'),
          )
          .map((asset) => {
          const active = asset.id === currentId
          const removable = !active && !usedIds.has(asset.id)
          return (
            <div key={asset.id} className="group relative size-16 shrink-0">
              <button
                type="button"
                disabled={disabled || active}
                onClick={() => onPick(asset.id)}
                title={
                  active
                    ? `当前画布 · ${KIND_LABELS[asset.kind]}`
                    : `采用这张 · ${KIND_LABELS[asset.kind]} ${asset.width}×${asset.height}`
                }
                className={`bg-canvas rounded-chip relative size-full overflow-hidden border-2 transition-all duration-150 disabled:cursor-default ${
                  active
                    ? 'border-brand shadow-control'
                    : 'border-line hover:border-brand hover:shadow-control active:scale-[0.97]'
                }`}
              >
                <img
                  src={asset.url}
                  alt=""
                  loading="lazy"
                  className="size-full object-contain transition-transform duration-150 group-hover:scale-[1.03]"
                />
                <span className="bg-ink/70 absolute right-0 bottom-0 left-0 py-0.5 text-[10px] text-white">
                  {KIND_LABELS[asset.kind]}
                </span>
              </button>
              {removable && (
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => onDetach(asset.id)}
                  title="从图片墙去掉这张（素材库里的原件不动）"
                  className="bg-ink/75 hover:bg-ink text-white pointer-events-none absolute top-0.5 right-0.5 grid size-5 place-items-center rounded-full opacity-0 transition-opacity duration-150 group-hover:pointer-events-auto group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:opacity-100 disabled:cursor-not-allowed"
                >
                  <CloseIcon />
                </button>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function CloseIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 12 12" fill="none" aria-hidden>
      <path d="M2.75 2.75l6.5 6.5M9.25 2.75l-6.5 6.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  )
}
