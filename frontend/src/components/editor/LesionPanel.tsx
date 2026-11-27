import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import {
  LESION_COLORS,
  LESION_LABELS,
  LESION_TYPES,
  lesionApi,
  type LesionType,
} from '@/api/lesion'
import Button from '@/components/ui/Button'

/** 0.5 是服务端默认值；0.8 沿用官方可视化脚本的口径，漏检更少但更保守 */
const THRESHOLDS: { value: number; label: string; hint: string }[] = [
  { value: 0.5, label: '标准', hint: '默认阈值，召回优先' },
  { value: 0.8, label: '严格', hint: '更保守，适合确认明显病灶' },
]

export default function LesionPanel({
  disabled,
  progress = null,
  onApply,
}: {
  disabled: boolean
  progress?: number | null
  onApply: (params: { types: LesionType[]; threshold: number }) => void
}) {
  const status = useQuery({
    queryKey: ['lesion', 'status'],
    queryFn: lesionApi.status,
    // 算力是按需开卡的，隔一会儿回来重问一次即可，不必每次挂载都探
    staleTime: 30_000,
    retry: false,
  })
  const [types, setTypes] = useState<LesionType[]>([...LESION_TYPES])
  const [threshold, setThreshold] = useState(0.5)

  const ready = status.data?.available === true
  const usable = LESION_TYPES.filter((name) => status.data?.models[name] !== false)
  const toggle = (name: LesionType) =>
    setTypes((current) =>
      current.includes(name) ? current.filter((item) => item !== name) : [...current, name],
    )

  const availability = status.isError
    ? '无法探测推理环境，请确认服务可用'
    : status.isLoading
      ? '正在检查推理环境…'
      : ready
        ? `已就绪 · ${status.data?.device === 'cuda' ? 'GPU' : 'CPU'}`
        : (status.data?.reason ?? '病灶分割暂不可用')

  return (
    <section className="border-line border-b px-4 py-4">
      <h2 className="text-muted mb-2.5 text-xs font-medium">病灶分割</h2>
      <p className="text-faint mb-2.5 text-[11px] leading-relaxed">
        面向眼底照片，四类病灶各生成一个半透明标注图层，可单独显隐、调透明度，可撤销；
        重跑同一类会替换旧标注，不会越叠越红。
      </p>

      <ul className="mb-3 space-y-1">
        {LESION_TYPES.map((name) => {
          const deployed = status.data?.models[name] !== false
          return (
            <li key={name}>
              <label className="text-ink flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={types.includes(name)}
                  disabled={disabled || !deployed}
                  onChange={() => toggle(name)}
                />
                <span
                  className="size-2 shrink-0 rounded-full"
                  style={{ background: LESION_COLORS[name] }}
                />
                <span className="flex-1">
                  {name} {LESION_LABELS[name]}
                </span>
                {!deployed && <span className="text-faint text-[10px]">未部署</span>}
              </label>
            </li>
          )
        })}
      </ul>

      <span className="text-muted mb-1 block text-[11px]">判定阈值</span>
      <div className="mb-3 flex gap-1">
        {THRESHOLDS.map((item) => (
          <Button
            key={item.value}
            variant="outline"
            size="sm"
            block
            active={threshold === item.value}
            disabled={disabled}
            title={item.hint}
            onClick={() => setThreshold(item.value)}
          >
            {item.label} {item.value}
          </Button>
        ))}
      </div>

      <div className="mb-2 flex items-center justify-between gap-2 text-[11px]">
        <span className={ready ? 'text-faint' : 'text-danger'}>{availability}</span>
        <button
          type="button"
          className="text-muted hover:text-ink shrink-0 underline"
          onClick={() => void status.refetch()}
        >
          重新检测
        </button>
      </div>

      <Button
        variant="solid"
        block
        disabled={disabled || !ready || types.length === 0 || usable.length === 0}
        progress={progress}
        title={
          ready
            ? types.length > 0
              ? '开始分割，结果追加为标注图层'
              : '至少勾选一类病灶'
            : '推理环境不可用'
        }
        onClick={() => onApply({ types, threshold })}
      >
        开始分割
      </Button>
    </section>
  )
}