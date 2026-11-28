import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import type { Asset } from '@/api/assets'
import {
  LESION_COLORS,
  LESION_LABELS,
  LESION_TYPES,
  lesionApi,
  type LesionType,
} from '@/api/lesion'
import Button from '@/components/ui/Button'
import { buttonClass } from '@/components/ui/buttonStyles'
import { errorMessage } from '@/hooks/useAuth'
import { toast } from '@/stores/toasts'

const DEFAULT_THRESHOLD = '0.5'
const MAX_THRESHOLD = 0.99
const MASK_ACCEPT = '.png,.jpg,.jpeg,.webp,.tif,.tiff'

export type LesionRunResult = {
  runId: string
  threshold?: number
  dice?: Record<string, number> | null
}

export type LesionApply = {
  types: LesionType[]
  threshold?: number
  auto_threshold: boolean
  gt_assets?: Partial<Record<LesionType, string>>
}

type UploadedMask = { id: string; name: string }

export default function LesionPanel({
  disabled,
  progress = null,
  result = null,
  onApply,
}: {
  disabled: boolean
  progress?: number | null
  result?: LesionRunResult | null
  onApply: (params: LesionApply) => void
}) {
  const status = useQuery({
    queryKey: ['lesion', 'status'],
    queryFn: lesionApi.status,
    // 算力是按需开卡的，隔一会儿回来重问一次即可，不必每次挂载都探
    staleTime: 30_000,
    retry: false,
  })
  const [types, setTypes] = useState<LesionType[]>([...LESION_TYPES])
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD)
  // 自动阈值只给验证集用：关掉后仍保留上次找到的值，可继续手改
  const [auto, setAuto] = useState(false)
  const [masks, setMasks] = useState<Partial<Record<LesionType, UploadedMask>>>({})
  const [uploading, setUploading] = useState<LesionType | null>(null)

  const ready = status.data?.available === true
  const usable = LESION_TYPES.filter((name) => status.data?.models[name] !== false)
  // 后端算出的最优阈值直接由任务结果派生，不在 effect 里回写 state
  const best =
    result && typeof result.threshold === 'number'
      ? { threshold: result.threshold, dice: result.dice ?? null }
      : null
  const value = Number(threshold)
  const thresholdError =
    threshold.trim() === ''
      ? '请输入阈值'
      : !Number.isFinite(value)
        ? '请输入 0–0.99 的数字'
        : value < 0 || value > MAX_THRESHOLD
          ? '阈值需在 0–0.99 之间'
          : null
  const missingMasks = types.filter((name) => !masks[name])
  const masksReady = types.length > 0 && missingMasks.length === 0
  const uploadLocked = disabled || !ready || uploading !== null

  const toggle = (name: LesionType) =>
    setTypes((current) =>
      current.includes(name) ? current.filter((item) => item !== name) : [...current, name],
    )

  const upload = async (name: LesionType, file: File) => {
    setUploading(name)
    try {
      const asset: Asset = await lesionApi.uploadMask(file)
      setMasks((current) => ({ ...current, [name]: { id: asset.id, name: file.name } }))
    } catch (error) {
      toast(errorMessage(error), 'danger')
    } finally {
      setUploading(null)
    }
  }

  const submit = () => {
    if (auto) {
      if (!masksReady) return
      const gt_assets: Partial<Record<LesionType, string>> = {}
      for (const name of types) {
        const mask = masks[name]
        if (mask) gt_assets[name] = mask.id
      }
      onApply({ types, auto_threshold: true, gt_assets })
      return
    }
    if (thresholdError !== null) return
    onApply({ types, threshold: value, auto_threshold: false })
  }

  const availability = status.isError
    ? '无法探测推理环境，请确认服务可用'
    : status.isLoading
      ? '正在检查推理环境…'
      : ready
        ? `已就绪 · ${status.data?.device === 'cuda' ? 'GPU' : 'CPU'}`
        : (status.data?.reason ?? '病灶分割暂不可用')

  const diceValues = best?.dice ? Object.values(best.dice) : []
  const meanDice = diceValues.length
    ? diceValues.reduce((sum, item) => sum + item, 0) / diceValues.length
    : null

  const startHint = !ready
    ? '推理环境不可用'
    : types.length === 0
      ? '至少勾选一类病灶'
      : auto
        ? masksReady
          ? '按真实标注遍历阈值，再用最优阈值分割'
          : '请先为勾选的每一类上传真实标注'
        : (thresholdError ?? '开始分割，结果追加为标注图层')

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

      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-muted text-[11px]">判定阈值</span>
        <span className="text-faint text-[10px] tabular-nums">0–{MAX_THRESHOLD}</span>
      </div>
      <input
        value={auto && best ? best.threshold.toFixed(2) : threshold}
        disabled={disabled || auto}
        inputMode="decimal"
        aria-label="判定阈值"
        aria-invalid={thresholdError !== null && !auto}
        className={`rounded-control mb-1 w-full border px-2.5 py-1.5 text-xs outline-none transition-colors duration-150 ${
          auto
            ? 'border-line bg-soft text-faint cursor-not-allowed'
            : thresholdError
              ? 'border-danger text-danger'
              : 'border-line text-ink focus:border-line-strong'
        }`}
        onChange={(event) => setThreshold(event.target.value)}
      />
      {auto ? (
        <p className="text-faint mb-2 text-[10px] leading-relaxed">
          {best
            ? `已回填最优阈值 ${best.threshold.toFixed(2)}${
                meanDice === null ? '' : ` · 平均 Dice ${meanDice.toFixed(3)}`
              }`
            : '阈值已锁定，分割完成后自动回填最优阈值'}
        </p>
      ) : thresholdError ? (
        <p className="text-danger mb-2 text-[10px] leading-relaxed">{thresholdError}</p>
      ) : (
        <p className="text-faint mb-2 text-[10px] leading-relaxed">
          输入多少用多少；数值越高越严格，漏检可能更多。
        </p>
      )}

      <label className="text-ink mb-2.5 flex items-start gap-2 text-[11px] leading-snug">
        <input
          type="checkbox"
          className="mt-0.5"
          checked={auto}
          disabled={disabled}
          onChange={(event) => {
            const next = event.target.checked
            // 关掉自动时把找到的最优值留在输入框，继续手改也方便
            if (!next && best) setThreshold(best.threshold.toFixed(2))
            setAuto(next)
          }}
        />
        <span>
          自动尝试最优阈值
          <span className="text-faint">（仅限验证集）</span>
        </span>
      </label>

      {auto && (
        <div className="mb-3 space-y-1.5">
          <p className="text-faint text-[10px] leading-relaxed">
            为勾选的每一类上传真实标注，模型逐档比对 Dice 后取最优阈值。
          </p>
          {types.map((name) => {
            const mask = masks[name]
            return (
              <div key={name} className="flex items-center gap-2 text-[11px]">
                <span className="text-ink w-16 shrink-0">
                  {name} {LESION_LABELS[name]}
                </span>
                <span
                  className={`min-w-0 flex-1 truncate ${mask ? 'text-faint' : 'text-danger'}`}
                  title={mask?.name}
                >
                  {mask ? mask.name : '未上传'}
                </span>
                <label
                  className={`${buttonClass({ variant: 'outline', size: 'sm' })} shrink-0 ${
                    uploadLocked ? 'cursor-not-allowed opacity-40' : 'cursor-pointer'
                  }`}
                >
                  {uploading === name ? '上传中' : mask ? '更换' : '上传标注'}
                  <input
                    type="file"
                    accept={MASK_ACCEPT}
                    className="hidden"
                    disabled={uploadLocked}
                    onChange={(event) => {
                      const file = event.target.files?.[0]
                      event.target.value = ''
                      if (file) void upload(name, file)
                    }}
                  />
                </label>
              </div>
            )
          })}
        </div>
      )}

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
        disabled={
          disabled ||
          !ready ||
          types.length === 0 ||
          usable.length === 0 ||
          uploading !== null ||
          (auto ? !masksReady : thresholdError !== null)
        }
        progress={progress}
        title={startHint}
        onClick={submit}
      >
        开始分割
      </Button>
    </section>
  )
}