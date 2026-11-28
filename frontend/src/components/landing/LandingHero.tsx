import { useRef, useState } from 'react'

import PromptComposer from './PromptComposer'

const SCENARIOS = [
  {
    label: '纯净背景',
    prompt: '把画面主体放在纯白背景上，居中构图，柔和顶光，边缘干净，输出 1:1 方图。',
  },
  {
    label: '场景融合',
    prompt: '把主体放到温暖的木质桌面场景，侧逆光、浅景深，保留主体原有材质与颜色。',
  },
  {
    label: '人物出镜',
    prompt: '生成人物与画面主体自然互动的半身展示图，简洁室内背景、自然光，主体细节清晰可辨。',
  },
  {
    label: '文字海报',
    prompt: '做一张海报：主标题「夏日远行」，副标题「去看看海」，右侧留出文字区，构图干净。',
  },
  {
    label: '多尺寸导出',
    prompt: '基于这张图输出 1:1、4:5、9:16 三个尺寸，主体不被裁切。',
  },
]

const FACTS = ['无需设计经验', '支持 JPG / PNG / WebP', '1:1 / 4:5 / 9:16 一次导出']

export default function LandingHero({
  initialPrompt,
  submitLabel,
  onStart,
}: {
  initialPrompt: string
  submitLabel: string
  onStart: (prompt: string) => void
}) {
  const [prompt, setPrompt] = useState(initialPrompt)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const pick = (text: string) => {
    setPrompt(text)
    inputRef.current?.focus()
  }

  return (
    <section className="relative isolate overflow-hidden">
      <div className="bg-glow absolute inset-x-0 -top-24 h-[560px]" aria-hidden />
      <div className="bg-grid absolute inset-x-0 -top-24 h-[560px]" aria-hidden />

      {/* 容器给到 4xl，让 14 字标题在桌面端稳定单行；正文与输入框各自再收窄 */}
      <div className="relative mx-auto max-w-4xl px-6 pt-16 pb-14 text-center sm:pt-24">
        <p
          className="border-line bg-paper/80 text-muted animate-rise inline-flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-xs backdrop-blur"
          style={{ animationDelay: '40ms' }}
        >
          <span className="bg-accent size-1.5 rounded-full" aria-hidden />
          AI 驱动的通用图片工作台
        </p>

        <h1
          className="text-ink animate-rise mt-6 text-[2rem] leading-[1.2] font-semibold tracking-tight text-balance sm:text-[2.7rem] lg:text-[3.25rem]"
          style={{ animationDelay: '120ms' }}
        >
          一句话，交付
          <span className="relative mx-1 whitespace-nowrap">
            {/* CJK 字身占满字框，高亮块需覆盖整个字高才像马克笔而非删除线 */}
            <span
              className="bg-accent/55 absolute inset-x-[-7px] top-[14%] bottom-[8%] -skew-x-6 rounded-[3px]"
              aria-hidden
            />
            <span className="relative">想要的画面</span>
          </span>
        </h1>

        <p
          className="text-muted animate-rise mx-auto mt-5 max-w-xl leading-relaxed"
          style={{ animationDelay: '200ms' }}
        >
          描述需求即可从零生成，也可上传图片继续编辑。抠图、换背景、局部精修、扩图、图层拆分与多尺寸导出，全部由 AI 自动编排。
        </p>

        <div className="animate-rise mx-auto mt-9 max-w-3xl" style={{ animationDelay: '280ms' }}>
          <PromptComposer
            value={prompt}
            onChange={setPrompt}
            onSubmit={() => onStart(prompt)}
            onAttach={() => onStart(prompt)}
            submitLabel={submitLabel}
            placeholder="描述你想要的画面，例：把这张照片的背景换成海边日落，人物与光线保持不变…"
            inputRef={inputRef}
          />
        </div>

        <div
          className="animate-rise mt-5 flex flex-wrap justify-center gap-2"
          style={{ animationDelay: '360ms' }}
        >
          {SCENARIOS.map((scenario) => (
            <button
              key={scenario.label}
              type="button"
              onClick={() => pick(scenario.prompt)}
              className="border-line bg-paper/70 text-muted hover:border-brand hover:text-brand-strong rounded-full border px-3.5 py-1.5 text-xs transition-colors"
            >
              {scenario.label}
            </button>
          ))}
        </div>

        <ul
          className="text-faint animate-rise mt-10 flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-xs"
          style={{ animationDelay: '440ms' }}
        >
          {FACTS.map((fact) => (
            <li key={fact} className="flex items-center gap-1.5">
              <span className="bg-line-strong size-1 rounded-full" aria-hidden />
              {fact}
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
