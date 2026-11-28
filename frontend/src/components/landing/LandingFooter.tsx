import BrandMark from '@/components/BrandMark'

export default function LandingFooter() {
  return (
    <footer className="border-line border-t">
      <div className="text-faint mx-auto flex max-w-5xl flex-col items-center justify-between gap-3 px-6 py-8 text-xs sm:flex-row">
        <BrandMark size="sm">
          <span className="text-muted font-medium">AI 修图智能体</span>
        </BrandMark>
        <p>面向所有人的 AI 图片生成与编辑工作台</p>
      </div>
    </footer>
  )
}
