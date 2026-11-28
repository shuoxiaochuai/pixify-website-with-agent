import type { Asset } from '@/api/assets'
import { api, apiError } from '@/api/client'

export type LesionType = 'EX' | 'HE' | 'MA' | 'SE'

/** 与后端 app/lesion/engine.py 的 TYPES 保持一致；顺序即界面顺序 */
export const LESION_TYPES: LesionType[] = ['EX', 'HE', 'MA', 'SE']

export const LESION_LABELS: Record<LesionType, string> = {
  EX: '出血点',
  HE: '硬性渗出',
  MA: '微血管瘤',
  SE: '软性渗出',
}

/** 与后端 COLORS 同色，勾选框旁的小圆点用 */
export const LESION_COLORS: Record<LesionType, string> = {
  EX: '#EB4034',
  HE: '#FFA81A',
  MA: '#26C6DA',
  SE: '#43D252',
}

export type LesionStatus = {
  available: boolean
  reason: string | null
  device: 'cuda' | 'cpu'
  models: Record<LesionType, boolean>
}

export const lesionApi = {
  status: () => api.get<LesionStatus>('/lesion/status'),
  /** 真实标注（IDRiD 的 .tif 也能传），后端统一转灰度 PNG 存成 mask 素材 */
  async uploadMask(file: File): Promise<Asset> {
    const body = new FormData()
    body.append('file', file)
    const response = await fetch('/api/lesion/masks', { method: 'POST', body })
    if (!response.ok) throw await apiError(response, '标注上传失败')
    return response.json()
  },
}