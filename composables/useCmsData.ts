export interface NewsArticle {
  title: string
  slug: string
  body: string
  cover?: string
  _publishedAt: string
}

export interface Profile {
  title: string
  slug: string
  avatar?: string
  role?: string
  roleType?: string
  generation?: string
  department?: string
  description?: string
}

export interface Sponsor {
  name: string
  slug: string
  logo?: string
  category?: string
  url?: string
}

export interface FaqItem {
  question: string
  answer: string
  slug: string
}

export interface CareerMentor {
  title: string
  slug: string
  avatar?: string
  role?: string
  generation?: string
  department?: string
  description?: string
}

export interface CareerScheduleItem {
  title: string
  slug: string
  startTime: string
  endTime: string
  description: string
}

export interface MainEventScheduleItem {
  title: string
  slug: string
  startTime: string
  endTime: string
  description: string
}

interface CmsData {
  news: any[]
  profiles: any[]
  sponsors: any[]
  faq: any[]
  donors: any[]
  other: any[]
  careerCrossroadsMentors: any[]
  careerCrossroadsSchedule: any[]
  mainEventSchedule: any[]
}

export const useCmsData = () => {
  const cmsData = useState<CmsData | null>('cmsData', () => null)
  const isLoading = useState('cmsDataLoading', () => false)

  const fetchData = async () => {
    if (cmsData.value) return
    if (isLoading.value) return

    isLoading.value = true
    try {
      const data = await $fetch<CmsData>('/data/cms-data.json')
      cmsData.value = data
    } catch (error) {
      console.error('Failed to fetch CMS data:', error)
      cmsData.value = { news: [], profiles: [], sponsors: [], faq: [], donors: [], other: [], careerCrossroadsMentors: [], careerCrossroadsSchedule: [], mainEventSchedule: [] }
    } finally {
      isLoading.value = false
    }
  }

  // Auto-fetch on first use
  if (!cmsData.value && !isLoading.value) {
    fetchData()
  }

  const news = computed<NewsArticle[]>(() => {
    if (!cmsData.value) return []
    return (cmsData.value.news || []).map((item: any) => ({
      title: item.title || '',
      slug: item.slug || '',
      body: item.body || '',
      cover: item.cover || '',
      _publishedAt: item._publishedAt || ''
    })).sort((a, b) => new Date(b._publishedAt).getTime() - new Date(a._publishedAt).getTime())
  })

  const profiles = computed<Profile[]>(() => {
    if (!cmsData.value) return []
    return (cmsData.value.profiles || []).map((item: any) => ({
      title: item.title || '',
      slug: item.slug || '',
      avatar: item.avatar || '',
      role: item.SA5o3FaJ || '',
      roleType: item.Ry9cHACp?.title || '',
      generation: item.ENXhgwbh?.[0]?.title || item.eZ9exVws || '',
      department: item.whYERZ_l?.[0]?.title || '',
      description: item.GEXEusGX || item.EYjLtiv0 || ''
    }))
  })

  const sponsors = computed<Sponsor[]>(() => {
    if (!cmsData.value) return []
    return (cmsData.value.sponsors || []).map((item: any) => ({
      name: item.name || '',
      slug: item.slug || '',
      logo: item.logo || item.Kr8gcaKy || '',
      category: item.lxykwQVY?.title || '',
      url: item.d4cBcZP0 || ''
    }))
  })

  const faq = computed<FaqItem[]>(() => {
    if (!cmsData.value) return []
    return (cmsData.value.faq || []).map((item: any) => ({
      question: item.question || '',
      answer: item.answer || '',
      slug: item.slug || ''
    }))
  })

  const careerMentors = computed<CareerMentor[]>(() => {
    if (!cmsData.value) return []
    return (cmsData.value.careerCrossroadsMentors || []).map((item: any) => ({
      title: item.title || '',
      slug: item.slug || '',
      avatar: item.avatar || '',
      role: item.role || '',
      generation: item.generation || '',
      department: item.department || '',
      description: item.description || ''
    }))
  })

  const careerSchedule = computed<CareerScheduleItem[]>(() => {
    if (!cmsData.value) return []
    return (cmsData.value.careerCrossroadsSchedule || []).map((item: any) => ({
      title: item.title || '',
      slug: item.slug || '',
      startTime: item.startTime || '',
      endTime: item.endTime || '',
      description: item.description || ''
    }))
  })

  const mainEventSchedule = computed<MainEventScheduleItem[]>(() => {
    if (!cmsData.value) return []
    return (cmsData.value.mainEventSchedule || []).map((item: any) => ({
      title: item.title || '',
      slug: item.slug || '',
      startTime: item.startTime || '',
      endTime: item.endTime || '',
      description: item.description || ''
    }))
  })

  const formatDate = (dateString: string): string => {
    if (!dateString) return ''
    const date = new Date(dateString)
    return `${date.getFullYear()}.${String(date.getMonth() + 1).padStart(2, '0')}.${String(date.getDate()).padStart(2, '0')}`
  }

  const stripHtml = (html: string): string => {
    return html.replace(/<[^>]*>/g, '').substring(0, 200)
  }

  return {
    news,
    profiles,
    sponsors,
    faq,
    careerMentors,
    careerSchedule,
    mainEventSchedule,
    formatDate,
    stripHtml,
    isLoading,
    fetchData
  }
}
