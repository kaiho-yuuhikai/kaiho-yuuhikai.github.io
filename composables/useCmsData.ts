import cmsData from '~/content/cms-data.json'

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
  generation?: string
  description?: string
}

export interface Sponsor {
  name: string
  slug: string
  logo?: string
}

export interface FaqItem {
  question: string
  answer: string
  slug: string
}

export const useCmsData = () => {
  const news = computed<NewsArticle[]>(() => {
    return (cmsData.news || []).map((item: any) => ({
      title: item.title || '',
      slug: item.slug || '',
      body: item.body || '',
      cover: item.cover || '',
      _publishedAt: item._publishedAt || ''
    })).sort((a, b) => new Date(b._publishedAt).getTime() - new Date(a._publishedAt).getTime())
  })

  const profiles = computed<Profile[]>(() => {
    return (cmsData.profiles || []).map((item: any) => ({
      title: item.title || '',
      slug: item.slug || '',
      avatar: item.avatar || '',
      role: item.SA5o3FaJ || '',
      generation: item.ENXhgwbh?.[0]?.title || item.eZ9exVws || '',
      description: item.GEXEusGX || item.EYjLtiv0 || ''
    }))
  })

  const sponsors = computed<Sponsor[]>(() => {
    return (cmsData.sponsors || []).map((item: any) => ({
      name: item.name || '',
      slug: item.slug || '',
      logo: item.logo || ''
    }))
  })

  const faq = computed<FaqItem[]>(() => {
    return (cmsData.faq || []).map((item: any) => ({
      question: item.question || '',
      answer: item.answer || '',
      slug: item.slug || ''
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
    formatDate,
    stripHtml
  }
}
