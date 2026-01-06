// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },

  modules: [
    '@nuxtjs/tailwindcss'
  ],

  app: {
    head: {
      htmlAttrs: {
        lang: 'ja'
      },
      title: '沖縄県立 開邦高校 第三回 大同窓会',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: '開邦高等学校は昭和61年に開校し、2025年で創立４０周年を迎えます。この度、４０周年を記念して下記のとおり第三回大同窓会を開催する運びとなりました。' },
        { property: 'og:site_name', content: '沖縄県立 開邦高校 第三回 大同窓会' },
        { property: 'og:title', content: '沖縄県立 開邦高校 第三回 大同窓会' },
        { property: 'og:description', content: '開邦高等学校は昭和61年に開校し、2025年で創立４０周年を迎えます。この度、４０周年を記念して下記のとおり第三回大同窓会を開催する運びとなりました。' },
        { property: 'og:type', content: 'website' },
        { property: 'og:image', content: '/images/og-image.png' },
        { property: 'twitter:card', content: 'summary_large_image' },
        { property: 'twitter:image', content: '/images/og-image.png' }
      ],
      link: [
        { rel: 'icon', type: 'image/png', href: '/images/favicon.png' },
        { rel: 'apple-touch-icon', type: 'image/png', href: '/images/favicon.png' },
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }
      ]
    }
  },

  css: ['~/assets/css/main.css'],

  // GitHub Pages用の静的生成設定
  ssr: false,
  nitro: {
    preset: 'github-pages'
  }
})
