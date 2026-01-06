<template>
  <div>
    <!-- Hero -->
    <section class="relative h-[40vh] min-h-[300px] flex items-center justify-center bg-kaiho-green">
      <div class="relative z-10 text-center text-white px-6">
        <p class="text-sm tracking-[0.3em] uppercase mb-4 text-white/60">News</p>
        <h1 class="text-4xl md:text-5xl lg:text-6xl font-light tracking-tight">
          お知らせ
        </h1>
      </div>
    </section>

    <!-- News List -->
    <section class="py-24 md:py-32">
      <div class="max-w-4xl mx-auto px-6 lg:px-8">
        <div class="space-y-8">
          <NuxtLink
            v-for="article in news"
            :key="article.slug"
            :to="`/news/${article.slug}`"
            class="group block border-b border-neutral-200 pb-8"
          >
            <div class="flex flex-col md:flex-row gap-6">
              <div v-if="article.cover" class="md:w-48 flex-shrink-0">
                <img
                  :src="article.cover"
                  :alt="article.title"
                  class="w-full h-32 md:h-28 object-cover rounded"
                />
              </div>
              <div class="flex-1">
                <time class="text-sm text-neutral-500 mb-2 block">{{ formatDate(article._publishedAt) }}</time>
                <h2 class="text-xl font-medium text-neutral-900 group-hover:text-kaiho-green transition-colors mb-3">
                  {{ article.title }}
                </h2>
                <p class="text-neutral-600 text-sm line-clamp-2">
                  {{ stripHtml(article.body) }}
                </p>
              </div>
            </div>
          </NuxtLink>
        </div>
      </div>
    </section>

    <!-- Back -->
    <section class="pb-24">
      <div class="max-w-7xl mx-auto px-6 lg:px-8 text-center">
        <NuxtLink to="/" class="inline-flex items-center gap-2 text-neutral-500 hover:text-neutral-900 transition-colors text-sm tracking-wider">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          トップページに戻る
        </NuxtLink>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
const { news, formatDate, stripHtml } = useCmsData()

useHead({
  title: 'お知らせ | 開邦高校 大同窓会'
})
</script>
