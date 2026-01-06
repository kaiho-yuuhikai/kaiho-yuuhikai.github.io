<template>
  <div v-if="article">
    <!-- Hero -->
    <section class="relative h-[50vh] min-h-[400px] flex items-center justify-center bg-neutral-900">
      <div v-if="article.cover" class="absolute inset-0">
        <img
          :src="article.cover"
          :alt="article.title"
          class="w-full h-full object-cover opacity-40"
        />
      </div>
      <div class="relative z-10 text-center text-white px-6 max-w-4xl">
        <time class="text-sm tracking-[0.2em] uppercase mb-4 text-white/60 block">
          {{ formatDate(article._publishedAt) }}
        </time>
        <h1 class="text-3xl md:text-4xl lg:text-5xl font-light tracking-tight">
          {{ article.title }}
        </h1>
      </div>
    </section>

    <!-- Content -->
    <section class="py-16 md:py-24">
      <div class="max-w-3xl mx-auto px-6 lg:px-8">
        <article class="prose prose-lg max-w-none prose-headings:font-normal prose-a:text-kaiho-green" v-html="article.body"></article>
      </div>
    </section>

    <!-- Back -->
    <section class="pb-24">
      <div class="max-w-7xl mx-auto px-6 lg:px-8 text-center">
        <NuxtLink to="/news" class="inline-flex items-center gap-2 text-neutral-500 hover:text-neutral-900 transition-colors text-sm tracking-wider">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          お知らせ一覧に戻る
        </NuxtLink>
      </div>
    </section>
  </div>
  <div v-else class="min-h-screen flex items-center justify-center">
    <p class="text-neutral-500">記事が見つかりませんでした</p>
  </div>
</template>

<script setup lang="ts">
const route = useRoute()
const { news, formatDate } = useCmsData()

const article = computed(() => {
  return news.value.find(n => n.slug === route.params.slug)
})

useHead({
  title: computed(() => article.value ? `${article.value.title} | 開邦高校 大同窓会` : 'お知らせ | 開邦高校 大同窓会')
})
</script>

<style>
.prose img {
  @apply rounded-lg my-8 w-full;
}
.prose figure {
  @apply my-8;
}
.prose figcaption {
  @apply text-center text-sm text-neutral-500 mt-2;
}
.prose hr {
  @apply my-12 border-neutral-200;
}
.prose h2, .prose h3 {
  @apply mt-12 mb-4;
}
.prose table {
  @apply w-full text-sm;
}
.prose th, .prose td {
  @apply border border-neutral-200 p-3;
}
.prose th {
  @apply bg-neutral-50;
}
</style>
