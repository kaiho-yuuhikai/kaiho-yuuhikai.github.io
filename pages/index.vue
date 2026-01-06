<template>
  <div>
    <!-- Hero Section -->
    <section class="relative h-screen flex items-center justify-center overflow-hidden">
      <!-- Background Image -->
      <div class="absolute inset-0">
        <img
          src="/images/cover-image.png"
          alt="開邦高校"
          class="w-full h-full object-cover"
        />
        <div class="absolute inset-0 bg-black/50"></div>
      </div>

      <!-- Content -->
      <div class="relative z-10 text-center text-white px-6 max-w-4xl">
        <p class="text-sm tracking-[0.3em] uppercase mb-6 text-kaiho-gold font-medium">
          今年で創立40周年！
        </p>
        <h1 class="text-4xl md:text-6xl lg:text-7xl font-light mb-6 tracking-tight">
          第3回 開邦高校大同窓会
        </h1>
        <p class="text-xl md:text-2xl font-light text-white/90 mb-12 max-w-2xl mx-auto leading-relaxed">
          懐かしい話も、未来の話も、しよう。
        </p>
        <div class="flex flex-col sm:flex-row gap-4 justify-center">
          <a href="#about" class="btn-white">
            詳細を見る
          </a>
        </div>
      </div>

      <!-- Scroll Indicator -->
      <div class="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
        <svg class="w-6 h-6 text-white/60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      </div>
    </section>

    <!-- About Section -->
    <section id="about" class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="grid lg:grid-cols-2 gap-16 items-center">
          <div>
            <p class="section-label">About</p>
            <h2 class="section-title mb-8">
              開邦高校について
            </h2>
            <div class="divider mb-8"></div>
            <p class="text-neutral-600 leading-relaxed text-lg mb-6">
              開邦高等学校は昭和61年に開校し、2025年で創立４０周年を迎えます。
            </p>
            <p class="text-neutral-600 leading-relaxed">
              この度、４０周年を記念して第三回大同窓会を開催する運びとなりました。
              久しぶりに恩師や旧友と再会し、学生時代の思い出に浸りながら、
              親睦を深める機会となれば幸いです。
            </p>
          </div>
          <div class="relative">
            <div class="aspect-[4/3] bg-neutral-100 rounded-sm overflow-hidden">
              <img
                src="/images/og-image.png"
                alt="大同窓会"
                class="w-full h-full object-cover"
              />
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Event Info Section -->
    <section class="py-24 md:py-32 bg-neutral-50">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center mb-16">
          <p class="section-label">Event</p>
          <h2 class="section-title">開催概要</h2>
        </div>

        <div class="max-w-2xl mx-auto">
          <div class="space-y-8">
            <div class="flex gap-8 items-start pb-8 border-b border-neutral-200">
              <span class="text-xs tracking-widest uppercase text-neutral-500 w-24 flex-shrink-0 pt-1">名称</span>
              <p class="text-neutral-900">開邦高校 第三回 大同窓会</p>
            </div>
            <div class="flex gap-8 items-start pb-8 border-b border-neutral-200">
              <span class="text-xs tracking-widest uppercase text-neutral-500 w-24 flex-shrink-0 pt-1">目的</span>
              <p class="text-neutral-900">創立40周年記念・同窓生の親睦</p>
            </div>
            <div class="flex gap-8 items-start pb-8 border-b border-neutral-200">
              <span class="text-xs tracking-widest uppercase text-neutral-500 w-24 flex-shrink-0 pt-1">対象</span>
              <p class="text-neutral-900">開邦高等学校 卒業生・教職員</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Message Section -->
    <section class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center">
          <p class="section-label">Message</p>
          <h2 class="section-title mb-12">ご挨拶</h2>
          <blockquote class="text-xl md:text-2xl font-light text-neutral-700 leading-relaxed mb-8">
            「久しぶりに恩師や旧友と再会し、<br class="hidden md:block">
            学生時代の思い出に浸りながら、<br class="hidden md:block">
            親睦を深める機会となれば幸いです。」
          </blockquote>
          <p class="text-sm tracking-wider text-neutral-500">
            是非お誘いあわせの上、ご参加ください。
          </p>
        </div>
      </div>
    </section>

    <!-- News Section -->
    <section class="py-24 md:py-32 bg-kaiho-green text-white">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center mb-16">
          <p class="text-xs font-medium tracking-widest uppercase text-white/60 mb-4">News</p>
          <h2 class="text-3xl md:text-4xl lg:text-5xl font-light tracking-tight text-white">お知らせ</h2>
        </div>

        <div class="max-w-2xl mx-auto space-y-6">
          <NuxtLink
            v-for="article in news.slice(0, 5)"
            :key="article.slug"
            :to="`/news/${article.slug}`"
            class="group flex gap-6 items-start pb-6 border-b border-white/20 block"
          >
            <time class="text-sm text-white/60 flex-shrink-0 w-24">{{ formatDate(article._publishedAt) }}</time>
            <div>
              <h3 class="text-white group-hover:text-white/80 transition-colors">
                {{ article.title }}
              </h3>
            </div>
          </NuxtLink>
        </div>

        <div class="text-center mt-12">
          <NuxtLink to="/news" class="inline-flex items-center gap-2 text-white/70 hover:text-white transition-colors text-sm tracking-wider">
            すべてのお知らせを見る
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </NuxtLink>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8 text-center">
        <p class="section-label">Join Us</p>
        <h2 class="section-title mb-8">
          皆様のご参加をお待ちしております
        </h2>
        <p class="text-neutral-600 mb-12 max-w-xl mx-auto">
          詳細情報やメニューについては、各ページをご確認ください。
        </p>
        <div class="flex flex-col sm:flex-row gap-4 justify-center">
          <NuxtLink to="/menu" class="btn-primary">
            メニューを見る
          </NuxtLink>
          <NuxtLink to="/career-crossroads" class="btn-outline">
            キャリア交差点
          </NuxtLink>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
const { news, formatDate, fetchData } = useCmsData()

onMounted(async () => {
  await fetchData()
})

useHead({
  title: '沖縄県立 開邦高校 第三回 大同窓会'
})
</script>
