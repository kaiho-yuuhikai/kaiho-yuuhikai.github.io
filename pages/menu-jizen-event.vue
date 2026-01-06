<template>
  <div>
    <!-- Hero -->
    <section class="relative h-[50vh] min-h-[400px] flex items-center justify-center bg-neutral-900">
      <div class="absolute inset-0">
        <img
          src="/images/cover-image.png"
          alt="事前イベント"
          class="w-full h-full object-cover opacity-30"
        />
      </div>
      <div class="relative z-10 text-center text-white px-6">
        <p class="text-sm tracking-[0.3em] uppercase mb-4 text-white/60">Pre-Event</p>
        <h1 class="text-4xl md:text-5xl lg:text-6xl font-light tracking-tight">
          事前イベント
        </h1>
      </div>
    </section>

    <!-- Content -->
    <section class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center">
          <p class="section-label">Information</p>
          <h2 class="section-title mb-8">事前イベントについて</h2>
          <div class="divider mx-auto mb-8"></div>
          <p class="text-neutral-600 leading-relaxed text-lg mb-8">
            大同窓会の開催に先立ち、卒業生の皆様に
            お楽しみいただける事前イベントを企画しております。
          </p>
          <p class="text-neutral-600 leading-relaxed">
            詳細は決まり次第、こちらのページでお知らせいたします。
          </p>
        </div>
      </div>
    </section>

    <!-- Coming Soon -->
    <section class="py-24 md:py-32 bg-neutral-50">
      <div class="max-w-7xl mx-auto px-6 lg:px-8 text-center">
        <p class="text-sm tracking-[0.2em] uppercase text-neutral-400">
          Coming Soon
        </p>
      </div>
    </section>

    <!-- Back -->
    <section class="py-24">
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
useHead({
  title: '事前イベント | 開邦高校 大同窓会'
})
</script>
