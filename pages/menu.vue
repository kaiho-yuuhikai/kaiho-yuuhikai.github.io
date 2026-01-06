<template>
  <div>
    <!-- Hero -->
    <section class="relative h-[50vh] min-h-[400px] flex items-center justify-center bg-neutral-900">
      <div class="absolute inset-0">
        <img
          src="/images/cover-image.png"
          alt="メニュー"
          class="w-full h-full object-cover opacity-30"
        />
      </div>
      <div class="relative z-10 text-center text-white px-6">
        <p class="text-sm tracking-[0.3em] uppercase mb-4 text-white/60">Menu</p>
        <h1 class="text-4xl md:text-5xl lg:text-6xl font-light tracking-tight">
          メニュー
        </h1>
      </div>
    </section>

    <!-- Content -->
    <section class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto">
          <div class="mb-16">
            <p class="section-label">Food</p>
            <h2 class="section-title mb-8">当日のお料理</h2>
            <div class="divider mb-8"></div>
            <p class="text-neutral-600 leading-relaxed text-lg">
              大同窓会では、沖縄の伝統料理や地元の食材を使った
              特別メニューをご用意しております。
              懐かしい味わいとともに、旧友との再会をお楽しみください。
            </p>
          </div>

          <div class="mb-16">
            <p class="section-label">Drinks</p>
            <h2 class="section-title mb-8">ドリンク</h2>
            <div class="divider mb-8"></div>
            <p class="text-neutral-600 leading-relaxed">
              ソフトドリンク、アルコール各種をご用意しております。
            </p>
          </div>

          <div class="bg-neutral-50 p-8 md:p-12 text-center">
            <p class="text-neutral-500 text-sm tracking-wider">
              詳細なメニューは当日会場にてご確認ください。
            </p>
          </div>
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
useHead({
  title: 'メニュー | 開邦高校 大同窓会'
})
</script>
