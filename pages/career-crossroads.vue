<template>
  <div>
    <!-- Hero -->
    <section class="relative h-[50vh] min-h-[400px] flex items-center justify-center bg-neutral-900">
      <div class="absolute inset-0">
        <img
          src="/images/cover-image.png"
          alt="キャリア交差点"
          class="w-full h-full object-cover opacity-30"
        />
      </div>
      <div class="relative z-10 text-center text-white px-6">
        <p class="text-sm tracking-[0.3em] uppercase mb-4 text-white/60">Career Crossroads</p>
        <h1 class="text-4xl md:text-5xl lg:text-6xl font-light tracking-tight">
          キャリア交差点
        </h1>
      </div>
    </section>

    <!-- Content -->
    <section class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="grid lg:grid-cols-2 gap-16">
          <div>
            <p class="section-label">About</p>
            <h2 class="section-title mb-8">キャリア交差点とは</h2>
            <div class="divider mb-8"></div>
            <p class="text-neutral-600 leading-relaxed text-lg mb-6">
              開邦高校の卒業生たちが、様々な分野で活躍しています。
            </p>
            <p class="text-neutral-600 leading-relaxed">
              このコーナーでは、卒業生のキャリアストーリーや、
              在校生・若い卒業生へのメッセージをお届けします。
            </p>
          </div>

          <div>
            <p class="section-label">Diversity</p>
            <h2 class="section-title mb-8">多様な進路</h2>
            <div class="divider mb-8"></div>
            <p class="text-neutral-600 leading-relaxed mb-6">
              開邦高校の卒業生は、医療、法律、教育、IT、芸術など、
              多様な分野で活躍しています。
            </p>
            <p class="text-neutral-600 leading-relaxed">
              大同窓会では、様々なバックグラウンドを持つ卒業生が
              一堂に会し、交流する貴重な機会となります。
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- Networking -->
    <section class="py-24 md:py-32 bg-neutral-50">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center">
          <p class="section-label">Networking</p>
          <h2 class="section-title mb-8">ネットワーキング</h2>
          <p class="text-neutral-600 leading-relaxed text-lg">
            大同窓会は、卒業生同士のネットワークを広げる絶好の機会です。<br>
            是非この機会をご活用ください。
          </p>
        </div>
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
  title: 'キャリア交差点 | 開邦高校 大同窓会'
})
</script>
