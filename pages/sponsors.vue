<template>
  <div>
    <!-- Hero -->
    <section class="relative h-[40vh] min-h-[300px] flex items-center justify-center bg-kaiho-green">
      <div class="relative z-10 text-center text-white px-6">
        <p class="text-sm tracking-[0.3em] uppercase mb-4 text-white/60">Sponsors</p>
        <h1 class="text-4xl md:text-5xl lg:text-6xl font-light tracking-tight">
          協賛企業・団体
        </h1>
      </div>
    </section>

    <!-- Content -->
    <section class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center mb-16">
          <p class="section-label">Thank You</p>
          <h2 class="section-title mb-8">ご協賛いただいた皆様</h2>
          <div class="divider mx-auto mb-8"></div>
          <p class="text-neutral-600 leading-relaxed">
            第三回 開邦高校大同窓会の開催にあたり、<br>
            多くの企業・団体・個人の皆様からご協賛をいただきました。<br>
            心より御礼申し上げます。
          </p>
        </div>

        <!-- Sponsors Grid -->
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          <component
            :is="sponsor.url ? 'a' : 'div'"
            v-for="sponsor in sponsors"
            :key="sponsor.slug"
            :href="sponsor.url || undefined"
            :target="sponsor.url ? '_blank' : undefined"
            :rel="sponsor.url ? 'noopener noreferrer' : undefined"
            class="bg-white border border-neutral-100 rounded-lg p-6 flex flex-col items-center justify-center min-h-[160px] hover:shadow-md transition-shadow"
            :class="{ 'cursor-pointer': sponsor.url }"
          >
            <div v-if="sponsor.logo" class="w-full flex-1 flex items-center justify-center mb-3">
              <img
                :src="sponsor.logo"
                :alt="sponsor.name"
                class="w-full h-16 object-contain"
              />
            </div>
            <p class="text-neutral-700 text-center text-sm font-medium" :class="{ 'flex-1 flex items-center': !sponsor.logo }">
              {{ sponsor.name }}
            </p>
            <p v-if="sponsor.url" class="text-xs text-kaiho-green mt-2 flex items-center gap-1">
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
              Website
            </p>
          </component>
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
const { sponsors, fetchData } = useCmsData()

onMounted(async () => {
  await fetchData()
})

useHead({
  title: '協賛企業・団体 | 開邦高校 大同窓会'
})
</script>
