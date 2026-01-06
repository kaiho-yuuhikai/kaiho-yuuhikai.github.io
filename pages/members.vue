<template>
  <div>
    <!-- Hero -->
    <section class="relative h-[40vh] min-h-[300px] flex items-center justify-center bg-kaiho-green">
      <div class="relative z-10 text-center text-white px-6">
        <p class="text-sm tracking-[0.3em] uppercase mb-4 text-white/60">Members</p>
        <h1 class="text-4xl md:text-5xl lg:text-6xl font-light tracking-tight">
          登壇者・メンター
        </h1>
      </div>
    </section>

    <!-- Content -->
    <section class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center mb-16">
          <p class="section-label">Speakers & Mentors</p>
          <h2 class="section-title mb-8">第三回大同窓会 登壇者</h2>
          <div class="divider mx-auto mb-8"></div>
          <p class="text-neutral-600 leading-relaxed">
            大同窓会では、様々な分野で活躍する卒業生の皆様に<br>
            ご登壇いただきました。
          </p>
        </div>

        <!-- Members Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div
            v-for="member in profiles"
            :key="member.slug"
            class="bg-white border border-neutral-100 rounded-lg overflow-hidden hover:shadow-lg transition-shadow"
          >
            <div v-if="member.avatar" class="aspect-square bg-neutral-100">
              <img
                :src="member.avatar"
                :alt="member.title"
                class="w-full h-full object-cover"
              />
            </div>
            <div class="p-6">
              <p v-if="member.generation" class="text-xs text-kaiho-green font-medium mb-1">
                {{ member.generation }}
              </p>
              <h3 class="text-lg font-medium text-neutral-900 mb-2">
                {{ member.title }}
              </h3>
              <p v-if="member.role" class="text-sm text-neutral-600 mb-3">
                {{ member.role }}
              </p>
              <p v-if="member.description" class="text-sm text-neutral-500 line-clamp-3">
                {{ member.description }}
              </p>
            </div>
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
const { profiles, fetchData } = useCmsData()

onMounted(async () => {
  await fetchData()
})

useHead({
  title: '登壇者・メンター | 開邦高校 大同窓会'
})
</script>
