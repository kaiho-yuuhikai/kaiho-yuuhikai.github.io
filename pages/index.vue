<template>
  <div>
    <!-- Hero Section -->
    <section class="relative bg-gradient-to-br from-primary-600 to-primary-800 text-white py-20 md:py-32">
      <div class="absolute inset-0 bg-black/20"></div>
      <div class="relative max-w-6xl mx-auto px-4 text-center">
        <p class="text-primary-200 text-lg mb-4">創立40周年記念</p>
        <h1 class="text-4xl md:text-6xl font-bold mb-6">
          沖縄県立 開邦高校<br class="md:hidden" />
          <span class="block mt-2">第三回 大同窓会</span>
        </h1>
        <p class="text-xl md:text-2xl text-primary-100 mb-8 max-w-3xl mx-auto leading-relaxed">
          久しぶりに恩師や旧友と再会し、<br class="hidden md:block" />
          学生時代の思い出に浸りながら、親睦を深める機会
        </p>
        <div class="flex flex-col sm:flex-row gap-4 justify-center">
          <a href="#event-info" class="btn-primary text-lg">
            開催概要を見る
          </a>
          <NuxtLink to="/menu" class="btn-secondary bg-transparent border-white text-white hover:bg-white/10 text-lg">
            メニューを見る
          </NuxtLink>
        </div>
      </div>
    </section>

    <!-- About Section -->
    <section class="py-16 md:py-24 bg-gray-50">
      <div class="max-w-6xl mx-auto px-4">
        <h2 class="section-title">開邦高校について</h2>
        <div class="max-w-3xl mx-auto text-center">
          <p class="text-lg text-gray-600 leading-relaxed">
            開邦高等学校は昭和61年に開校し、2025年で創立４０周年を迎えます。
            この度、４０周年を記念して第三回大同窓会を開催する運びとなりました。
          </p>
        </div>
      </div>
    </section>

    <!-- Event Info Section -->
    <section id="event-info" class="py-16 md:py-24">
      <div class="max-w-6xl mx-auto px-4">
        <h2 class="section-title">開催概要</h2>
        <div class="max-w-2xl mx-auto">
          <div class="card">
            <dl class="space-y-6">
              <div class="flex flex-col sm:flex-row sm:gap-4">
                <dt class="font-semibold text-gray-900 sm:w-32 flex-shrink-0">イベント名</dt>
                <dd class="text-gray-600 mt-1 sm:mt-0">開邦高校 第三回 大同窓会</dd>
              </div>
              <div class="flex flex-col sm:flex-row sm:gap-4">
                <dt class="font-semibold text-gray-900 sm:w-32 flex-shrink-0">目的</dt>
                <dd class="text-gray-600 mt-1 sm:mt-0">創立40周年記念・同窓生の親睦</dd>
              </div>
              <div class="flex flex-col sm:flex-row sm:gap-4">
                <dt class="font-semibold text-gray-900 sm:w-32 flex-shrink-0">対象</dt>
                <dd class="text-gray-600 mt-1 sm:mt-0">開邦高等学校 卒業生・教職員</dd>
              </div>
            </dl>
          </div>
        </div>
      </div>
    </section>

    <!-- Message Section -->
    <section class="py-16 md:py-24 bg-primary-50">
      <div class="max-w-6xl mx-auto px-4">
        <h2 class="section-title">ご挨拶</h2>
        <div class="max-w-3xl mx-auto">
          <div class="bg-white rounded-2xl shadow-lg p-8 md:p-12">
            <p class="text-gray-600 leading-relaxed text-lg">
              久しぶりに恩師や旧友と再会し、学生時代の思い出に浸りながら、
              親睦を深める機会となれば幸いです。
              是非お誘いあわせの上、ご参加ください。
            </p>
            <p class="mt-6 text-right text-gray-500">
              開邦高校友会
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="py-16 md:py-24 bg-gradient-to-r from-primary-600 to-primary-700 text-white">
      <div class="max-w-6xl mx-auto px-4 text-center">
        <h2 class="text-3xl md:text-4xl font-bold mb-6">
          皆様のご参加をお待ちしております
        </h2>
        <p class="text-xl text-primary-100 mb-8 max-w-2xl mx-auto">
          詳細情報やメニューについては、各ページをご確認ください。
        </p>
        <div class="flex flex-col sm:flex-row gap-4 justify-center">
          <NuxtLink to="/menu" class="btn-primary bg-white text-primary-700 hover:bg-gray-100">
            メニューを確認
          </NuxtLink>
          <NuxtLink to="/career-crossroads" class="btn-secondary border-white text-white hover:bg-white/10">
            キャリア交差点
          </NuxtLink>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
useHead({
  title: '沖縄県立 開邦高校 第三回 大同窓会'
})
</script>
