<template>
  <div>
    <!-- Header -->
    <section class="bg-gradient-to-br from-primary-600 to-primary-800 text-white py-16">
      <div class="max-w-6xl mx-auto px-4 text-center">
        <h1 class="text-4xl md:text-5xl font-bold mb-4">メニュー</h1>
        <p class="text-xl text-primary-100">大同窓会の当日メニュー</p>
      </div>
    </section>

    <!-- Menu Content -->
    <section class="py-16 md:py-24">
      <div class="max-w-6xl mx-auto px-4">
        <div class="max-w-3xl mx-auto">
          <div class="card mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">当日のお料理</h2>
            <p class="text-gray-600 leading-relaxed">
              大同窓会では、沖縄の伝統料理や地元の食材を使った
              特別メニューをご用意しております。
            </p>
            <p class="text-gray-600 leading-relaxed mt-4">
              懐かしい味わいとともに、旧友との再会をお楽しみください。
            </p>
          </div>

          <div class="card mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">ドリンク</h2>
            <p class="text-gray-600 leading-relaxed">
              ソフトドリンク、アルコール各種をご用意しております。
            </p>
          </div>

          <div class="bg-primary-50 rounded-xl p-6 text-center">
            <p class="text-gray-600">
              詳細なメニューは当日会場にてご確認ください。
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- Back to Home -->
    <section class="py-8 bg-gray-50">
      <div class="max-w-6xl mx-auto px-4 text-center">
        <NuxtLink to="/" class="text-primary-600 hover:text-primary-700 font-medium">
          &larr; トップページに戻る
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
