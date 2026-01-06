<template>
  <div>
    <!-- Header -->
    <section class="bg-gradient-to-br from-primary-600 to-primary-800 text-white py-16">
      <div class="max-w-6xl mx-auto px-4 text-center">
        <h1 class="text-4xl md:text-5xl font-bold mb-4">キャリア交差点</h1>
        <p class="text-xl text-primary-100">卒業生のキャリアストーリー</p>
      </div>
    </section>

    <!-- Content -->
    <section class="py-16 md:py-24">
      <div class="max-w-6xl mx-auto px-4">
        <div class="max-w-3xl mx-auto">
          <div class="card mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">キャリア交差点とは</h2>
            <p class="text-gray-600 leading-relaxed">
              開邦高校の卒業生たちが、様々な分野で活躍しています。
              このコーナーでは、卒業生のキャリアストーリーや、
              在校生・若い卒業生へのメッセージをお届けします。
            </p>
          </div>

          <div class="card mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">多様な進路</h2>
            <p class="text-gray-600 leading-relaxed">
              開邦高校の卒業生は、医療、法律、教育、IT、芸術など、
              多様な分野で活躍しています。
            </p>
            <p class="text-gray-600 leading-relaxed mt-4">
              大同窓会では、様々なバックグラウンドを持つ卒業生が
              一堂に会し、交流する貴重な機会となります。
            </p>
          </div>

          <div class="card">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">ネットワーキング</h2>
            <p class="text-gray-600 leading-relaxed">
              大同窓会は、卒業生同士のネットワークを広げる
              絶好の機会です。是非この機会をご活用ください。
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
  title: 'キャリア交差点 | 開邦高校 大同窓会'
})
</script>
