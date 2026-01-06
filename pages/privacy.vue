<template>
  <div>
    <!-- Header -->
    <section class="bg-gradient-to-br from-gray-700 to-gray-900 text-white py-16">
      <div class="max-w-6xl mx-auto px-4 text-center">
        <h1 class="text-4xl md:text-5xl font-bold mb-4">プライバシーポリシー</h1>
        <p class="text-xl text-gray-300">個人情報の取り扱いについて</p>
      </div>
    </section>

    <!-- Content -->
    <section class="py-16 md:py-24">
      <div class="max-w-4xl mx-auto px-4">
        <div class="prose prose-lg max-w-none">
          <div class="card mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-4">1. 個人情報の収集</h2>
            <p class="text-gray-600 leading-relaxed">
              本サイトでは、お問い合わせやイベント参加申し込みの際に、
              お名前、メールアドレス、電話番号等の個人情報を収集する場合があります。
            </p>
          </div>

          <div class="card mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-4">2. 個人情報の利用目的</h2>
            <p class="text-gray-600 leading-relaxed">
              収集した個人情報は、以下の目的で利用いたします。
            </p>
            <ul class="list-disc list-inside mt-4 text-gray-600 space-y-2">
              <li>大同窓会に関するご連絡</li>
              <li>お問い合わせへの回答</li>
              <li>イベント運営に必要な連絡</li>
            </ul>
          </div>

          <div class="card mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-4">3. 個人情報の第三者提供</h2>
            <p class="text-gray-600 leading-relaxed">
              収集した個人情報は、法令に基づく場合を除き、
              ご本人の同意なく第三者に提供することはありません。
            </p>
          </div>

          <div class="card mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-4">4. 個人情報の管理</h2>
            <p class="text-gray-600 leading-relaxed">
              収集した個人情報は、適切な安全管理措置を講じ、
              不正アクセス、紛失、破壊、改ざん等を防止いたします。
            </p>
          </div>

          <div class="card">
            <h2 class="text-2xl font-bold text-gray-900 mb-4">5. お問い合わせ</h2>
            <p class="text-gray-600 leading-relaxed">
              個人情報の取り扱いに関するお問い合わせは、
              開邦高校友会までご連絡ください。
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
  title: 'プライバシーポリシー | 開邦高校 大同窓会'
})
</script>
