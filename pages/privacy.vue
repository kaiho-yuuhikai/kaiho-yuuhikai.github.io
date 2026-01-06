<template>
  <div>
    <!-- Hero -->
    <section class="relative h-[40vh] min-h-[300px] flex items-center justify-center bg-neutral-900">
      <div class="relative z-10 text-center text-white px-6">
        <p class="text-sm tracking-[0.3em] uppercase mb-4 text-white/60">Privacy Policy</p>
        <h1 class="text-4xl md:text-5xl lg:text-6xl font-light tracking-tight">
          プライバシーポリシー
        </h1>
      </div>
    </section>

    <!-- Content -->
    <section class="py-24 md:py-32">
      <div class="max-w-3xl mx-auto px-6 lg:px-8">
        <div class="space-y-16">
          <div>
            <h2 class="text-xl font-medium text-neutral-900 mb-6">1. 個人情報の収集</h2>
            <p class="text-neutral-600 leading-relaxed">
              本サイトでは、お問い合わせやイベント参加申し込みの際に、
              お名前、メールアドレス、電話番号等の個人情報を収集する場合があります。
            </p>
          </div>

          <div>
            <h2 class="text-xl font-medium text-neutral-900 mb-6">2. 個人情報の利用目的</h2>
            <p class="text-neutral-600 leading-relaxed mb-4">
              収集した個人情報は、以下の目的で利用いたします。
            </p>
            <ul class="space-y-2 text-neutral-600">
              <li class="flex items-start gap-3">
                <span class="w-1 h-1 bg-neutral-400 rounded-full mt-2.5 flex-shrink-0"></span>
                大同窓会に関するご連絡
              </li>
              <li class="flex items-start gap-3">
                <span class="w-1 h-1 bg-neutral-400 rounded-full mt-2.5 flex-shrink-0"></span>
                お問い合わせへの回答
              </li>
              <li class="flex items-start gap-3">
                <span class="w-1 h-1 bg-neutral-400 rounded-full mt-2.5 flex-shrink-0"></span>
                イベント運営に必要な連絡
              </li>
            </ul>
          </div>

          <div>
            <h2 class="text-xl font-medium text-neutral-900 mb-6">3. 個人情報の第三者提供</h2>
            <p class="text-neutral-600 leading-relaxed">
              収集した個人情報は、法令に基づく場合を除き、
              ご本人の同意なく第三者に提供することはありません。
            </p>
          </div>

          <div>
            <h2 class="text-xl font-medium text-neutral-900 mb-6">4. 個人情報の管理</h2>
            <p class="text-neutral-600 leading-relaxed">
              収集した個人情報は、適切な安全管理措置を講じ、
              不正アクセス、紛失、破壊、改ざん等を防止いたします。
            </p>
          </div>

          <div>
            <h2 class="text-xl font-medium text-neutral-900 mb-6">5. お問い合わせ</h2>
            <p class="text-neutral-600 leading-relaxed">
              個人情報の取り扱いに関するお問い合わせは、
              開邦高校友会までご連絡ください。
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
  title: 'プライバシーポリシー | 開邦高校 大同窓会'
})
</script>
