<template>
  <div>
    <!-- Hero Section -->
    <section class="relative h-screen flex items-center justify-center overflow-hidden">
      <!-- Background Image -->
      <div class="absolute inset-0">
        <img
          src="/images/cover-image.png"
          alt="開邦高校"
          class="w-full h-full object-cover"
        />
        <div class="absolute inset-0 bg-black/50"></div>
      </div>

      <!-- Content -->
      <div class="relative z-10 text-center text-white px-6 max-w-4xl">
        <p class="text-sm tracking-[0.3em] uppercase mb-6 text-kaiho-gold font-medium">
          今年で創立40周年！
        </p>
        <h1 class="text-4xl md:text-6xl lg:text-7xl font-light mb-6 tracking-tight">
          第3回 開邦高校大同窓会
        </h1>
        <p class="text-xl md:text-2xl font-light text-white/90 mb-12 max-w-2xl mx-auto leading-relaxed">
          懐かしい話も、未来の話も、しよう。
        </p>
        <div class="flex flex-col sm:flex-row gap-4 justify-center">
          <a href="#about" class="btn-white">
            詳細を見る
          </a>
        </div>
      </div>

      <!-- Scroll Indicator -->
      <div class="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
        <svg class="w-6 h-6 text-white/60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      </div>
    </section>

    <!-- About Section -->
    <section id="about" class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="grid lg:grid-cols-2 gap-16 items-center">
          <div>
            <p class="section-label">About</p>
            <h2 class="section-title mb-8">
              開邦高校について
            </h2>
            <div class="divider mb-8"></div>
            <p class="text-neutral-600 leading-relaxed text-lg mb-6">
              開邦高等学校は昭和61年に開校し、2025年で創立４０周年を迎えます。
            </p>
            <p class="text-neutral-600 leading-relaxed">
              この度、４０周年を記念して第三回大同窓会を開催する運びとなりました。
              久しぶりに恩師や旧友と再会し、学生時代の思い出に浸りながら、
              親睦を深める機会となれば幸いです。
            </p>
          </div>
          <div class="relative">
            <div class="aspect-[4/3] bg-neutral-100 rounded-sm overflow-hidden">
              <img
                src="/images/og-image.png"
                alt="大同窓会"
                class="w-full h-full object-cover"
              />
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Event Info Section -->
    <section class="py-24 md:py-32 bg-neutral-50">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center mb-16">
          <p class="section-label">Event</p>
          <h2 class="section-title">開催概要</h2>
        </div>

        <div class="max-w-2xl mx-auto">
          <div class="space-y-8">
            <div class="flex gap-8 items-start pb-8 border-b border-neutral-200">
              <span class="text-xs tracking-widest uppercase text-neutral-500 w-24 flex-shrink-0 pt-1">名称</span>
              <p class="text-neutral-900">開邦高校 第三回 大同窓会</p>
            </div>
            <div class="flex gap-8 items-start pb-8 border-b border-neutral-200">
              <span class="text-xs tracking-widest uppercase text-neutral-500 w-24 flex-shrink-0 pt-1">日時</span>
              <div>
                <p class="text-neutral-900">2025年12月28日（日）</p>
                <p class="text-neutral-500 text-sm mt-1">開場 14:30 ／ 開演 15:00〜17:30</p>
              </div>
            </div>
            <div class="flex gap-8 items-start pb-8 border-b border-neutral-200">
              <span class="text-xs tracking-widest uppercase text-neutral-500 w-24 flex-shrink-0 pt-1">会場</span>
              <div>
                <p class="text-neutral-900">パシフィックホテル沖縄</p>
                <p class="text-neutral-500 text-sm mt-1">万座の間＋珊瑚の間</p>
                <p class="text-neutral-500 text-sm">〒900-0036 沖縄県那覇市西3-6-1</p>
              </div>
            </div>
            <div class="flex gap-8 items-start pb-8 border-b border-neutral-200">
              <span class="text-xs tracking-widest uppercase text-neutral-500 w-24 flex-shrink-0 pt-1">参加費</span>
              <div>
                <p class="text-neutral-900 font-medium mb-2">前売り（12月22日まで）</p>
                <p class="text-neutral-900">1期〜30期生：5,500円</p>
                <p class="text-neutral-900">31期〜37期生：3,000円</p>
                <p class="text-neutral-900 font-medium mt-4 mb-2">当日</p>
                <p class="text-neutral-900">1期〜30期生：7,000円</p>
                <p class="text-neutral-900">31期〜37期生：4,000円</p>
              </div>
            </div>
            <div class="flex gap-8 items-start pb-8 border-b border-neutral-200">
              <span class="text-xs tracking-widest uppercase text-neutral-500 w-24 flex-shrink-0 pt-1">申込</span>
              <div>
                <a href="https://peatix.com/event/4577195/view" target="_blank" class="text-kaiho-green hover:underline">
                  Peatixでお申し込み →
                </a>
              </div>
            </div>
            <div class="flex gap-8 items-start pb-8 border-b border-neutral-200">
              <span class="text-xs tracking-widest uppercase text-neutral-500 w-24 flex-shrink-0 pt-1">目的</span>
              <p class="text-neutral-900">創立40周年記念・同窓生の親睦</p>
            </div>
            <div class="flex gap-8 items-start pb-8 border-b border-neutral-200">
              <span class="text-xs tracking-widest uppercase text-neutral-500 w-24 flex-shrink-0 pt-1">対象</span>
              <p class="text-neutral-900">開邦高等学校 卒業生・教職員</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Timeline Section -->
    <section class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center mb-16">
          <p class="section-label">Timeline</p>
          <h2 class="section-title">タイムライン</h2>
        </div>

        <div class="max-w-4xl mx-auto">
          <!-- Pre-Event Summary -->
          <div class="bg-kaiho-green/10 rounded-lg p-6 mb-8">
            <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div>
                <span class="bg-kaiho-green text-white text-xs px-3 py-1 rounded-full">事前イベント</span>
                <p class="text-xl font-medium text-neutral-900 mt-3">13:00 - 14:30　開邦キャリア・クロスロード</p>
                <p class="text-neutral-600 text-sm mt-1">40名のメンターと学生・若手が交わるトークセッション</p>
              </div>
              <NuxtLink to="/career-crossroads" class="text-kaiho-green hover:underline text-sm font-medium whitespace-nowrap">
                詳細を見る →
              </NuxtLink>
            </div>
          </div>

          <!-- Main Event Schedule -->
          <div class="bg-kaiho-gold/10 rounded-lg p-6">
            <div class="flex items-center gap-3 mb-6">
              <span class="bg-kaiho-gold text-white text-xs px-3 py-1 rounded-full">本編</span>
              <p class="text-lg font-medium text-neutral-900">15:00 - 17:30（開場 14:30）</p>
            </div>

            <div class="space-y-4">
              <div
                v-for="(item, index) in mainEventSchedule"
                :key="item.slug"
                class="flex gap-4 pb-4"
                :class="{ 'border-b border-neutral-200': index < mainEventSchedule.length - 1 }"
              >
                <div class="flex-shrink-0 w-28 text-sm font-medium text-kaiho-gold">
                  {{ item.startTime }} - {{ item.endTime }}
                </div>
                <div class="flex-1">
                  <h4 class="text-neutral-900 font-medium">{{ item.title }}</h4>
                  <p v-if="item.description" class="text-neutral-600 text-sm mt-1">
                    {{ item.description }}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Message Section -->
    <section class="py-24 md:py-32 bg-neutral-50">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto">
          <div class="text-center mb-12">
            <p class="section-label">Message</p>
            <h2 class="section-title">実行委員長 挨拶</h2>
          </div>
          <div class="bg-white rounded-lg p-8 md:p-12 shadow-sm">
            <div class="md:flex md:gap-8 mb-8">
              <div class="flex-shrink-0 mb-6 md:mb-0">
                <img
                  src="https://storage.googleapis.com/studio-cms-assets/projects/BVqXkelzOR/s-300x300_webp_c662db93-a50f-4164-baa1-865c34f72919.png"
                  alt="宮城 潤"
                  class="w-32 h-32 md:w-40 md:h-40 rounded-full object-cover mx-auto md:mx-0"
                />
              </div>
              <div>
                <p class="text-neutral-700 leading-relaxed mb-6">
                  このたび、私どもの母校・開邦高等学校の創立40周年を記念し、第3回大同窓会を開催いたします。
                </p>
                <p class="text-neutral-700 leading-relaxed mb-6">
                  本校は、校訓「開邦雄飛」（邦を開き、世界に羽ばたく人材の育成）の理念のもと、知の探究と豊かな表現を育んできました。これまでに37期の卒業生を輩出し、各分野で活躍する同窓生の姿は、私たちの誇りであり大きな励みです。
                </p>
              </div>
            </div>
            <p class="text-neutral-700 leading-relaxed mb-6">
              本会が、同窓生が世代や活動分野を超えてつながり、楽しく語り合える場となることを願っています。あわせて、在校生の教育環境充実のためのご協賛・ご寄付も募っております。温かなご支援に心より感謝申し上げます。
            </p>
            <p class="text-neutral-700 leading-relaxed mb-8">
              母校への誇りを確かめ、後輩たちの励みとなり、そして何よりも同窓生一人ひとりが明日への活力を得る場となるよう、旧交を温め、開邦らしい対話と笑顔に満ちた一日にいたしましょう。どうぞご参加のうえ、大いにお楽しみください。
            </p>
            <div class="flex items-center justify-end gap-4">
              <div class="text-right text-neutral-600">
                <p class="text-sm">第3回開邦高校大同窓会実行委員会</p>
                <p class="font-medium">実行委員長　宮城　潤</p>
                <p class="text-xs text-neutral-500 mt-1">3期・芸術科</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- News Section -->
    <section class="py-24 md:py-32 bg-kaiho-green text-white">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center mb-16">
          <p class="text-xs font-medium tracking-widest uppercase text-white/60 mb-4">News</p>
          <h2 class="text-3xl md:text-4xl lg:text-5xl font-light tracking-tight text-white">お知らせ</h2>
        </div>

        <div class="max-w-2xl mx-auto space-y-6">
          <NuxtLink
            v-for="article in news.slice(0, 5)"
            :key="article.slug"
            :to="`/news/${article.slug}`"
            class="group flex gap-6 items-start pb-6 border-b border-white/20 block"
          >
            <time class="text-sm text-white/60 flex-shrink-0 w-24">{{ formatDate(article._publishedAt) }}</time>
            <div>
              <h3 class="text-white group-hover:text-white/80 transition-colors">
                {{ article.title }}
              </h3>
            </div>
          </NuxtLink>
        </div>

        <div class="text-center mt-12">
          <NuxtLink to="/news" class="inline-flex items-center gap-2 text-white/70 hover:text-white transition-colors text-sm tracking-wider">
            すべてのお知らせを見る
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </NuxtLink>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8 text-center">
        <p class="section-label">Join Us</p>
        <h2 class="section-title mb-8">
          皆様のご参加をお待ちしております
        </h2>
        <p class="text-neutral-600 mb-12 max-w-xl mx-auto">
          詳細情報やメニューについては、各ページをご確認ください。
        </p>
        <div class="flex flex-col sm:flex-row gap-4 justify-center flex-wrap">
          <NuxtLink to="/menu" class="btn-primary">
            メニューを見る
          </NuxtLink>
          <NuxtLink to="/members" class="btn-outline">
            登壇者・メンター
          </NuxtLink>
          <NuxtLink to="/sponsors" class="btn-outline">
            協賛企業・団体
          </NuxtLink>
          <NuxtLink to="/career-crossroads" class="btn-outline">
            キャリア交差点
          </NuxtLink>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
const { news, mainEventSchedule, formatDate, fetchData } = useCmsData()

onMounted(async () => {
  await fetchData()
})

useHead({
  title: '沖縄県立 開邦高校 第三回 大同窓会'
})
</script>
