<template>
  <div>
    <!-- Hero -->
    <section class="relative h-[50vh] min-h-[400px] flex items-center justify-center bg-neutral-900">
      <div class="absolute inset-0">
        <img
          src="/images/cover-image.png"
          alt="キャリアクロスロード"
          class="w-full h-full object-cover opacity-30"
        />
      </div>
      <div class="relative z-10 text-center text-white px-6">
        <p class="text-sm tracking-[0.3em] uppercase mb-4 text-white/60">Pre-Event</p>
        <h1 class="text-4xl md:text-5xl lg:text-6xl font-light tracking-tight mb-4">
          開邦キャリア・クロスロード
        </h1>
        <p class="text-lg text-white/70">
          大同窓会 事前イベント
        </p>
      </div>
    </section>

    <!-- About -->
    <section class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center mb-16">
          <p class="section-label">About</p>
          <h2 class="section-title mb-8">イベントについて</h2>
          <div class="divider mx-auto mb-8"></div>
          <p class="text-neutral-600 leading-relaxed text-lg mb-6">
            本イベントは、先輩卒業生（メンター）と参加者が、<br class="hidden md:block">
            立場や世代を越えて、気軽に交流できる場として企画しています。
          </p>
          <p class="text-neutral-600 leading-relaxed">
            「誰かが教える場」ではなく、<br class="hidden md:block">
            それぞれの立場にとって学びや気づきのある交流の場です。
          </p>
        </div>

        <div class="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          <div class="bg-neutral-50 rounded-lg p-8">
            <h3 class="text-lg font-medium text-neutral-900 mb-4">同窓生・社会人の方</h3>
            <ul class="space-y-3 text-neutral-600">
              <li class="flex gap-3">
                <span class="text-kaiho-green flex-shrink-0">-</span>
                今の在校生や学生が、何に悩み、何を考えているのかを知りたい方
              </li>
              <li class="flex gap-3">
                <span class="text-kaiho-green flex-shrink-0">-</span>
                自身のこれまでの選択や経験を、あらためて振り返ってみたい方
              </li>
              <li class="flex gap-3">
                <span class="text-kaiho-green flex-shrink-0">-</span>
                同業種・異業種の同窓生と、自然な形でつながりたい方
              </li>
            </ul>
          </div>
          <div class="bg-neutral-50 rounded-lg p-8">
            <h3 class="text-lg font-medium text-neutral-900 mb-4">在校生・学生の方</h3>
            <ul class="space-y-3 text-neutral-600">
              <li class="flex gap-3">
                <span class="text-kaiho-green flex-shrink-0">-</span>
                少し先を歩く先輩のリアルな話を、直接聞くことができます
              </li>
              <li class="flex gap-3">
                <span class="text-kaiho-green flex-shrink-0">-</span>
                正解が決まっていない進路や選択について、「迷っていい」と感じられる時間になります
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <!-- Schedule -->
    <section class="py-24 md:py-32 bg-kaiho-green text-white">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center mb-16">
          <p class="text-xs font-medium tracking-widest uppercase text-white/60 mb-4">Schedule</p>
          <h2 class="text-3xl md:text-4xl lg:text-5xl font-light tracking-tight text-white">タイムテーブル</h2>
        </div>

        <div class="max-w-2xl mx-auto">
          <div class="space-y-6">
            <div
              v-for="item in careerSchedule"
              :key="item.slug"
              class="flex gap-6 pb-6 border-b border-white/20"
            >
              <div class="flex-shrink-0 w-28 text-white/60 text-sm">
                {{ item.startTime }}<span v-if="item.endTime"> - {{ item.endTime }}</span>
              </div>
              <div>
                <h3 class="text-white font-medium mb-2">{{ item.title }}</h3>
                <p v-if="item.description" class="text-white/70 text-sm leading-relaxed">
                  {{ item.description }}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Mentors -->
    <section class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center mb-16">
          <p class="section-label">Mentors</p>
          <h2 class="section-title mb-8">メンター紹介</h2>
          <div class="divider mx-auto mb-8"></div>
          <p class="text-neutral-600 leading-relaxed">
            様々な分野で活躍する先輩卒業生がメンターとして参加しました。
          </p>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          <button
            v-for="mentor in careerMentors"
            :key="mentor.slug"
            class="bg-white border border-neutral-100 rounded-lg overflow-hidden hover:shadow-lg transition-shadow text-left cursor-pointer"
            @click="openModal(mentor)"
          >
            <div v-if="mentor.avatar" class="aspect-square bg-neutral-100">
              <img
                :src="mentor.avatar"
                :alt="mentor.title"
                class="w-full h-full object-cover"
              />
            </div>
            <div class="p-4">
              <p v-if="mentor.generation" class="text-xs text-kaiho-green font-medium mb-1">
                {{ mentor.generation }}<span v-if="mentor.department"> {{ mentor.department }}</span>
              </p>
              <h3 class="text-sm font-medium text-neutral-900 mb-1">
                {{ mentor.title }}
              </h3>
              <p v-if="mentor.role" class="text-xs text-neutral-500 line-clamp-2">
                {{ mentor.role }}
              </p>
            </div>
          </button>
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

    <!-- Modal -->
    <Teleport to="body">
      <Transition name="modal">
        <div
          v-if="selectedMentor"
          class="fixed inset-0 z-50 flex items-center justify-center p-4"
          @click.self="closeModal"
        >
          <div class="absolute inset-0 bg-black/60 backdrop-blur-sm" @click="closeModal"></div>

          <div class="relative bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
            <button
              class="absolute top-4 right-4 z-10 w-10 h-10 flex items-center justify-center rounded-full bg-white/90 hover:bg-white text-neutral-500 hover:text-neutral-900 transition-colors shadow-md"
              @click="closeModal"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            <div v-if="selectedMentor.avatar" class="aspect-[4/3] bg-neutral-100">
              <img
                :src="selectedMentor.avatar"
                :alt="selectedMentor.title"
                class="w-full h-full object-cover"
              />
            </div>

            <div class="p-8">
              <p v-if="selectedMentor.generation" class="text-sm text-kaiho-green font-medium mb-2">
                {{ selectedMentor.generation }}<span v-if="selectedMentor.department"> {{ selectedMentor.department }}</span>
              </p>
              <h2 class="text-2xl md:text-3xl font-medium text-neutral-900 mb-3">
                {{ selectedMentor.title }}
              </h2>
              <p v-if="selectedMentor.role" class="text-lg text-neutral-600 mb-6">
                {{ selectedMentor.role }}
              </p>
              <div class="border-t border-neutral-100 pt-6">
                <p v-if="selectedMentor.description" class="text-neutral-600 leading-relaxed whitespace-pre-wrap">
                  {{ selectedMentor.description }}
                </p>
                <p v-else class="text-neutral-400 italic">
                  詳細情報はありません
                </p>
              </div>
            </div>
          </div>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import type { CareerMentor } from '~/composables/useCmsData'

const { careerMentors, careerSchedule, fetchData } = useCmsData()

const selectedMentor = ref<CareerMentor | null>(null)

const openModal = (mentor: CareerMentor) => {
  selectedMentor.value = mentor
  document.body.style.overflow = 'hidden'
}

const closeModal = () => {
  selectedMentor.value = null
  document.body.style.overflow = ''
}

onMounted(async () => {
  await fetchData()

  const handleEscape = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && selectedMentor.value) {
      closeModal()
    }
  }
  window.addEventListener('keydown', handleEscape)

  onUnmounted(() => {
    window.removeEventListener('keydown', handleEscape)
    document.body.style.overflow = ''
  })
})

useHead({
  title: '開邦キャリア・クロスロード | 開邦高校 大同窓会'
})
</script>

<style scoped>
.modal-enter-active,
.modal-leave-active {
  transition: opacity 0.3s ease;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}
</style>
