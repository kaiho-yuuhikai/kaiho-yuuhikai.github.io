<template>
  <div>
    <!-- Hero -->
    <section class="relative h-[40vh] min-h-[300px] flex items-center justify-center bg-kaiho-green">
      <div class="relative z-10 text-center text-white px-6">
        <p class="text-sm tracking-[0.3em] uppercase mb-4 text-white/60">Members</p>
        <h1 class="text-4xl md:text-5xl lg:text-6xl font-light tracking-tight">
          登壇者・運営
        </h1>
      </div>
    </section>

    <!-- Content -->
    <section class="py-24 md:py-32">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="max-w-3xl mx-auto text-center mb-16">
          <p class="section-label">Speakers & Staff</p>
          <h2 class="section-title mb-8">第三回大同窓会 登壇者・運営</h2>
          <div class="divider mx-auto mb-8"></div>
          <p class="text-neutral-600 leading-relaxed">
            大同窓会では、様々な分野で活躍する卒業生の皆様に<br>
            ご登壇いただきました。
          </p>
        </div>

        <!-- Members Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <button
            v-for="member in profiles"
            :key="member.slug"
            class="bg-white border border-neutral-100 rounded-lg overflow-hidden hover:shadow-lg transition-shadow text-left cursor-pointer flex flex-col"
            @click="openModal(member)"
          >
            <div class="aspect-square bg-neutral-100">
              <img
                v-if="member.avatar"
                :src="member.avatar"
                :alt="member.title"
                class="w-full h-full object-cover"
              />
              <div v-else class="w-full h-full flex items-center justify-center">
                <svg class="w-16 h-16 text-neutral-300" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                </svg>
              </div>
            </div>
            <div class="p-6 flex-1">
              <p v-if="member.generation || member.department" class="text-xs text-kaiho-green font-medium mb-1">
                {{ member.generation }}<span v-if="member.generation && member.department">・</span>{{ member.department }}
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
          v-if="selectedMember"
          class="fixed inset-0 z-50 flex items-center justify-center p-4"
          @click.self="closeModal"
        >
          <!-- Backdrop -->
          <div class="absolute inset-0 bg-black/60 backdrop-blur-sm" @click="closeModal"></div>

          <!-- Modal Content -->
          <div class="relative bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
            <!-- Close Button -->
            <button
              class="absolute top-4 right-4 z-10 w-10 h-10 flex items-center justify-center rounded-full bg-white/90 hover:bg-white text-neutral-500 hover:text-neutral-900 transition-colors shadow-md"
              @click="closeModal"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            <!-- Member Image -->
            <div v-if="selectedMember.avatar" class="bg-neutral-100">
              <img
                :src="selectedMember.avatar"
                :alt="selectedMember.title"
                class="w-full h-auto"
              />
            </div>

            <!-- Member Info -->
            <div class="p-8">
              <p v-if="selectedMember.generation || selectedMember.department" class="text-sm text-kaiho-green font-medium mb-2">
                {{ selectedMember.generation }}<span v-if="selectedMember.generation && selectedMember.department">・</span>{{ selectedMember.department }}
              </p>
              <h2 class="text-2xl md:text-3xl font-medium text-neutral-900 mb-3">
                {{ selectedMember.title }}
              </h2>
              <p v-if="selectedMember.role" class="text-lg text-neutral-600 mb-6">
                {{ selectedMember.role }}
              </p>
              <div class="border-t border-neutral-100 pt-6">
                <p v-if="selectedMember.description" class="text-neutral-600 leading-relaxed whitespace-pre-wrap">
                  {{ selectedMember.description }}
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
import type { Profile } from '~/composables/useCmsData'

const { profiles, fetchData } = useCmsData()

const selectedMember = ref<Profile | null>(null)

const openModal = (member: Profile) => {
  selectedMember.value = member
  document.body.style.overflow = 'hidden'
}

const closeModal = () => {
  selectedMember.value = null
  document.body.style.overflow = ''
}

// Close modal on Escape key
onMounted(async () => {
  await fetchData()

  const handleEscape = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && selectedMember.value) {
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
  title: '登壇者・運営 | 開邦高校 大同窓会'
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

.modal-enter-active .relative,
.modal-leave-active .relative {
  transition: transform 0.3s ease;
}

.modal-enter-from .relative,
.modal-leave-to .relative {
  transform: scale(0.95);
}
</style>
