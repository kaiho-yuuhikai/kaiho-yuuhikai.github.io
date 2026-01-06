<template>
  <div class="min-h-screen flex flex-col">
    <header class="bg-white shadow-sm sticky top-0 z-50">
      <nav class="max-w-6xl mx-auto px-4 py-4">
        <div class="flex items-center justify-between">
          <NuxtLink to="/" class="text-xl font-bold text-primary-700">
            開邦高校 大同窓会
          </NuxtLink>
          <div class="hidden md:flex items-center space-x-6">
            <NuxtLink to="/" class="text-gray-600 hover:text-primary-600 transition-colors">
              ホーム
            </NuxtLink>
            <NuxtLink to="/menu" class="text-gray-600 hover:text-primary-600 transition-colors">
              メニュー
            </NuxtLink>
            <NuxtLink to="/career-crossroads" class="text-gray-600 hover:text-primary-600 transition-colors">
              キャリア交差点
            </NuxtLink>
            <NuxtLink to="/privacy" class="text-gray-600 hover:text-primary-600 transition-colors">
              プライバシー
            </NuxtLink>
          </div>
          <button
            @click="mobileMenuOpen = !mobileMenuOpen"
            class="md:hidden p-2 text-gray-600 hover:text-primary-600"
            aria-label="メニューを開く"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                v-if="!mobileMenuOpen"
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M4 6h16M4 12h16M4 18h16"
              />
              <path
                v-else
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
        <!-- Mobile Menu -->
        <div v-if="mobileMenuOpen" class="md:hidden mt-4 pb-4 border-t pt-4">
          <div class="flex flex-col space-y-3">
            <NuxtLink
              to="/"
              class="text-gray-600 hover:text-primary-600 transition-colors"
              @click="mobileMenuOpen = false"
            >
              ホーム
            </NuxtLink>
            <NuxtLink
              to="/menu"
              class="text-gray-600 hover:text-primary-600 transition-colors"
              @click="mobileMenuOpen = false"
            >
              メニュー
            </NuxtLink>
            <NuxtLink
              to="/career-crossroads"
              class="text-gray-600 hover:text-primary-600 transition-colors"
              @click="mobileMenuOpen = false"
            >
              キャリア交差点
            </NuxtLink>
            <NuxtLink
              to="/privacy"
              class="text-gray-600 hover:text-primary-600 transition-colors"
              @click="mobileMenuOpen = false"
            >
              プライバシー
            </NuxtLink>
          </div>
        </div>
      </nav>
    </header>

    <main class="flex-1">
      <slot />
    </main>

    <footer class="bg-gray-800 text-white py-8">
      <div class="max-w-6xl mx-auto px-4">
        <div class="text-center">
          <p class="text-lg font-semibold mb-2">沖縄県立 開邦高校 第三回 大同窓会</p>
          <p class="text-gray-400 text-sm">創立40周年記念</p>
          <div class="mt-4 flex justify-center space-x-4">
            <NuxtLink to="/privacy" class="text-gray-400 hover:text-white text-sm transition-colors">
              プライバシーポリシー
            </NuxtLink>
          </div>
          <p class="mt-6 text-gray-500 text-xs">
            &copy; 2025 開邦高校友会. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
const mobileMenuOpen = ref(false)
</script>
