<template>
  <div class="min-h-screen bg-gray-50">
    <nav class="bg-white shadow-sm">
      <div class="container mx-auto px-4">
        <div class="flex justify-between items-center h-16">
          <router-link 
            to="/" 
            class="text-xl font-bold text-blue-600 hover:text-blue-700"
          >
            Image Analyzer
          </router-link>
          
          <div class="flex space-x-4">
            <router-link 
              to="/" 
              class="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md"
              active-class="text-blue-600"
            >
              Home
            </router-link>
            <router-link 
            to="/how-it-works" 
            class="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md"
            >
            How It Works
          </router-link>
          
          <router-link 
          to="/contact" 
          class="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md"
          >
          Contact
          </router-link>
          </div>
        </div>
      </div>
    </nav>

    <router-view v-slot="{ Component }">
      <transition 
        name="fade" 
        mode="out-in"
      >
        <component :is="Component" />
      </transition>
    </router-view>
  </div>
</template>

<script setup>
// No additional setup needed for now
</script>

<style>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* Global styles */
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}
</style>
