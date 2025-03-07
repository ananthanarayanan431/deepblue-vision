<template>
  <div class="container mx-auto px-4 py-8 bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
    <div class="max-w-5xl mx-auto space-y-8">
      <!-- Loading State -->
      <div v-if="loading" class="text-center bg-white rounded-2xl shadow-xl p-12">
        <div class="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <p class="text-xl text-gray-600">Processing your image...</p>
      </div>

      <!-- Error State -->
      <div v-else-if="error" class="text-center bg-white rounded-2xl shadow-xl p-12">
        <div class="w-16 h-16 mx-auto mb-4 text-red-500">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <p class="text-xl text-red-600 mb-4">{{ error }}</p>
        <button
          @click="analyzeAnotherImage"
          class="bg-red-500 text-white px-6 py-2 rounded-lg hover:bg-red-600 transition-colors"
        >
          Try Again
        </button>
      </div>

      <!-- No Results State -->
      <div v-else-if="!analysisResults" class="text-center bg-white rounded-2xl shadow-xl p-12">
        <div class="w-full h-48 bg-gray-100 rounded-lg mb-6 flex items-center justify-center">
          <span class="text-gray-500 text-xl">No Image Available</span>
        </div>
        <p class="text-xl text-gray-600">No analysis results found. Please upload an image first.</p>
        <button
          @click="analyzeAnotherImage"
          class="mt-6 bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition-colors"
        >
          Upload Image
        </button>
      </div>

      <!-- Analysis Results -->
      <template v-else>
        <AnalysisResults 
          :results="analysisResults"
          @retry="analyzeAnotherImage"
        />

        <div class="text-center mt-8">
          <button
            @click="analyzeAnotherImage"
            class="bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-8 py-3 rounded-lg hover:from-blue-600 hover:to-indigo-700 transition-all transform hover:scale-105 shadow-lg"
          >
            Analyze Another Image
          </button>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import AnalysisResults from '@/components/AnalysisResults.vue'

const router = useRouter()
const analysisResults = ref(null)
const loading = ref(false)
const error = ref(null)

const analyzeAnotherImage = () => {
  router.push('/')
}

onMounted(() => {
  const state = window.history.state

  if (state && state.analysisResults) {
    try {
      loading.value = true
      error.value = null
      analysisResults.value = state.analysisResults
    } catch (err) {
      error.value = 'Failed to load analysis results. Please try again.'
      console.error('Error loading analysis results:', err)
    } finally {
      loading.value = false
    }
  } else {
    router.push('/')
  }
})
</script>

<style scoped>
.animate-spin {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}
</style>