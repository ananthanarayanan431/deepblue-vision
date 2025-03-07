<template>
  <div class="container mx-auto px-4 py-8">
    <!-- Main Heading -->
    <h1 class="text-4xl font-bold text-center mb-8 text-gray-800">
      Welcome to the Image Analysis Tool
    </h1>
    
    <!-- Subheading -->
    <p class="text-lg text-center text-gray-600 mb-12">
      Upload an image, and our AI will analyze it to provide object dimensions, type, and attributes.
    </p>

    <!-- Image Uploader Component -->
    <ImageUploader @analysis-complete="handleAnalysisComplete" />

    <!-- Supporting Section -->
    <div class="mt-16">
      <h2 class="text-3xl font-semibold text-center mb-6 text-gray-800">
        Why Choose Our Tool?
      </h2>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <!-- Feature 1 -->
        <div class="text-center p-6 bg-blue-50 rounded-lg shadow-md">
          <div class="w-16 h-16 mx-auto bg-blue-500 text-white rounded-full flex items-center justify-center mb-4">
            <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
          </div>
          <h3 class="text-xl font-semibold mb-2 text-gray-800">Accurate Dimensions</h3>
          <p class="text-gray-600">Get precise measurements of objects in your uploaded image.</p>
        </div>

        <!-- Feature 2 -->
        <div class="text-center p-6 bg-green-50 rounded-lg shadow-md">
          <div class="w-16 h-16 mx-auto bg-green-500 text-white rounded-full flex items-center justify-center mb-4">
            <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 class="text-xl font-semibold mb-2 text-gray-800">AI-Powered Analysis</h3>
          <p class="text-gray-600">Leverage advanced AI models for object detection and classification.</p>
        </div>

        <!-- Feature 3 -->
        <div class="text-center p-6 bg-orange-50 rounded-lg shadow-md">
          <div class="w-16 h-16 mx-auto bg-orange-500 text-white rounded-full flex items-center justify-center mb-4">
            <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 class="text-xl font-semibold mb-2 text-gray-800">User-Friendly Interface</h3>
          <p class="text-gray-600">Enjoy an intuitive design that simplifies your image analysis process.</p>
        </div>
      </div>
    </div>

    <!-- Call to Action -->
    <div class="text-center mt-16">
      <div class="mt-12 text-center">
      <h2 class="text-3xl font-extrabold text-gray-800 mb-4">Why Choose Us?</h2>
      <p class="text-lg text-gray-700 mb-6">Our tool is designed to provide accurate, quick, and user-friendly analysis tailored to your needs.</p>
    </div>
    </div>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import ImageUploader from '../components/ImageUploader.vue'

const router = useRouter()

const handleAnalysisComplete = (results) => {
  router.push({
    name: 'results',
    params: { results: JSON.stringify(results) },
  })
}
</script>
