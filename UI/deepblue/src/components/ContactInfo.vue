<template>
    <div class="container mx-auto px-4 py-8">
      <div class="mt-12 bg-gradient-to-r from-gray-50 via-gray-100 to-gray-200 p-8 rounded-lg shadow-md">
        <h2 class="text-3xl font-extrabold text-gray-800 text-center mb-6">Features & Benefits</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div class="p-6 bg-white rounded-lg shadow-md text-center transition-transform transform hover:scale-105">
            <img src="C:\Users\anant\OneDrive\Desktop\DeepBlue\UI\deepblue\src\Images\upload_icon.svg" alt="Upload Icon" class="w-16 h-16 mx-auto mb-4">
            <h3 class="text-xl font-bold mb-2">Easy Image Upload</h3>
            <p class="text-gray-600">Drag and drop images or browse from your device with a simple click.</p>
          </div>
          <div class="p-6 bg-white rounded-lg shadow-md text-center transition-transform transform hover:scale-105">
            <img src="C:\Users\anant\OneDrive\Desktop\DeepBlue\UI\deepblue\src\Images\analysis_icon.svg" alt="Analysis Icon" class="w-16 h-16 mx-auto mb-4">
            <h3 class="text-xl font-bold mb-2">AI-Powered Analysis</h3>
            <p class="text-gray-600">Leverage advanced AI to extract object dimensions and attributes accurately.</p>
          </div>
          <div class="p-6 bg-white rounded-lg shadow-md text-center transition-transform transform hover:scale-105">
            <img src="C:\Users\anant\OneDrive\Desktop\DeepBlue\UI\deepblue\src\Images\results_icon.svg" alt="Results Icon" class="w-16 h-16 mx-auto mb-4">
            <h3 class="text-xl font-bold mb-2">Detailed Insights</h3>
            <p class="text-gray-600">Receive detailed results, including object type, dimensions, and attributes.</p>
          </div>
        </div>
      </div>
  
      <div class="mt-12 text-center">
        <h2 class="text-3xl font-extrabold text-gray-800 mb-4">Why Choose Us?</h2>
        <p class="text-lg text-gray-700 mb-6">Our tool is designed to provide accurate, quick, and user-friendly analysis tailored to your needs.</p>
        <img src="C:\Users\anant\OneDrive\Desktop\DeepBlue\UI\deepblue\src\Images\hero_image.svg" alt="Illustration" class="mx-auto max-w-md">
      </div>
  
      <div class="mt-12 bg-white p-8 rounded-lg shadow-md">
        <h2 class="text-3xl font-bold text-gray-800 mb-6 text-center">Contact Us</h2>
        <form @submit.prevent="handleSubmit" class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Name</label>
            <input
              v-model="formData.name"
              type="text"
              class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              required
            >
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input
              v-model="formData.email"
              type="email"
              class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              required
            >
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Message</label>
            <textarea
              v-model="formData.message"
              rows="4"
              class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              required
            ></textarea>
          </div>
          <button
            type="submit"
            class="w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition-colors"
            :disabled="isSubmitting"
          >
            {{ isSubmitting ? 'Sending...' : 'Send Message' }}
          </button>
        </form>
        <div class="border-t pt-6 mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 class="text-lg font-semibold mb-2">Email Us</h3>
            <p class="text-gray-600">support@imageanalyzer.com</p>
            <p class="text-gray-600">business@imageanalyzer.com</p>
          </div>
          <div>
            <h3 class="text-lg font-semibold mb-2">Working Hours</h3>
            <p class="text-gray-600">Monday - Friday: 9:00 AM - 6:00 PM</p>
            <p class="text-gray-600">Weekend: Closed</p>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue'
  import { useRouter } from 'vue-router'
  import ImageUploader from '../components/ImageUploader.vue'
  
  const router = useRouter()
  
  const formData = ref({
    name: '',
    email: '',
    message: ''
  })
  
  const isSubmitting = ref(false)
  
  const handleSubmit = async () => {
    isSubmitting.value = true
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500))
      alert('Message sent successfully!')
      formData.value = {
        name: '',
        email: '',
        message: ''
      }
    } catch (error) {
      alert('Failed to send message. Please try again.')
    } finally {
      isSubmitting.value = false
    }
  }
  
  const handleAnalysisComplete = (results) => {
    router.push({ 
      name: 'results',
      params: { results: JSON.stringify(results) }
    })
  }
  </script>
  