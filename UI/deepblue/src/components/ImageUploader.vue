<template>
  <div 
    class="max-w-xl mx-auto bg-gradient-to-r from-blue-50 to-indigo-100 rounded-xl shadow-lg p-8"
    @dragover.prevent
    @drop.prevent="handleDrop"
  >
    <div
      class="border-2 border-dashed rounded-xl p-8 text-center transition-all"
      :class="{ 'border-blue-500 bg-blue-100': isDragging, 'border-gray-300 bg-white': !isDragging }"
      @dragenter.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
    >
      <input
        type="file"
        ref="fileInput"
        class="hidden"
        accept="image/*"
        @change="handleFileSelect"
      >

      <template v-if="!selectedImage">
        <div class="mb-4">
          <i class="fas fa-cloud-upload-alt text-6xl text-gray-400 animate-bounce"></i>
        </div>
        <p class="text-lg font-medium text-gray-600 mb-2">Drag and drop your image here</p>
        <p class="text-gray-500 mb-4">- or -</p>
        <button
          @click="$refs.fileInput.click()"
          class="bg-gradient-to-r from-blue-500 to-indigo-500 text-white px-6 py-3 rounded-full shadow-md hover:shadow-lg hover:from-blue-600 hover:to-indigo-600 transition-all"
        >
          Browse Files
        </button>
      </template>

      <template v-else>
        <img 
          :src="imagePreview" 
          alt="Selected image"
          class="max-h-64 mx-auto mb-4 rounded-lg shadow-md"
        />
        <div class="flex justify-center gap-4">
          <button
            @click="analyzeImage"
            class="bg-gradient-to-r from-green-400 to-green-600 text-white px-6 py-3 rounded-full shadow-md hover:shadow-lg hover:from-green-500 hover:to-green-700 transition-all"
            :disabled="isAnalyzing"
          >
            {{ isAnalyzing ? 'Analyzing...' : 'Analyze Image' }}
          </button>
          <button
            @click="resetImage"
            class="bg-gradient-to-r from-red-400 to-red-600 text-white px-6 py-3 rounded-full shadow-md hover:shadow-lg hover:from-red-500 hover:to-red-700 transition-all"
            :disabled="isAnalyzing"
          >
            Cancel
          </button>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

import { useRouter } from 'vue-router'
const router = useRouter()

const emit = defineEmits(['analysis-complete'])

const isDragging = ref(false)
const selectedImage = ref(null)
const imagePreview = ref('')
const isAnalyzing = ref(false)

const handleDrop = (e) => {
  isDragging.value = false
  const file = e.dataTransfer.files[0]
  if (file && file.type.startsWith('image/')) {
    processSelectedFile(file)
  }
}

const handleFileSelect = (e) => {
  const file = e.target.files[0]
  if (file) {
    processSelectedFile(file)
  }
}

const processSelectedFile = (file) => {
  selectedImage.value = file
  imagePreview.value = URL.createObjectURL(file)
}

const resetImage = () => {
  selectedImage.value = null
  imagePreview.value = ''
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

const analyzeImage = async () => {
  if (!selectedImage.value) return;

  isAnalyzing.value = true;
  try {
    const formData = new FormData();
    formData.append('file', selectedImage.value);

    const response = await axios.post('http://localhost:8000/analyze-image/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });

    console.log("API Response", response.data);  // Check the API response in the console

    // Ensure that the response is structured correctly to be passed to the results page
    router.push({
      path: '/results',
      state: {
        analysisResults: response.data  // Pass the entire response data here
      }
    });
  } catch (error) {
    console.error('Analysis failed:', error);
  } finally {
    isAnalyzing.value = false;
  }
};

</script>

<style scoped>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');
</style>