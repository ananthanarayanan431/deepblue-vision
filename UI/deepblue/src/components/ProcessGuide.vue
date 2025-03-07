<template>
    <div class="max-w-4xl mx-auto p-8 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl shadow-lg">
      <h2 class="text-3xl font-extrabold mb-8 text-gray-800 text-center">How It Works</h2>
  
      <div class="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
        <div class="bg-blue-100 p-8 rounded-xl shadow-md hover:shadow-lg transition-shadow">
          <div class="w-16 h-16 bg-gradient-to-r from-blue-500 to-blue-700 rounded-full flex items-center justify-center mb-6 mx-auto">
            <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
            </svg>
          </div>
          <h3 class="text-xl font-bold text-center text-gray-800 mb-3">1. Upload Image</h3>
          <p class="text-gray-700 text-center">Drag and drop your image or click to browse. Supported formats: JPG, PNG.</p>
        </div>
  
        <div class="bg-green-100 p-8 rounded-xl shadow-md hover:shadow-lg transition-shadow">
          <div class="w-16 h-16 bg-gradient-to-r from-green-500 to-green-700 rounded-full flex items-center justify-center mb-6 mx-auto">
            <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
            </svg>
          </div>
          <h3 class="text-xl font-bold text-center text-gray-800 mb-3">2. Processing</h3>
          <p class="text-gray-700 text-center">Our AI analyzes your image to detect objects, dimensions, and attributes.</p>
        </div>
  
        <div class="bg-orange-100 p-8 rounded-xl shadow-md hover:shadow-lg transition-shadow">
          <div class="w-16 h-16 bg-gradient-to-r from-orange-500 to-orange-700 rounded-full flex items-center justify-center mb-6 mx-auto">
            <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
            </svg>
          </div>
          <h3 class="text-xl font-bold text-center text-gray-800 mb-3">3. View Results</h3>
          <p class="text-gray-700 text-center">Get detailed analysis including dimensions, object type, and attributes.</p>
        </div>
      </div>
    </div>
  </template>
  
  <script setup>
  </script>
  