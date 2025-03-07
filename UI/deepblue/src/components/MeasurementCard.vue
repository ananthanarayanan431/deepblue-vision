<template>
    <div :class="`p-4 rounded-lg bg-white shadow-md ${extraClasses}`">
      <div class="flex items-center gap-3">
        <component 
          :is="icon" 
          class="w-6 h-6"
          :class="iconColor"
        />
        <div>
          <p class="text-sm text-gray-500">{{ title }}</p>
          <p class="text-xl font-semibold">{{ value }}</p>
        </div>
      </div>
    </div>
  </template>
  
  <script setup>
  defineProps({
    title: {
      type: String,
      required: true
    },
    value: {
      type: String,
      required: true
    },
    icon: {
      type: Object,
      required: true
    },
    iconColor: {
      type: String,
      default: 'text-gray-600'
    },
    extraClasses: {
      type: String,
      default: ''
    }
  });
  </script>