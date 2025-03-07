<template>
  <div class="min-h-screen bg-gray-50 p-8">
    <div class="max-w-7xl mx-auto space-y-6">
      <!-- Main Results Cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <!-- Height Estimation Card -->
        <div class="bg-white p-4 rounded-md shadow-md">
          <div class="border-b border-gray-100 mb-4">
            <h2 class="text-2xl font-bold flex items-center gap-2">
              <Ruler class="w-6 h-6 text-blue-600" />
              Height Estimation
            </h2>
          </div>
          <div class="p-4">
            <div class="grid grid-cols-1 sm:grid-cols-1 gap-4">
              <MeasurementCard
                title="Estimated Height"
                :value="`${formatHeightValue(processedResults.height_estimation?.estimated_height)} cm`"
                :icon="Ruler"
                iconColor="text-blue-600"
                extraClasses="border-l-4 border-blue-500"
              />
              <MeasurementCard
                title="Height Confidence"
                :value="`${formatConfidenceScore(processedResults.height_estimation?.confidence_score)}%`"
                :icon="AlertCircle"
                iconColor="text-blue-600"
                extraClasses="border-l-4 border-blue-500"
              />
            </div>
          </div>
        </div>

        <!-- Weight Estimation Card -->
        <div class="bg-white p-4 rounded-md shadow-md">
          <div class="border-b border-gray-100 mb-4">
            <h2 class="text-2xl font-bold flex items-center gap-2">
              <Scale class="w-6 h-6 text-green-600" />
              Weight Estimation
            </h2>
          </div>
          <div class="p-4">
            <div class="grid grid-cols-1 sm:grid-cols-1 gap-4">
              <MeasurementCard
                title="Estimated Weight"
                :value="`${formatWeightValue(processedResults.weight_estimation?.estimated_weight)} kg`"
                :icon="Scale"
                iconColor="text-green-600"
                extraClasses="border-l-4 border-green-500"
              />
              <MeasurementCard
                title="Body Type"
                :value="formatBodyType(processedResults.weight_estimation?.body_type || 'N/A')"
                :icon="Activity"
                iconColor="text-green-600"
                extraClasses="border-l-4 border-green-500"
              />
            </div>
          </div>
        </div>

        <!-- Age Estimation Card -->
        <div class="bg-white p-4 rounded-md shadow-md">
          <div class="border-b border-gray-100 mb-4">
            <h2 class="text-2xl font-bold flex items-center gap-2">
              <Clock class="w-6 h-6 text-purple-600" />
              Age Estimation
            </h2>
          </div>
          <div class="p-4">
            <div class="grid grid-cols-1 sm:grid-cols-1 gap-4">
              <MeasurementCard
                title="Estimated Age"
                :value="formatAgeValue(processedResults.age_estimation?.estimated_age)"
                :icon="Calendar"
                iconColor="text-purple-600"
                extraClasses="border-l-4 border-purple-500"
              />
              <MeasurementCard
                title="Age Range"
                :value="processedResults.age_estimation?.age_range || 'N/A'"
                :icon="Clock"
                iconColor="text-purple-600"
                extraClasses="border-l-4 border-purple-500"
              />
            </div>
          </div>
        </div>
      </div>

      <!-- Measurements Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Body Measurements -->
        <div class="bg-white p-4 rounded-md shadow-md">
          <div class="mb-4">
            <h3 class="text-xl font-semibold flex items-center gap-2">
              <User class="w-5 h-5 text-blue-600" />
              Facial Measurements
            </h3>
          </div>
          <div class="space-y-3">
            <div
              v-for="[key, value] in Object.entries(processedResults.height_estimation?.measurements || {})"
              :key="key"
              class="flex justify-between items-center p-2 hover:bg-gray-50 rounded"
            >
              <span class="text-gray-600 font-medium">{{ formatMeasurementKey(key) }}</span>
              <span class="text-gray-900 font-semibold">{{ formatValue(value) }} cm</span>
            </div>
          </div>
        </div>

        <!-- Image Preview -->
        <div v-if="processedResults.imageUrl" class="bg-white p-4 rounded-md shadow-md">
          <div class="mb-4">
            <h3 class="text-xl font-semibold flex items-center gap-2">
              <Camera class="w-5 h-5 text-blue-600" />
              Analyzed Image
            </h3>
          </div>
          <div class="p-4">
            <div class="relative aspect-square rounded-lg overflow-hidden">
              <img 
                :src="processedResults.imageUrl" 
                alt="Analyzed subject"
                class="object-cover w-full h-full"
              />
              <div class="absolute bottom-0 left-0 right-0 bg-black/50 text-white p-2 text-sm">
                {{ processedResults.dimensions?.width }} x {{ processedResults.dimensions?.height }} px
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Analysis Section -->
      <div v-if="processedResults.analysis" class="bg-white p-4 rounded-md shadow-md">
        <div class="mb-4">
          <h3 class="text-xl font-semibold flex items-center gap-2">
            <BarChart class="w-5 h-5 text-blue-600" />
            Comprehensive Analysis
          </h3>
        </div>
        <div class="p-6">
          <div class="space-y-6">
            <div
              v-for="(section, index) in analysisSections"
              :key="index"
              class="space-y-2"
            >
              <h3 class="text-lg font-medium text-gray-800">
                {{ section.title }}
              </h3>
              <ul class="space-y-2">
                <li
                  v-for="(item, idx) in section.items"
                  :key="idx"
                  class="text-gray-600 pl-4 border-l-2 border-gray-200"
                >
                  {{ item }}
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { Ruler, AlertCircle, Camera, User, BarChart, Scale, Activity, Clock, Calendar } from 'lucide-vue-next';
import MeasurementCard from './MeasurementCard.vue';

const props = defineProps({
  results: {
    type: Object,
    required: true,
    validator: (value) => {
      return value.analysis && (value.imageUrl || value.dimensions);
    }
  }
});

// Computed properties
const processedResults = computed(() => {
  return {
    ...props.results,
    height_estimation: props.results.height_estimation || {},
    weight_estimation: props.results.weight_estimation || {},
    age_estimation: props.results.age_estimation || {},
    analysis: props.results.analysis || ''
  };
});

const analysisSections = computed(() => {
  return processedResults.value.analysis.split('###')
    .slice(1)
    .map(section => {
      const lines = section.trim().split('\n');
      return {
        title: lines[0].trim(),
        items: lines.slice(1).map(line => line.trim()).filter(Boolean)
      };
    });
});

// Utility functions
const formatMeasurementKey = (key) => key.replace(/_/g, ' ').toUpperCase();

const formatHeightValue = (value) => {
  return value?.toFixed(1) || 'N/A';
};

const formatWeightValue = (value) => {
  return value?.toFixed(1) || 'N/A';
};

const formatAgeValue = (value) => {
  return value ? `${value} years` : 'N/A';
};

const formatBodyType = (bodyType) => {
  if (!bodyType || bodyType === 'N/A') return 'N/A';
  
  // Convert snake_case or kebab-case to Title Case
  return bodyType
    .replace(/[-_]/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

const formatConfidenceScore = (score) => {
  return ((score || 0) * 100).toFixed(1);
};

const formatValue = (value) => {
  return typeof value === 'number' ? value.toFixed(2) : 'N/A';
};
</script>

<style scoped>
.aspect-square {
  aspect-ratio: 1 / 1;
}
</style>