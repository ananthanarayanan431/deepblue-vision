import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import ResultsView from '../views/ResultsView.vue'
import ProcessGuide from '@/components/ProcessGuide.vue'
import ContactInfo from '@/components/ContactInfo.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView
  },
  {
    path: '/results',
    name: 'results',
    component: ResultsView
  },
  {
    path: '/how-it-works',
    name: 'process',
    component: ProcessGuide
  },
  {
    path: '/Contact',
    name: "Contact",
    component: ContactInfo
  }
]

export default createRouter({
  history: createWebHistory(),
  routes
})