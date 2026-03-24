import { useEffect, useState } from 'react'
import { useAppStore } from '@/stores/app-store'

type Theme = 'light' | 'dark' | 'system'

export function useTheme() {
  const { settings, updateSettings } = useAppStore()
  const [theme, setTheme] = useState<Theme>(settings.theme)
  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('dark')

  useEffect(() => {
    const root = document.documentElement

    const resolveTheme = (theme: Theme): 'light' | 'dark' => {
      if (theme === 'system') {
        return window.matchMedia('(prefers-color-scheme: dark)').matches
          ? 'dark'
          : 'light'
      }
      return theme
    }

    const newTheme = resolveTheme(theme)
    setResolvedTheme(newTheme)

    root.classList.remove('light', 'dark')
    root.classList.add(newTheme)
  }, [theme])

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')

    const handleChange = () => {
      if (settings.theme === 'system') {
        const newTheme = mediaQuery.matches ? 'dark' : 'light'
        setResolvedTheme(newTheme)
        document.documentElement.classList.remove('light', 'dark')
        document.documentElement.classList.add(newTheme)
      }
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [settings.theme])

  const setThemeWithPersist = (newTheme: Theme) => {
    setTheme(newTheme)
    updateSettings({ theme: newTheme })
  }

  return { theme, setTheme: setThemeWithPersist, resolvedTheme }
}
