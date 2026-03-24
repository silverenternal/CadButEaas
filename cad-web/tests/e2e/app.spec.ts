import { test, expect } from '@playwright/test'

test.describe('CAD Web UI', () => {
  test('should load the application', async ({ page }) => {
    await page.goto('/')

    // Check if the app loads
    await expect(page).toHaveTitle(/CAD 几何智能处理系统/)

    // Check if the main toolbar is visible
    await expect(page.getByText('几何智能处理系统')).toBeVisible()
  })

  test('should display canvas area', async ({ page }) => {
    await page.goto('/')

    // Check if canvas area is present
    const canvas = page.locator('canvas')
    await expect(canvas).toBeVisible()
  })

  test('should open file dialog when clicking open button', async ({ page }) => {
    await page.goto('/')

    // Click the open file button
    const openButton = page.getByRole('button', { name: /打开/ })
    await openButton.click()

    // File input should be triggered
    // Note: Actual file selection requires a real file
  })

  test('should switch tools when clicking tool buttons', async ({ page }) => {
    await page.goto('/')

    // Click trace tool
    const traceButton = page.getByRole('button', { name: /追踪/ })
    await traceButton.click()

    // Tool should be activated (check for active state)
    await expect(traceButton).toHaveClass(/bg-secondary/)
  })

  test('should toggle sidebar when clicking sidebar button', async ({ page }) => {
    await page.goto('/')

    // Sidebar should be visible initially
    const sidebar = page.locator('aside').first()
    await expect(sidebar).toBeVisible()

    // Click toggle button
    const toggleButton = page.getByRole('button', { name: /侧边/ })
    await toggleButton.click()

    // Sidebar should be hidden
    await expect(sidebar).not.toBeVisible()
  })
})
