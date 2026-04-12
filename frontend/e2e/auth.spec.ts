import { test, expect } from '@playwright/test';
import { loginViaDex } from './helpers';

test.describe('OIDC Authentication Flow', () => {
	test('login redirects to Dex', async ({ page }) => {
		await page.goto('/login');
		await page.waitForURL(/localhost:5556\/dex/, { timeout: 10000 });
		await expect(page.locator('input[type="text"], input[name="login"]')).toBeVisible({ timeout: 5000 });
	});

	test('full login flow with Dex password connector', async ({ page }) => {
		await loginViaDex(page);

		const token = await page.evaluate(() => localStorage.getItem('access_token'));
		expect(token).toBeTruthy();
		expect(token!.length).toBeGreaterThan(10);
	});

	test('authenticated user sees nav links', async ({ page }) => {
		await loginViaDex(page);

		await page.goto('/');
		await expect(page.locator('a:has-text("Dashboard")')).toBeVisible({ timeout: 5000 });
		await expect(page.locator('a:has-text("Instances")')).toBeVisible();
		await expect(page.locator('a:has-text("Settings")')).toBeVisible();
	});
});
