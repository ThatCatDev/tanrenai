import { test, expect } from '@playwright/test';
import { loginViaDex } from './helpers';

test.describe('Dashboard', () => {
	test('dashboard loads after login', async ({ page }) => {
		await loginViaDex(page);
		await page.goto('/');

		// Should show either "No GPU Instance" or an instance card — either means the page works
		const dashboard = page.locator('h1:has-text("Dashboard")');
		await expect(dashboard).toBeVisible({ timeout: 5000 });
	});

	test('instance status endpoint returns valid response', async ({ page }) => {
		await loginViaDex(page);

		const token = await page.evaluate(() => localStorage.getItem('access_token'));
		expect(token).toBeTruthy();

		const resp = await page.request.get('http://localhost:3000/api/instance/status', {
			headers: { Authorization: `Bearer ${token!}` },
		});
		expect(resp.ok()).toBeTruthy();
		const body = await resp.json();
		// Status should be a valid value
		expect(['none', 'pending', 'provisioning', 'running', 'destroying', 'destroyed']).toContain(body.status);
	});
});

test.describe('Settings', () => {
	test('settings page loads with sections', async ({ page }) => {
		await loginViaDex(page);
		await page.goto('/settings');

		await expect(page.locator('text=Vast.ai API Key')).toBeVisible({ timeout: 5000 });
		await expect(page.locator('text=Instance Defaults')).toBeVisible();
	});

	test('can update user settings', async ({ page }) => {
		await loginViaDex(page);
		await page.goto('/settings');

		await expect(page.locator('button:has-text("Save Settings")')).toBeVisible({ timeout: 5000 });
		await page.click('button:has-text("Save Settings")');
		await expect(page.locator('text=Settings saved')).toBeVisible({ timeout: 5000 });
	});
});

test.describe('Instances page', () => {
	test('instances page shows provision form', async ({ page }) => {
		await loginViaDex(page);
		await page.goto('/instances');

		await expect(page.locator('text=Provision New Instance')).toBeVisible({ timeout: 5000 });
		await expect(page.locator('input[placeholder*="8b"]')).toBeVisible();
	});
});
