import { test, expect } from '@playwright/test';

test.describe('Health checks', () => {
	test('platform API is reachable', async ({ request }) => {
		const resp = await request.get('http://localhost:3000/health');
		expect(resp.ok()).toBeTruthy();
		const body = await resp.json();
		expect(body.status).toBe('ok');
	});

	test('Dex OIDC discovery is reachable', async ({ request }) => {
		const resp = await request.get('http://localhost:5556/dex/.well-known/openid-configuration');
		expect(resp.ok()).toBeTruthy();
		const body = await resp.json();
		expect(body.issuer).toBe('http://localhost:5556/dex');
		expect(body.authorization_endpoint).toContain('/auth');
		expect(body.token_endpoint).toContain('/token');
	});

	test('frontend loads', async ({ page }) => {
		await page.goto('/');
		await expect(page.locator('nav')).toBeVisible();
		await expect(page.locator('a:has-text("Tanrenai")')).toBeVisible();
	});

	test('frontend shows login link when not authenticated', async ({ page }) => {
		await page.goto('/');
		await expect(page.locator('nav a:has-text("Login")')).toBeVisible();
	});
});
