import type { Page } from '@playwright/test';

/**
 * Logs in via Dex OIDC flow and waits for the token to be stored.
 */
export async function loginViaDex(page: Page) {
	await page.goto('/login');
	await page.waitForURL(/localhost:5556\/dex/, { timeout: 10000 });

	// Dex may show connector selection first
	const emailButton = page.locator('a:has-text("Email"), button:has-text("Email"), a:has-text("Log in")');
	if (await emailButton.isVisible({ timeout: 3000 }).catch(() => false)) {
		await emailButton.click();
	}

	// Fill in credentials
	await page.fill('input[type="text"], input[name="login"]', 'admin@tanrenai.local');
	await page.fill('input[type="password"], input[name="password"]', 'password');
	await page.click('button[type="submit"]');

	// Dex may show a grant/approval page
	const grantButton = page.locator('button:has-text("Grant Access"), button:has-text("Approve")');
	if (await grantButton.isVisible({ timeout: 3000 }).catch(() => false)) {
		await grantButton.click();
	}

	// Wait for redirect back to app
	await page.waitForURL(/localhost:5173/, { timeout: 15000 });

	// Wait for async token exchange to complete
	await page.waitForFunction(
		() => localStorage.getItem('access_token') !== null,
		{ timeout: 10000 },
	);
}
