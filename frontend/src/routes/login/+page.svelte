<script lang="ts">
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';

	const OIDC_ISSUER = import.meta.env.VITE_OIDC_ISSUER || 'http://localhost:5556/dex';
	const CLIENT_ID = import.meta.env.VITE_OIDC_CLIENT_ID || 'tanrenai-frontend';

	onMount(async () => {
		if (!browser) return;

		const redirectUri = `${window.location.origin}/callback`;

		// Generate PKCE verifier and challenge
		const verifier = generateCodeVerifier();
		sessionStorage.setItem('pkce_verifier', verifier);

		const challenge = await generateCodeChallenge(verifier);
		const state = crypto.randomUUID();
		sessionStorage.setItem('oauth_state', state);

		const params = new URLSearchParams({
			response_type: 'code',
			client_id: CLIENT_ID,
			redirect_uri: redirectUri,
			scope: 'openid email profile offline_access',
			state,
			code_challenge: challenge,
			code_challenge_method: 'S256',
		});

		window.location.href = `${OIDC_ISSUER}/auth?${params}`;
	});

	function generateCodeVerifier(): string {
		const bytes = new Uint8Array(32);
		crypto.getRandomValues(bytes);
		return base64UrlEncode(bytes);
	}

	async function generateCodeChallenge(verifier: string): Promise<string> {
		const encoder = new TextEncoder();
		const data = encoder.encode(verifier);
		const hash = await crypto.subtle.digest('SHA-256', data);
		return base64UrlEncode(new Uint8Array(hash));
	}

	function base64UrlEncode(bytes: Uint8Array): string {
		return btoa(String.fromCharCode(...bytes))
			.replace(/\+/g, '-')
			.replace(/\//g, '_')
			.replace(/=/g, '');
	}
</script>

<div class="max-w-md mx-auto mt-20 text-center">
	<p class="text-surface-400">Redirecting to login...</p>
</div>
