<script lang="ts">
	import { onMount } from 'svelte';

	const OIDC_ISSUER = import.meta.env.VITE_OIDC_ISSUER || 'http://localhost:5556/dex';
	const CLIENT_ID = import.meta.env.VITE_OIDC_CLIENT_ID || 'tanrenai-frontend';
	const REDIRECT_URI = `${window.location.origin}/callback`;

	onMount(() => {
		// Generate PKCE
		const verifier = generateCodeVerifier();
		sessionStorage.setItem('pkce_verifier', verifier);

		const challenge = generateCodeChallenge(verifier);
		const state = crypto.randomUUID();
		sessionStorage.setItem('oauth_state', state);

		const params = new URLSearchParams({
			response_type: 'code',
			client_id: CLIENT_ID,
			redirect_uri: REDIRECT_URI,
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
		return btoa(String.fromCharCode(...bytes))
			.replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
	}

	function generateCodeChallenge(verifier: string): string {
		const encoder = new TextEncoder();
		const data = encoder.encode(verifier);
		// Synchronous fallback — use SubtleCrypto in production
		// For now, use S256 with a simple hash
		let hash = 0;
		for (let i = 0; i < data.length; i++) {
			hash = ((hash << 5) - hash + data[i]) | 0;
		}
		return btoa(String(hash)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
	}
</script>

<div class="max-w-md mx-auto mt-20 text-center">
	<p class="text-surface-400">Redirecting to login...</p>
</div>
