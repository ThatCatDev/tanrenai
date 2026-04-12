<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { fetchUser } from '$lib/stores/user';

	const OIDC_ISSUER = import.meta.env.VITE_OIDC_ISSUER || 'http://localhost:5556/dex';
	const CLIENT_ID = import.meta.env.VITE_OIDC_CLIENT_ID || 'tanrenai-frontend';
	const REDIRECT_URI = `${window.location.origin}/callback`;

	let error = $state('');

	onMount(async () => {
		const params = new URLSearchParams(window.location.search);
		const code = params.get('code');
		const state = params.get('state');
		const savedState = sessionStorage.getItem('oauth_state');
		const verifier = sessionStorage.getItem('pkce_verifier');

		if (!code) {
			error = params.get('error_description') || 'No authorization code received';
			return;
		}

		if (state !== savedState) {
			error = 'State mismatch — possible CSRF attack';
			return;
		}

		try {
			const resp = await fetch(`${OIDC_ISSUER}/token`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
				body: new URLSearchParams({
					grant_type: 'authorization_code',
					code,
					redirect_uri: REDIRECT_URI,
					client_id: CLIENT_ID,
					code_verifier: verifier || '',
				}),
			});

			if (!resp.ok) {
				const body = await resp.text();
				error = `Token exchange failed: ${body}`;
				return;
			}

			const data = await resp.json();
			const idToken = data.id_token;

			if (!idToken) {
				error = 'No id_token in response';
				return;
			}

			localStorage.setItem('access_token', idToken);
			if (data.refresh_token) {
				localStorage.setItem('refresh_token', data.refresh_token);
			}

			// Clean up
			sessionStorage.removeItem('oauth_state');
			sessionStorage.removeItem('pkce_verifier');

			await fetchUser();
			goto('/');
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error during login';
		}
	});
</script>

<div class="max-w-md mx-auto mt-20 text-center">
	{#if error}
		<div class="card p-6 bg-error-500/20 text-error-300">
			<h2 class="text-lg font-bold mb-2">Login Failed</h2>
			<p>{error}</p>
			<a href="/login" class="text-primary-400 underline mt-4 inline-block">Try again</a>
		</div>
	{:else}
		<p class="text-surface-400">Completing login...</p>
	{/if}
</div>
