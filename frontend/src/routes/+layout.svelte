<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { currentUser, fetchUser } from '$lib/stores/user';

	let { children } = $props();

	const isLoggedIn = $derived(!!$currentUser);

	onMount(() => {
		const token = localStorage.getItem('access_token');
		if (token) {
			fetchUser();
		}
	});
</script>

<div class="h-screen flex flex-col bg-surface-900 text-surface-50">
	<!-- Nav -->
	<nav class="flex items-center gap-4 px-6 py-3 bg-surface-800 border-b border-surface-700">
		<a href="/" class="text-lg font-bold text-primary-400">Tanrenai</a>

		{#if isLoggedIn}
			<a href="/" class="text-sm hover:text-primary-300">Dashboard</a>
			<a href="/instances" class="text-sm hover:text-primary-300">Instances</a>
			<a href="/settings" class="text-sm hover:text-primary-300">Settings</a>
			<span class="ml-auto text-sm text-surface-400">{$currentUser?.email}</span>
		{:else}
			<a href="/login" class="ml-auto text-sm hover:text-primary-300">Login</a>
		{/if}
	</nav>

	<!-- Main -->
	<main class="flex-1 overflow-auto p-6">
		{@render children()}
	</main>
</div>
