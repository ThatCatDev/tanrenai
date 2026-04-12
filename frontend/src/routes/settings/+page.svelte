<script lang="ts">
	import { onMount } from 'svelte';
	import { currentUser, fetchUser } from '$lib/stores/user';
	import { updateSettings, setVastaiKey, deleteVastaiKey } from '$lib/api';

	let idleTimeout = $state(60);
	let maxCost = $state(1.0);
	let preferredGpu = $state('');
	let apiKey = $state('');
	let saving = $state(false);
	let savingKey = $state(false);
	let message = $state('');

	const user = $derived($currentUser);

	onMount(async () => {
		await fetchUser();
		if ($currentUser) {
			idleTimeout = $currentUser.idle_timeout_min;
			maxCost = $currentUser.max_cost_per_hr;
			preferredGpu = $currentUser.preferred_gpu;
		}
	});

	async function handleSaveSettings() {
		saving = true;
		message = '';
		try {
			await updateSettings({
				idle_timeout_min: idleTimeout,
				max_cost_per_hr: maxCost,
				preferred_gpu: preferredGpu,
			});
			message = 'Settings saved!';
			await fetchUser();
		} catch (e) {
			message = e instanceof Error ? e.message : 'Failed to save';
		}
		saving = false;
	}

	async function handleSetApiKey() {
		if (!apiKey) return;
		savingKey = true;
		message = '';
		try {
			await setVastaiKey(apiKey);
			apiKey = '';
			message = 'API key saved!';
			await fetchUser();
		} catch (e) {
			message = e instanceof Error ? e.message : 'Failed to save key';
		}
		savingKey = false;
	}

	async function handleDeleteApiKey() {
		if (!confirm('Remove your vast.ai API key?')) return;
		try {
			await deleteVastaiKey();
			message = 'API key removed.';
			await fetchUser();
		} catch (e) {
			message = e instanceof Error ? e.message : 'Failed to remove key';
		}
	}

	function handleLogout() {
		localStorage.removeItem('access_token');
		localStorage.removeItem('refresh_token');
		window.location.href = '/';
	}
</script>

<div class="max-w-xl mx-auto space-y-6">
	<h1 class="text-2xl font-bold">Settings</h1>

	{#if message}
		<div class="p-3 rounded bg-primary-500/20 text-primary-300 text-sm">{message}</div>
	{/if}

	<!-- Vast.ai API Key -->
	<div class="card p-6 bg-surface-800 space-y-4">
		<h2 class="text-lg font-semibold">Vast.ai API Key</h2>
		<p class="text-sm text-surface-400">
			{user?.has_vastai_key ? 'API key is stored.' : 'No API key stored.'}
		</p>

		<div class="flex gap-2">
			<input
				type="password"
				class="input flex-1"
				placeholder="Enter vast.ai API key"
				bind:value={apiKey}
			/>
			<button class="btn bg-primary-500 text-white" onclick={handleSetApiKey} disabled={savingKey}>
				{savingKey ? 'Saving...' : 'Save'}
			</button>
		</div>

		{#if user?.has_vastai_key}
			<button class="btn bg-error-500/20 text-error-300 text-sm" onclick={handleDeleteApiKey}>
				Remove API Key
			</button>
		{/if}
	</div>

	<!-- Instance Settings -->
	<div class="card p-6 bg-surface-800 space-y-4">
		<h2 class="text-lg font-semibold">Instance Defaults</h2>

		<label class="block">
			<span class="text-sm text-surface-400">Idle Timeout (minutes)</span>
			<input type="range" class="w-full mt-1" min="15" max="240" step="15" bind:value={idleTimeout} />
			<span class="text-sm">{idleTimeout} min</span>
		</label>

		<label class="block">
			<span class="text-sm text-surface-400">Max Cost ($/hr)</span>
			<input type="number" class="input mt-1" step="0.1" min="0.1" max="10" bind:value={maxCost} />
		</label>

		<label class="block">
			<span class="text-sm text-surface-400">Preferred GPU</span>
			<input type="text" class="input mt-1" placeholder="e.g. A100, RTX 4090 (leave empty for any)" bind:value={preferredGpu} />
		</label>

		<button class="btn bg-primary-500 text-white" onclick={handleSaveSettings} disabled={saving}>
			{saving ? 'Saving...' : 'Save Settings'}
		</button>
	</div>

	<!-- Logout -->
	<div class="card p-6 bg-surface-800">
		<button class="btn bg-surface-600 text-surface-200" onclick={handleLogout}>
			Logout
		</button>
	</div>
</div>
