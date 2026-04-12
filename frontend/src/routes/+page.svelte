<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { instanceStatus, startPolling, stopPolling } from '$lib/stores/instance';
	import { currentUser } from '$lib/stores/user';
	import { destroyInstance } from '$lib/api';
	import StatusBadge from '$lib/components/StatusBadge.svelte';

	const status = $derived($instanceStatus);
	const user = $derived($currentUser);

	let destroying = $state(false);

	onMount(() => {
		if (localStorage.getItem('access_token')) {
			startPolling(5000);
		}
	});

	onDestroy(() => stopPolling());

	async function handleDestroy() {
		if (!confirm('Destroy the GPU instance? This will stop billing.')) return;
		destroying = true;
		try {
			await destroyInstance();
		} catch (e) {
			alert(e instanceof Error ? e.message : 'Failed to destroy');
		}
		destroying = false;
	}
</script>

<div class="max-w-2xl mx-auto space-y-6">
	<h1 class="text-2xl font-bold">Dashboard</h1>

	{#if !user}
		<div class="card p-6 bg-surface-800">
			<p class="text-surface-400">Not logged in. <a href="/login" class="text-primary-400 underline">Login</a> to get started.</p>
		</div>
	{:else if !status || status.status === 'none'}
		<div class="card p-6 bg-surface-800 space-y-3">
			<h2 class="text-lg font-semibold">No GPU Instance</h2>
			<p class="text-surface-400">No instance running. <a href="/instances" class="text-primary-400 underline">Provision one</a> to get started.</p>
		</div>
	{:else}
		<div class="card p-6 bg-surface-800 space-y-4">
			<div class="flex items-center justify-between">
				<h2 class="text-lg font-semibold">GPU Instance</h2>
				<StatusBadge status={status.status} provisionState={status.provision_state} />
			</div>

			<div class="grid grid-cols-2 gap-4 text-sm">
				<div>
					<span class="text-surface-400">GPU</span>
					<p class="font-mono">{status.gpu_name || '—'}</p>
				</div>
				<div>
					<span class="text-surface-400">Cost</span>
					<p class="font-mono">${status.cost_per_hr?.toFixed(3)}/hr</p>
				</div>
				<div>
					<span class="text-surface-400">Status</span>
					<p class="font-mono">{status.status}</p>
				</div>
				<div>
					<span class="text-surface-400">Last Activity</span>
					<p class="font-mono">{status.last_activity ? new Date(status.last_activity).toLocaleTimeString() : '—'}</p>
				</div>
			</div>

			{#if status.status === 'running'}
				<button
					class="btn bg-error-500 text-white"
					onclick={handleDestroy}
					disabled={destroying}
				>
					{destroying ? 'Destroying...' : 'Destroy Instance'}
				</button>
			{/if}
		</div>
	{/if}
</div>
