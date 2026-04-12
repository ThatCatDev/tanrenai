<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { instanceStatus, startPolling, stopPolling } from '$lib/stores/instance';
	import { provisionInstance, destroyInstance } from '$lib/api';
	import StatusBadge from '$lib/components/StatusBadge.svelte';

	let modelSize = $state('');
	let maxCost = $state(1.0);
	let gpuPref = $state('');
	let provisioning = $state(false);
	let destroying = $state(false);
	let errorMsg = $state('');

	const status = $derived($instanceStatus);

	onMount(() => startPolling(3000));
	onDestroy(() => stopPolling());

	async function handleProvision() {
		if (!modelSize) {
			errorMsg = 'Model size is required (e.g. 8b, 72b, 120b)';
			return;
		}
		errorMsg = '';
		provisioning = true;
		try {
			await provisionInstance({
				model_size: modelSize,
				max_cost_per_hr: maxCost,
				gpu_name: gpuPref || undefined,
			});
		} catch (e) {
			errorMsg = e instanceof Error ? e.message : 'Provisioning failed';
		}
		provisioning = false;
	}

	async function handleDestroy() {
		if (!confirm('Destroy this instance?')) return;
		destroying = true;
		try {
			await destroyInstance();
		} catch (e) {
			errorMsg = e instanceof Error ? e.message : 'Destroy failed';
		}
		destroying = false;
	}
</script>

<div class="max-w-2xl mx-auto space-y-6">
	<h1 class="text-2xl font-bold">GPU Instances</h1>

	<!-- Current Instance -->
	{#if status && status.status !== 'none'}
		<div class="card p-6 bg-surface-800 space-y-4">
			<div class="flex items-center justify-between">
				<h2 class="text-lg font-semibold">Current Instance</h2>
				<StatusBadge status={status.status} provisionState={status.provision_state} />
			</div>

			<div class="grid grid-cols-2 gap-3 text-sm">
				<div><span class="text-surface-400">GPU</span><p>{status.gpu_name || '—'}</p></div>
				<div><span class="text-surface-400">Cost</span><p>${status.cost_per_hr?.toFixed(3)}/hr</p></div>
				<div><span class="text-surface-400">URL</span><p class="font-mono text-xs break-all">{status.gpu_url || '—'}</p></div>
				<div><span class="text-surface-400">Model</span><p>{status.model_loaded || '—'}</p></div>
			</div>

			{#if status.status === 'running' || status.status === 'provisioning'}
				<button class="btn bg-error-500 text-white" onclick={handleDestroy} disabled={destroying}>
					{destroying ? 'Destroying...' : 'Destroy'}
				</button>
			{/if}
		</div>
	{/if}

	<!-- Provision Form -->
	<div class="card p-6 bg-surface-800 space-y-4">
		<h2 class="text-lg font-semibold">Provision New Instance</h2>

		{#if errorMsg}
			<div class="p-3 rounded bg-error-500/20 text-error-300 text-sm">{errorMsg}</div>
		{/if}

		<label class="block">
			<span class="text-sm text-surface-400">Model Size</span>
			<input
				type="text"
				class="input mt-1"
				placeholder="e.g. 8b, 72b, 120b"
				bind:value={modelSize}
			/>
		</label>

		<label class="block">
			<span class="text-sm text-surface-400">Max Cost ($/hr)</span>
			<input
				type="number"
				class="input mt-1"
				step="0.1"
				min="0.1"
				bind:value={maxCost}
			/>
		</label>

		<label class="block">
			<span class="text-sm text-surface-400">GPU Preference (optional)</span>
			<input
				type="text"
				class="input mt-1"
				placeholder="e.g. A100, RTX 4090"
				bind:value={gpuPref}
			/>
		</label>

		<button
			class="btn bg-primary-500 text-white"
			onclick={handleProvision}
			disabled={provisioning}
		>
			{provisioning ? 'Provisioning...' : 'Provision Instance'}
		</button>
	</div>
</div>
