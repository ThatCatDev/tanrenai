<script lang="ts">
	let { status, provisionState = '' }: { status: string; provisionState?: string } = $props();

	const colors: Record<string, string> = {
		running: 'bg-success-500',
		provisioning: 'bg-warning-500',
		destroying: 'bg-error-500',
		destroyed: 'bg-surface-600',
		none: 'bg-surface-600',
		pending: 'bg-surface-500',
	};

	const color = $derived(colors[status] || 'bg-surface-500');
	const label = $derived(
		status === 'provisioning' && provisionState
			? `${status} (${provisionState})`
			: status
	);
</script>

<span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium text-white {color}">
	<span class="w-1.5 h-1.5 rounded-full bg-white/80"></span>
	{label}
</span>
