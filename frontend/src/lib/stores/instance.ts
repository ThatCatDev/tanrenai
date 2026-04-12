import { writable } from 'svelte/store';
import { getInstanceStatus, type InstanceStatus } from '$lib/api';

export const instanceStatus = writable<InstanceStatus | null>(null);
export const instanceLoading = writable(false);
export const instanceError = writable<string | null>(null);

let pollInterval: ReturnType<typeof setInterval> | null = null;

export function startPolling(intervalMs = 5000) {
	stopPolling();
	fetchStatus();
	pollInterval = setInterval(fetchStatus, intervalMs);
}

export function stopPolling() {
	if (pollInterval) {
		clearInterval(pollInterval);
		pollInterval = null;
	}
}

async function fetchStatus() {
	try {
		instanceLoading.set(true);
		const status = await getInstanceStatus();
		instanceStatus.set(status);
		instanceError.set(null);
	} catch (e) {
		instanceError.set(e instanceof Error ? e.message : 'Unknown error');
	} finally {
		instanceLoading.set(false);
	}
}
