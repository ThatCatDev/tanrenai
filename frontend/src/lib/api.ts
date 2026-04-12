const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:3000';

function getToken(): string | null {
	if (typeof window === 'undefined') return null;
	return localStorage.getItem('access_token');
}

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
	const token = getToken();
	const headers: Record<string, string> = {
		'Content-Type': 'application/json',
		...((options.headers as Record<string, string>) || {}),
	};
	if (token) {
		headers['Authorization'] = `Bearer ${token}`;
	}

	const resp = await fetch(`${API_BASE}${path}`, {
		...options,
		headers,
	});

	if (!resp.ok) {
		const body = await resp.text();
		throw new Error(`${resp.status}: ${body}`);
	}

	return resp.json();
}

// User
export interface User {
	id: string;
	email: string;
	name: string;
	has_vastai_key: boolean;
	idle_timeout_min: number;
	max_cost_per_hr: number;
	preferred_gpu: string;
}

export const getMe = () => apiFetch<User>('/api/user/me');

export const updateSettings = (settings: {
	idle_timeout_min: number;
	max_cost_per_hr: number;
	preferred_gpu: string;
}) => apiFetch('/api/user/settings', { method: 'PUT', body: JSON.stringify(settings) });

export const setVastaiKey = (apiKey: string) =>
	apiFetch('/api/user/vastai-key', { method: 'POST', body: JSON.stringify({ api_key: apiKey }) });

export const deleteVastaiKey = () =>
	apiFetch('/api/user/vastai-key', { method: 'DELETE' });

// Instance
export interface InstanceStatus {
	status: string;
	provision_state: string;
	gpu_name: string;
	gpu_url: string;
	cost_per_hr: number;
	model_loaded: string;
	created_at: string;
	last_activity: string;
}

export interface CostInfo {
	cost_per_hr: number;
	running_hours: number;
	total_cost: number;
	gpu_name: string;
}

export const getInstanceStatus = () => apiFetch<InstanceStatus>('/api/instance/status');

export const getInstanceCost = () => apiFetch<CostInfo>('/api/instance/cost');

export const provisionInstance = (params: {
	model_size: string;
	max_cost_per_hr?: number;
	gpu_name?: string;
}) => apiFetch('/api/instance/provision', { method: 'POST', body: JSON.stringify(params) });

export const destroyInstance = () =>
	apiFetch('/api/instance/destroy', { method: 'POST' });

// Health
export const getHealth = () => apiFetch<{ status: string }>('/health');
