import { writable } from 'svelte/store';
import { getMe, type User } from '$lib/api';

export const currentUser = writable<User | null>(null);
export const userLoading = writable(false);

export async function fetchUser() {
	try {
		userLoading.set(true);
		const user = await getMe();
		currentUser.set(user);
	} catch {
		currentUser.set(null);
	} finally {
		userLoading.set(false);
	}
}
