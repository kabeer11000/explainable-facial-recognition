import { atom } from 'nanostores';

/**
 * @typedef {Object} ConnectionStatus
 * @property {boolean} connected
 * @property {string} statusMessage
 */

export const $ConnectionStatus = atom<{
    connected: boolean;
    statusMessage: string; // Add statusMessage as a prop  
}>({
    connected: false,
    statusMessage: ''
});