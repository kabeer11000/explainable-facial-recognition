import { atom } from 'nanostores';

export const $PersonInFrame = atom<boolean>(false);
export const $ImageData = atom<ImageData | null>(null);


export const $PersonResponse = atom<{label: string, confidence: number} | null>(null);