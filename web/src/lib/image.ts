export async function imageBlobToBase64(imageUrl: string) {
    try {
        const response = await fetch(imageUrl);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const blob = await response.blob();

        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                // reader.result will be a Data URL (e.g., "data:image/jpeg;base64,...")
                // We need to extract just the base64 part
                const base64String = reader?.result?.split(',')[1];
                resolve(base64String);
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob); // This reads the blob as a Data URL
        });
    } catch (error) {
        console.error("Error fetching or converting image:", error);
        return null; // Or throw the error, depending on your error handling strategy
    }
}
export function encodeFileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            resolve(reader.result); // reader.result contains the data URL (base64 string)
        };
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file); // Read the file as a Data URL
    });
}

export function imageDataToBase64(imageData:ImageData, mimeType = 'image/png', quality = 0.92) {
    if (!(imageData instanceof ImageData)) {
        console.error("Input is not an ImageData object.");
        return null;
    }

    // 1. Create a new, temporary canvas element
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');

    if (!tempCtx) {
        console.error("Could not get 2D rendering context for temporary canvas.");
        return null;
    }

    // 2. Set the dimensions of the temporary canvas to match the ImageData
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;

    // 3. Put the ImageData onto the temporary canvas
    tempCtx.putImageData(imageData, 0, 0);

    // 4. Convert the temporary canvas content to a Base64 data URL
    try {
        const dataURL = tempCanvas.toDataURL(mimeType, quality);
        return dataURL?.split(',')[1];
    } catch (error) {
        console.error("Error converting canvas to data URL:", error);
        return null;
    }
}
