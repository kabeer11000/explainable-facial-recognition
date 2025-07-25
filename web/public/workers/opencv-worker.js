importScripts('/opencv/opencv.js');

let faceCascade = null;
let eyeCascade = null;
let cv = null; // Reference to the OpenCV.js object

// Utility function to load cascade files
async function loadCascadeFile(url, fileName) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to download ${url}: ${response.statusText}`);
        }
        const buffer = await response.arrayBuffer();
        const data = new Uint8Array(buffer);
        cv.FS_createDataFile('/', fileName, data, true, false, false);
        console.log(`Worker: Downloaded ${fileName}.`);
    } catch (error) {
        console.error(`Worker: Error loading cascade file ${fileName}:`, error);
        throw error;
    }
}

// Function to initialize OpenCV.js and load cascades inside the worker
async function initializeOpenCV() {
    try {
        // Wait for OpenCV.js to be ready
        // In a worker, `cv` might be a Promise initially, so await it.
        cv = await self.cv;

        // Load cascade files from the public directory (relative to worker's URL)
        const faceCascadeUrl = '/haarcascade_frontalface_default.xml';
        const eyeCascadeUrl = '/haarcascade_eye.xml';

        await loadCascadeFile(faceCascadeUrl, 'face_cascade_file');
        faceCascade = new cv.CascadeClassifier();
        if (!faceCascade.load('face_cascade_file')) {
            throw new Error('Worker: Could not load face cascade classifier.');
        }

        await loadCascadeFile(eyeCascadeUrl, 'eye_cascade_file');
        eyeCascade = new cv.CascadeClassifier();
        if (!eyeCascade.load('eye_cascade_file')) {
            throw new Error('Worker: Could not load eye cascade classifier.');
        }

        console.log("Worker: Haar cascade classifiers initialized.");
        postMessage({ type: 'ready' }); // Notify main thread that worker is ready
    } catch (error) {
        console.error("Worker: Error initializing OpenCV:", error);
        postMessage({ type: 'error', message: error.message });
        // Clean up if initialization fails
        if (faceCascade) faceCascade.delete();
        if (eyeCascade) eyeCascade.delete();
        faceCascade = null;
        eyeCascade = null;
        cv = null;
    }
}

// Listen for messages from the main thread
self.onmessage = (event) => {
    if (event.data.type === 'processFrame') {
        const { imageData, width, height } = event.data;

        if (!cv || !faceCascade || !eyeCascade) {
            console.warn("Worker: OpenCV or cascades not ready for processing.");
            postMessage({ type: 'error', message: 'OpenCV or cascades not ready.' });
            return;
        }

        // Create cv.Mat from ImageData
        // ImageData is RGBA, so use CV_8UC4
        let src = new cv.Mat(height, width, cv.CV_8UC4);
        src.data.set(imageData.data); // Copy pixel data

        let gray = new cv.Mat();
        let faces = new cv.RectVector();
        let eyes = new cv.RectVector();

        try {
            // Convert to grayscale
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

            // Detect faces
            faceCascade.detectMultiScale(gray, faces, 1.3, 5, 0, new cv.Size(30, 30), new cv.Size());

            const detectedFacesData = [];

            for (let i = 0; i < faces.size(); ++i) {
                let faceRect = faces.get(i);
                
                // Region of interest for eye detection
                let roiGray = gray.roi(faceRect);

                let eyesInFace = [];
                // Detect eyes within the face ROI
                eyeCascade.detectMultiScale(roiGray, eyes, 1.1, 5, 0, new cv.Size(15, 15), new cv.Size());

                for (let j = 0; j < eyes.size(); ++j) {
                    let eyeRect = eyes.get(j);
                    eyesInFace.push({ x: eyeRect.x, y: eyeRect.y, width: eyeRect.width, height: eyeRect.height });
                }

                detectedFacesData.push({
                    rect: { x: faceRect.x, y: faceRect.y, width: faceRect.width, height: faceRect.height },
                    eyes: eyesInFace
                });

                // Release ROI memory
                roiGray.delete();
            }

            // Send results back to main thread
            postMessage({
                type: 'detectionResult',
                detections: detectedFacesData,
                width: width,
                height: height
            });

        } catch (e) {
            console.error("Worker: Error during detection:", e);
            postMessage({ type: 'error', message: `Detection error: ${e.message}` });
        } finally {
            // IMPORTANT: Release memory
            src.delete();
            gray.delete();
            faces.delete();
            eyes.delete();
        }
    } else if (event.data.type === 'terminate') {
        // Clean up resources when told to terminate
        if (faceCascade) faceCascade.delete();
        if (eyeCascade) eyeCascade.delete();
        faceCascade = null;
        eyeCascade = null;
        cv = null;
        console.log("Worker: Terminated and cleaned up.");
        self.close(); // Close the worker
    }
};

// Start initialization when the worker script loads
initializeOpenCV();