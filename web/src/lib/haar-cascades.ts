// src/components/WebcamFaceDetector.js

/**
 * @typedef {object} Rect
 * @property {number} x
 * @property {number} y
 * @property {number} width
 * @property {number} height
 */

/**
 * @typedef {object} FaceDetection
 * @property {Rect} rect
 * @property {Rect[]} eyes
 */

export class HAARDetectorWrapper {
    /**
     * @param {string} videoElementId - ID of the <video> element.
     * @param {string} canvasElementId - ID of the <canvas> element.
     * @param {string} statusElementId - ID of the element to display status messages.
     * @param {string} startButtonId - ID of the start camera button.
     * @param {string} stopButtonId - ID of the stop camera button.
     * @param {string} workerScriptUrl - URL to the Web Worker script (e.g., '/workers/opencvWorker.js').
     */
    constructor(videoElementId, canvasElementId, statusElementId, startButtonId, stopButtonId, workerScriptUrl) {
        this.videoInput = document.getElementById(videoElementId);
        this.canvasOutput = document.getElementById(canvasElementId);
        this.statusElement = document.getElementById(statusElementId);
        this.startButton = document.getElementById(startButtonId);
        this.stopButton = document.getElementById(stopButtonId);
        this.workerScriptUrl = workerScriptUrl;

        this.ctx = this.canvasOutput.getContext('2d');
        this.stream = null;
        this.worker = null;
        this.animationFrameId = null;
        this.processing = false; // Flag to prevent sending frames too fast

        // Bind event handlers to the class instance
        this.startCamera = this.startCamera.bind(this);
        this.stopCamera = this.stopCamera.bind(this);
        this.processFrame = this.processFrame.bind(this);
        this.handleWorkerMessage = this.handleWorkerMessage.bind(this);
        this.handleWorkerError = this.handleWorkerError.bind(this);
    }

    /**
     * Initializes the detector by setting up event listeners and the Web Worker.
     */
    init() {
        this.updateStatus('Initializing detector...');
        this.startButton.addEventListener('click', this.startCamera);
        this.stopButton.addEventListener('click', this.stopCamera);

        // Initialize Web Worker
        this.initializeWorker();

        // Clean up on page unload
        window.addEventListener('beforeunload', this.stopCamera);
    }

    /**
     * Initializes the Web Worker and sets up its message listeners.
     */
    initializeWorker() {
        this.updateStatus('Initializing Web Worker...');
        // Create a classic worker (not module) to allow importScripts for opencv.js
        this.worker = new Worker(this.workerScriptUrl); 

        this.worker.onmessage = this.handleWorkerMessage;
        this.worker.onerror = this.handleWorkerError;
    }

    /**
     * Handles messages received from the Web Worker.
     * @param {MessageEvent} event
     */
    handleWorkerMessage(event) {
        if (event.data.type === 'ready') {
            this.updateStatus('OpenCV.js and Haar Cascades ready in worker.');
            this.startButton.disabled = false;
            this.stopButton.disabled = true;
        } else if (event.data.type === 'detectionResult') {
            const { detections, width, height } = event.data;
            this.processing = false; // Reset flag after worker is done

            this.drawDetections(detections, width, height);

            // Send detection results via HTTP
            this.sendDetectionResults(detections);

        } else if (event.data.type === 'error') {
            this.updateStatus(`Worker Error: ${event.data.message}`);
            console.error('Worker Error:', event.data.message);
            this.stopCamera(); // Attempt to stop camera on worker error
        }
    }

    /**
     * Handles errors from the Web Worker.
     * @param {ErrorEvent} error
     */
    handleWorkerError(error) {
        this.updateStatus(`Worker Load Error: ${error.message}`);
        console.error('Worker Load Error:', error);
        this.stopCamera();
    }

    /**
     * Starts the camera stream and begins frame processing.
     */
    async startCamera() {
        try {
            this.updateStatus('Requesting camera access...');
            this.stream = await navigator.mediaDevices.getUserMedia({ video: true });
            this.videoInput.srcObject = this.stream;
            await this.videoInput.play();

            this.updateStatus('Camera started. Processing frames...');
            this.startButton.disabled = true;
            this.stopButton.disabled = false;

            // Ensure canvas matches video dimensions
            this.canvasOutput.width = this.videoInput.videoWidth;
            this.canvasOutput.height = this.videoInput.videoHeight;
            
            // Start sending frames to worker
            this.animationFrameId = requestAnimationFrame(this.processFrame);

        } catch (err) {
            this.updateStatus(`Error accessing camera: ${err.name} - ${err.message}`);
            console.error('Error accessing camera:', err);
            this.startButton.disabled = false;
            this.stopButton.disabled = true;
        }
    }

    /**
     * Stops the camera stream and terminates the Web Worker.
     */
    stopCamera() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.videoInput.srcObject = null;
            this.stream = null;
        }
        if (this.worker) {
            this.worker.postMessage({ type: 'terminate' }); // Tell worker to clean up
            this.worker.terminate(); // Terminate the worker
            this.worker = null;
            // Re-initialize worker if needed for a new start
            this.initializeWorker(); 
        }
        this.ctx.clearRect(0, 0, this.canvasOutput.width, this.canvasOutput.height); // Clear canvas
        this.processing = false;
        this.updateStatus('Camera stopped.');
        this.startButton.disabled = false;
        this.stopButton.disabled = true;
    }

    /**
     * Captures a frame from the video, sends it to the worker, and schedules the next frame.
     */
    processFrame() {
        if (!this.videoInput.paused && !this.videoInput.ended && this.worker && !this.processing) {
            this.processing = true; // Set flag to true

            // Draw current video frame to canvas
            this.ctx.drawImage(this.videoInput, 0, 0, this.canvasOutput.width, this.canvasOutput.height);
            const imageData = this.ctx.getImageData(0, 0, this.canvasOutput.width, this.canvasOutput.height);

            // Send ImageData to worker as a transferable object for efficiency
            this.worker.postMessage({
                type: 'processFrame',
                imageData: imageData,
                width: this.canvasOutput.width,
                height: this.canvasOutput.height
            }, [imageData.data.buffer]); // Transferable object
        }
        this.animationFrameId = requestAnimationFrame(this.processFrame);
    }

    /**
     * Sends detection results to a backend API via HTTP POST.
     * @param {FaceDetection[]} results - Array of detected faces and eyes.
     */
    async sendDetectionResults(results) {
        if (results.length === 0) {
            // console.log("No faces or eyes detected, not sending data.");
            return;
        }
        try {
            // Replace with your actual API endpoint
            const apiUrl = 'https://your-api-endpoint.com/detections'; 
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    timestamp: new Date().toISOString(),
                    detections: results
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            console.log('Detection results sent successfully:', data);
        } catch (error) {
            console.error('Error sending detection results:', error);
        }
    }

    /**
     * Draws the detected faces and eyes on the canvas.
     * @param {FaceDetection[]} detections - Array of detected faces and eyes.
     * @param {number} width - Width of the canvas.
     * @param {number} height - Height of the canvas.
     */
    drawDetections(detections, width, height) {
        // Clear canvas and redraw video frame
        this.ctx.clearRect(0, 0, width, height);
        this.ctx.drawImage(this.videoInput, 0, 0, width, height); 

        detections.forEach(face => {
            const { rect, eyes } = face;
            // Draw face rectangle
            this.ctx.strokeStyle = 'red';
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);

            // Draw eye rectangles
            eyes.forEach(eye => {
                this.ctx.strokeStyle = 'lime';
                this.ctx.lineWidth = 1;
                // Eye coordinates are relative to the face ROI, so add face's x,y
                this.ctx.strokeRect(rect.x + eye.x, rect.y + eye.y, eye.width, eye.height);
            });
        });
    }

    /**
     * Updates the status message displayed on the UI.
     * @param {string} message - The message to display.
     */
    updateStatus(message) {
        if (this.statusElement) {
            this.statusElement.textContent = message;
        }
        console.log(`Status: ${message}`);
    }
}