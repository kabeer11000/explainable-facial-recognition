// Assume opencv.js defines 'cv' globally.
// If you have @types/opencv-ts or similar, you can import types.
// Otherwise, declare it to avoid TypeScript errors for global 'cv'.
declare const cv: any;

// Extend Window interface to include OPENCV_READY flag
declare global {
  interface Window {
    OPENCV_READY: boolean;
  }
}

/**
 * Interface for a detected face, including its bounding box and detected eyes.
 */
export interface DetectedFace {
  rect: { x: number; y: number; width: number; height: number };
  eyes: { x: number; y: number; width: number; height: number }[];
}

/**
 * A class for performing face and eye detection using Haar Cascades with OpenCV.js.
 * It handles loading cascade classifiers and detecting features in image/video frames.
 */
export class HaarDetector {
  private faceCascade: any | null = null; // Stores cv.CascadeClassifier for faces
  private eyeCascade: any | null = null;  // Stores cv.CascadeClassifier for eyes
  private cascadeBasePath: string;        // Base path for the XML cascade files

  /**
   * Initializes the HaarDetector with a base path for cascade XML files.
   * @param cascadeBasePath The base URL path where your Haar cascade XML files are located.
   *                        Defaults to '/cascades/' if not provided.
   *                        Example: If files are at /public/cascades/haarcascade_frontalface_default.xml,
   *                        use '/cascades/'.
   */
  constructor(cascadeBasePath: string = '/cascades/') {
    // Ensure the base path ends with a slash for consistent URL construction
    this.cascadeBasePath = cascadeBasePath.endsWith('/') ? cascadeBasePath : cascadeBasePath + '/';
    console.log(`HaarDetector initialized with cascade base path: ${this.cascadeBasePath}`);
  }

  /**
   * Asynchronously initializes and loads the Haar Cascade classifiers.
   * This method must be called and awaited before calling `detect()`.
   * It checks for `window.OPENCV_READY` to ensure OpenCV.js is ready.
   * @returns {Promise<boolean>} True if classifiers are successfully loaded, false otherwise.
   */
  public async init(): Promise<boolean> {
    // Check if OpenCV.js is available and loaded
    if (typeof window === 'undefined' || !window.OPENCV_READY) {
      console.error("OpenCV.js is not loaded. Ensure 'window.OPENCV_READY' is true before calling init().");
      return false;
    }

    // If classifiers are already loaded, return true
    if (this.faceCascade && this.eyeCascade) {
      console.log("Haar cascade classifiers already initialized.");
      return true;
    }

    // Construct full paths to the cascade XML files
    const faceCascadeUrl = `${this.cascadeBasePath}haarcascade_frontalface_default.xml`;
    const eyeCascadeUrl = `${this.cascadeBasePath}haarcascade_eye.xml`;

    console.log(`Attempting to load face cascade from: ${faceCascadeUrl}`);
    console.log(`Attempting to load eye cascade from: ${eyeCascadeUrl}`);

    try {
      this.faceCascade = new cv.CascadeClassifier();
      this.eyeCascade = new cv.CascadeClassifier();

      // Load both cascades concurrently
      const [faceLoaded, eyeLoaded] = await Promise.all([
        this.loadCascade(this.faceCascade, faceCascadeUrl),
        this.loadCascade(this.eyeCascade, eyeCascadeUrl)
      ]);

      if (faceLoaded && eyeLoaded) {
        console.log("Successfully loaded Haar cascade classifiers.");
        return true;
      } else {
        console.error("Failed to load one or more Haar cascade classifiers. Check console for details.");
        // Clear references if loading failed
        this.dispose();
        return false;
      }
    } catch (error) {
      console.error("Error during Haar detector initialization:", error);
      this.dispose(); // Clean up if an error occurs during initialization
      return false;
    }
  }

  /**
   * Helper method to fetch an XML cascade file and load it into a cv.CascadeClassifier.
   * OpenCV.js requires files to be loaded into its virtual file system first.
   * @param classifier The cv.CascadeClassifier instance to load into.
   * @param url The URL to the cascade XML file.
   * @returns {Promise<boolean>} True if loaded successfully, false otherwise.
   */
  private async loadCascade(classifier: any, url: string): Promise<boolean> {
    const filename = url.split('/').pop(); // Extract filename from URL

    if (!filename) {
      console.error(`Invalid URL for cascade filename extraction: ${url}`);
      return false;
    }

    try {
      // 1. Fetch the XML file's content as an ArrayBuffer
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} from ${url}`);
      }
      const arrayBuffer = await response.arrayBuffer();

      // 2. Create a virtual file in OpenCV.js's Emscripten file system
      // Path: '/', Filename: filename, Data: Uint8Array, CanRead: true, CanWrite: false, CanDelete: false
      cv.FS_createDataFile('/', filename, new Uint8Array(arrayBuffer), true, false, false);

      // 3. Load the classifier using the filename from the virtual file system
      const success = classifier.load(filename);

      if (success) {
        console.log(`Successfully loaded cascade: ${filename}`);
        return true;
      } else {
        console.error(`Error: Could not load cascade classifier from virtual file system: ${filename}`);
        // Optionally delete the virtual file if loading failed
        cv.FS_unlink('/' + filename);
        return false;
      }
    } catch (error) {
      console.error(`Failed to fetch or load cascade from ${url}:`, error);
      return false;
    }
  }

  /**
   * Detects faces and eyes in an image or video frame.
   * Ensure `init()` has been successfully called before using this method.
   *
   * @param source The input HTML element (img, video, or canvas) from which to read the image data.
   * @returns {DetectedFace[]} An array of detected faces, each containing its rectangle and an array of eye rectangles.
   *                           Returns an empty array if classifiers are not loaded or an error occurs.
   */
  public detect(source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement): DetectedFace[] {
    if (!this.faceCascade || !this.eyeCascade) {
      console.error("Haar cascade classifiers are not loaded. Call init() first and await its completion.");
      return [];
    }

    // Allocate Mats (OpenCV.js image matrices) for processing
    let src = new cv.Mat(source.height, source.width, cv.CV_8UC4); // RGBA for HTML elements
    let gray = new cv.Mat();
    let faces = new cv.RectVector();
    let eyes = new cv.RectVector();
    let detectedFacesData: DetectedFace[] = [];

    try {
      // Read the image data from the HTML element into 'src' Mat
      cv.imread(source, src);

      // Convert to grayscale for detection
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

      // Detect faces in the grayscale image
      // Parameters: image, faces (output), scaleFactor, minNeighbors, flags, minSize, maxSize
      this.faceCascade.detectMultiScale(
        gray,
        faces,
        1.1, // Scale factor (how much the image size is reduced at each image scale)
        3,   // Minimum number of neighbors each candidate rectangle should have
        0,   // Flag for old cascade format (not commonly used with newer XMLs)
        new cv.Size(30, 30), // Minimum possible object size. Objects smaller than that are ignored.
        new cv.Size(src.cols, src.rows) // Maximum possible object size. Objects larger are ignored.
      );

      // Iterate through detected faces
      for (let i = 0; i < faces.size(); ++i) {
        const faceRect = faces.get(i);

        // Define the Region of Interest (ROI) for the current face in the grayscale image
        const roiGray = gray.roi(faceRect);

        const eyesInFace: { x: number; y: number; width: number; height: number }[] = [];

        // Detect eyes within the detected face region
        this.eyeCascade.detectMultiScale(
          roiGray,
          eyes,
          1.1,
          3,
          0,
          new cv.Size(15, 15), // Minimum eye size
          new cv.Size(roiGray.cols, roiGray.rows) // Maximum eye size (within face ROI)
        );

        // Iterate through detected eyes within the current face's ROI
        for (let j = 0; j < eyes.size(); ++j) {
          const eyeRect = eyes.get(j);
          // Convert eye coordinates from ROI-relative to original image-relative
          eyesInFace.push({
            x: faceRect.x + eyeRect.x,
            y: faceRect.y + eyeRect.y,
            width: eyeRect.width,
            height: eyeRect.height,
          });
        }

        // Store the detected face and its eyes
        detectedFacesData.push({
          rect: {
            x: faceRect.x,
            y: faceRect.y,
            width: faceRect.width,
            height: faceRect.height,
          },
          eyes: eyesInFace,
        });
        console.log(detectedFacesData);
        roiGray.delete(); // Release memory for the ROI Mat
      }
    } catch (error) {
      console.error("Error during face and eye detection:", error);
      // It's crucial to delete allocated Mats even if an error occurs
    } finally {
      // Always release memory for OpenCV.js Mats and Vectors
      src.delete();
      gray.delete();
      faces.delete();
      eyes.delete();
    }

    return detectedFacesData;
  }

  /**
   * Disposes of the loaded cascade classifiers and releases their memory.
   * Call this when the detector is no longer needed to prevent memory leaks.
   */
  public dispose(): void {
    if (this.faceCascade) {
      this.faceCascade.delete();
      this.faceCascade = null;
    }
    if (this.eyeCascade) {
      this.eyeCascade.delete();
      this.eyeCascade = null;
    }
    console.log("Haar cascade classifiers disposed.");
  }
}