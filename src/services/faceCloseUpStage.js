import * as faceapi from '@vladmandic/face-api';

/**
 * FaceCloseUpStage using face-api.js in-browser.
 * - Detects the largest face
 * - Applies padding
 * - Crops and resizes to fixed width
 */
class FaceCloseUpStage {
  /**
   * @param {number} outputWidth width of output canvas
   * @param {number} paddingFactor fraction of face box to pad
   * @param {string} modelUri base URI for face-api.js models
   * @param {boolean} debug enable verbose logging
   */
  constructor(outputWidth = 256, paddingFactor = 0.2, modelUri = '/models', debug = false) {
    this.outputWidth = outputWidth;
    this.paddingFactor = paddingFactor;
    this.modelUri = modelUri;
    this.modelsLoaded = false;
    this.debug = debug;
  }

  /** Load Tiny Face Detector model from public/models/ */
  // No external model load needed; face-landmarks from FaceMesh will be used
  async loadModels() {
    if (this.debug) console.log('[FaceCloseUpStage] Skipping model load (using FaceMesh landmarks)');
    this.modelsLoaded = true;
  }

  /**
   * Process an HTMLCanvasElement: crop, pad, and resize a face region
   * @param {HTMLCanvasElement} canvas current video frame
   * @param {{x:number,y:number,width:number,height:number}} faceBox face bounding box
   * @returns {string|null} JPEG dataURL of cropped face or null if none
   */
  async processCanvas(canvas, faceBox) {
    if (this.debug) console.log('[FaceCloseUpStage] processCanvas()');
    if (!faceBox) {
      if (this.debug) console.warn('[FaceCloseUpStage] No faceBox provided, skipping crop');
      return null;
    }
    if (this.debug) console.log(`[FaceCloseUpStage] Face box: x=${faceBox.x}, y=${faceBox.y}, w=${faceBox.width}, h=${faceBox.height}`);
    // Apply padding
    const padW = faceBox.width * this.paddingFactor;
    const padH = faceBox.height * this.paddingFactor;
    const x = Math.max(0, faceBox.x - padW);
    const y = Math.max(0, faceBox.y - padH);
    const w = Math.min(canvas.width, faceBox.x + faceBox.width + padW) - x;
    const h = Math.min(canvas.height, faceBox.y + faceBox.height + padH) - y;
    if (this.debug) console.log(`[FaceCloseUpStage] Crop rect: x=${x}, y=${y}, w=${w}, h=${h}`);
    // offscreen canvas
    const outH = Math.max(1, Math.round((h / w) * this.outputWidth));
    if (this.debug) console.log(`[FaceCloseUpStage] Output size: ${this.outputWidth}x${outH}`);
    const off = document.createElement('canvas');
    off.width = this.outputWidth;
    off.height = outH;
    const ctx = off.getContext('2d');
    ctx.drawImage(canvas, x, y, w, h, 0, 0, this.outputWidth, outH);
    const dataUrl = off.toDataURL('image/jpeg');
    if (this.debug) console.log('[FaceCloseUpStage] processCanvas() done');
    return dataUrl;
  }
}

export default FaceCloseUpStage;
