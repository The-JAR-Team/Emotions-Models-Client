// src/config/modelConfig.js
const MODEL_CONFIGS = {
  "ferplus_transformer_small_v1": {
    id: "ferplus_transformer_small_v1",
    name: "Emotion Transformer Small v1 (FER+ 8 Classes)",
    filename: "emotion_transformer_small.onnx", // New model filename
    processingOptions: {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    },
    inputFormat: {
      sequenceLength: 1, // Model processes one frame (set of landmarks) at a time
      numLandmarks: 478, // MediaPipe FaceMesh outputs 478 landmarks
      numCoords: 3,      // x, y, z coordinates
      tensorShape: [1, 478, 3], // Expected input tensor shape [batch_size, num_landmarks, num_coords]
      requiresNormalization: true, // This model expects specific normalization
    },
    normalizationType: 'ferplus', // Specify that this model uses FER+ style normalization
    outputFormat: {
      outputType: 'classification',
      numClasses: 8,
      classLabels: { // FER+ 8 emotion classes
        0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness',
        4: 'Anger', 5: 'Disgust', 6: 'Fear', 7: 'Contempt'
      },
      outputNames: { logits: 'logits', embedding: 'embedding' }, // Model outputs both
      applySoftmax: true // The Python script model outputs logits, so apply softmax in client
    }
  },
  "emotion_transformer_v1": {
    id: "emotion_transformer_v1",
    name: "Emotion Transformer v1 (AffectNet 8 Classes)",
    filename: "emotion_transformer_v1.onnx",
    processingOptions: { // Added default processingOptions
      executionProviders: ['wasm'], // Default to WASM
      graphOptimizationLevel: 'all'
    },
    inputFormat: {
      sequenceLength: 1,
      numLandmarks: 478,
      numCoords: 3,
      tensorShape: [1, 478, 3], // Adjusted for single frame, 478 landmarks, 3 coords
      requiresNormalization: false, // Disabled normalization per user request
    },
    outputFormat: {
      outputType: 'classification', // Model primarily outputs classification logits
      numClasses: 8,
      classLabels: { // Based on AffectNet 8 classes
        0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Surprise',
        4: 'Fear', 5: 'Disgust', 6: 'Anger', 7: 'Contempt'
      },
      outputNames: { logits: 'logits', embedding: 'embedding' } // Specify output names
    }
  }
  // Add other model configurations here if needed
};

let currentActiveModelId = "ferplus_transformer_small_v1"; // Default to the new FERPlus model

export const getActiveModelConfig = () => {
  return MODEL_CONFIGS[currentActiveModelId];
};

export const getModelConfig = (modelId) => {
  return MODEL_CONFIGS[modelId];
};

export const setActiveModel = (modelId) => {
  if (MODEL_CONFIGS[modelId]) {
    currentActiveModelId = modelId;
    console.log(`Switched active model to: ${modelId}`);
    return true;
  }
  console.warn(`Model ID '${modelId}' not found in configurations.`);
  return false;
};

export const getAllModelConfigs = () => {
  return Object.values(MODEL_CONFIGS); // Return an array of model config objects
};

export const getModelPaths = (filename) => {
  // Define potential paths for Vite to find the model in the public directory
  // These paths are relative to the domain root when served.
  return [
    `/models/${filename}`,         // Standard for Vite public assets
    `./models/${filename}`,        // Relative path
    // Add any other paths if your deployment structure is different
  ];
};
