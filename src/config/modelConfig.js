// src/config/modelConfig.js
const MODEL_CONFIGS = {
  "ferplus_transformer_small_v1": {
    id: "ferplus_transformer_small_v1",
    name: "Emotion Transformer Small v1 (FER+ 8 Classes)",
    filename: "emotion_transformer_smallV1.onnx", // Updated to match public/models filename
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
  "ferplus_transformer_small_v2": {
    id: "ferplus_transformer_small_v2",
    name: "Emotion Transformer Small v2 (FER+ 8 Classes)",
    filename: "emotion_transformer_smallV2.onnx",
    processingOptions: {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    },
    inputFormat: {
      sequenceLength: 1,
      numLandmarks: 478,
      numCoords: 3,
      tensorShape: [1, 478, 3],
      requiresNormalization: true
    },
    normalizationType: 'ferplus',
    outputFormat: {
      outputType: 'classification',
      numClasses: 8,
      classLabels: {
        0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness',
        4: 'Anger', 5: 'Disgust', 6: 'Fear', 7: 'Contempt'
      },
      outputNames: { logits: 'logits', embedding: 'embedding' },
      applySoftmax: true
    }
  },
  // Add other model configurations here if needed
};

// Default to the latest FERPlus small model version
let currentActiveModelId = "ferplus_transformer_small_v2";

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
