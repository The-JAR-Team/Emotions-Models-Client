// src/config/modelConfig.js
const MODEL_CONFIGS = {
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

let currentActiveModelId = "emotion_transformer_v1"; // Default model

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
  return MODEL_CONFIGS;
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
