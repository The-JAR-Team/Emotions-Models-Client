import * as ort from 'onnxruntime-web';
import { getOnnxModelUri } from './onnxModelLoader';
import { fetchModelFromHooks } from './onnxHooksFallback';
import { getActiveModelConfig, getModelPaths, getModelConfig as getModelConfigUtil, setActiveModel as setActiveModelUtil, getAllModelConfigs as getAllModelConfigsUtil } from '../config/modelConfig'; // Renamed imports to avoid conflict

// Reference landmark indices for distance normalization (specific to this service's normalization logic)
// These might differ from MediaPipe's own indices if custom normalization is applied.
// The Python script uses: Nose: 1, Left Eye Inner: 133, Right Eye Inner: 362 for its normalization.
// This service uses: Nose: 1, Left Eye Outer: 33, Right Eye Outer: 263.
// Ensure this matches the preprocessing expected by the model or adjust.
// For the model from the Python script, its internal normalization uses 1, 133, 362.
// If preprocessLandmarks here is used, it should align.
const NOSE_TIP_IDX = 1; // As per this service's applyDistanceNormalization
const LEFT_EYE_OUTER_IDX = 33; // As per this service's applyDistanceNormalization
const RIGHT_EYE_OUTER_IDX = 263; // As per this service's applyDistanceNormalization

let onnxSession = null;
let currentModelConfig = null; // Stores the config of the currently loaded model

const getModelDimensions = () => {
  const config = currentModelConfig || getActiveModelConfig(); // Use loaded model's config first
  return {
    SEQ_LEN: config.inputFormat.sequenceLength,
    NUM_LANDMARKS: config.inputFormat.numLandmarks,
    NUM_COORDS: config.inputFormat.numCoords
  };
};

export const initializeOnnxModel = async (modelId = null) => {
  // Explicitly set the base path for ONNX runtime WASM and MJS files to CDN
  // This avoids serving local WASM/MJS from Vite public, preventing import errors
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

  try {
    if (modelId) {
      if (!setActiveModelUtil(modelId)) {
        console.error(`Failed to set active model to ${modelId}. It might not exist.`);
        // Fallback to current active or default if setting fails
        currentModelConfig = getActiveModelConfig();
      } else {
        currentModelConfig = getModelConfigUtil(modelId);
      }
    } else {
      currentModelConfig = getActiveModelConfig();
    }

    if (!currentModelConfig) {
        throw new Error("No model configuration found.");
    }
    
    console.log(`Initializing ONNX model: ${currentModelConfig.name} (ID: ${currentModelConfig.id})`);
    
    const options = {
      executionProviders: currentModelConfig.processingOptions.executionProviders,
      graphOptimizationLevel: currentModelConfig.processingOptions.graphOptimizationLevel
    };
    
    const possiblePaths = getModelPaths(currentModelConfig.filename);
    let modelLoaded = false;
    let lastError = null;

    console.log("Attempting to load model using getOnnxModelUri...");
    try {
      const modelUri = await getOnnxModelUri(currentModelConfig.filename);
      if (modelUri) {
        onnxSession = await ort.InferenceSession.create(modelUri, options);
        console.log('ONNX model loaded successfully via getOnnxModelUri.');
        URL.revokeObjectURL(modelUri); // Clean up blob URL
        modelLoaded = true;
      } else {
        console.warn('getOnnxModelUri did not return a URI.');
      }
    } catch (error) {
      console.warn(`Failed to load model from getOnnxModelUri (blob URL): ${error.message}`);
      lastError = error;
    }
    
    if (!modelLoaded) {
      console.log("Attempting to load model using fetchModelFromHooks...");
      try {
        const hooksModelUri = await fetchModelFromHooks(currentModelConfig.filename);
        if (hooksModelUri) {
          onnxSession = await ort.InferenceSession.create(hooksModelUri, options);
          console.log('ONNX model loaded successfully via fetchModelFromHooks.');
          URL.revokeObjectURL(hooksModelUri); // Clean up blob URL
          modelLoaded = true;
        } else {
          console.warn('fetchModelFromHooks did not return a URI.');
        }
      } catch (error) {
        console.warn(`Failed to load model from fetchModelFromHooks (blob URL): ${error.message}`);
        lastError = error;
      }
    }
    
    if (!modelLoaded) {
      console.log("Attempting to load model from direct paths...");
      for (const path of possiblePaths) {
        try {
          console.log(`Trying direct path: ${path}`);
          onnxSession = await ort.InferenceSession.create(path, options);
          console.log(`ONNX model loaded successfully from direct path: ${path}`);
          modelLoaded = true;
          break;
        } catch (error) {
          console.warn(`Failed to load model from ${path}: ${error.message}`);
          lastError = error;
        }
      }
    }
    
    if (!modelLoaded) {
      throw lastError || new Error('Failed to load model from any of the possible methods/paths');
    }
    
    console.log("ONNX Session Input Names:", onnxSession.inputNames);
    console.log("ONNX Session Output Names:", onnxSession.outputNames);
    // Log class labels mapping for verification
    console.log("ONNX Model Class Labels:", currentModelConfig.outputFormat.classLabels);
    return true;
  } catch (error) {
    console.error('Failed to initialize ONNX model:', error);
    onnxSession = null; // Ensure session is null on failure
    currentModelConfig = null;
    return false;
  }
};

export const preprocessLandmarks = (landmarks) => {
  // Convert single-frame landmarks (array of objects) into an array of frames
  const frames = Array.isArray(landmarks) && landmarks.length > 0 && typeof landmarks[0].x === 'number'
    ? [landmarks]
    : Array.isArray(landmarks)
      ? landmarks
      : [];

  const { SEQ_LEN, NUM_LANDMARKS, NUM_COORDS } = getModelDimensions();
  const processedFrames = [];

  for (let i = 0; i < SEQ_LEN; i++) {
    const frame = frames[i] || [];
    // Initialize all coords to -1
    const frameArray = Array.from({ length: NUM_LANDMARKS }, () => Array(NUM_COORDS).fill(-1.0));
    // Fill with actual landmark coords
    for (let j = 0; j < Math.min(frame.length, NUM_LANDMARKS); j++) {
      const lm = frame[j];
      if (lm && typeof lm.x === 'number' && typeof lm.y === 'number' && typeof lm.z === 'number') {
        frameArray[j][0] = lm.x;
        frameArray[j][1] = lm.y;
        frameArray[j][2] = lm.z;
      }
    }
    processedFrames.push(frameArray);
  }

  const activeConfig = currentModelConfig || getActiveModelConfig();
  let finalFrames = processedFrames;
  if (activeConfig.inputFormat.requiresNormalization) {
    finalFrames = applyDistanceNormalization(processedFrames);
  }

  // Flatten to 1D Float32Array
  const flat = finalFrames.flat(2);
  return new Float32Array(flat);
};

export const applyDistanceNormalization = (landmarksFrames) => {
  // This normalization is specific to this service.
  // It uses NOSE_TIP_IDX=1, LEFT_EYE_OUTER_IDX=33, RIGHT_EYE_OUTER_IDX=263.
  // The Python training script uses different landmarks for normalization:
  // Nose: 1, Left Eye Inner: 133, Right Eye Inner: 362.
  // This discrepancy can lead to poor model performance if not aligned.
  const normalizedFrames = [];
  for (let frameIdx = 0; frameIdx < landmarksFrames.length; frameIdx++) {
    const frameLandmarks = landmarksFrames[frameIdx];
    const isEntireFrameInvalid = frameLandmarks.every(coords => coords.every(val => val === -1.0));
    if (isEntireFrameInvalid) {
      normalizedFrames.push(frameLandmarks);
      continue;
    }
    try {
      const centerLandmarkCoords = frameLandmarks[NOSE_TIP_IDX];
      const p1Coords = frameLandmarks[LEFT_EYE_OUTER_IDX];
      const p2Coords = frameLandmarks[RIGHT_EYE_OUTER_IDX];
      
      const isInvalid = 
        centerLandmarkCoords.some(val => val === -1.0) ||
        p1Coords.some(val => val === -1.0) ||
        p2Coords.some(val => val === -1.0);
      
      if (isInvalid) {
        normalizedFrames.push(frameLandmarks);
        continue;
      }
      
      const translatedLandmarks = frameLandmarks.map(coords => {
        if (coords.some(val => val === -1.0)) return coords;
        return [
          coords[0] - centerLandmarkCoords[0],
          coords[1] - centerLandmarkCoords[1],
          coords[2] - centerLandmarkCoords[2]
        ];
      });
      
      const scaleDistance = Math.sqrt(
        Math.pow(p1Coords[0] - p2Coords[0], 2) +
        Math.pow(p1Coords[1] - p2Coords[1], 2)
      );
      
      let scaledLandmarks;
      if (scaleDistance < 1e-6) {
        scaledLandmarks = translatedLandmarks;
      } else {
        scaledLandmarks = translatedLandmarks.map(coords => {
          if (coords.some(val => val === -1.0)) return coords;
          const distanceFromCenter = Math.sqrt(Math.pow(coords[0], 2) + Math.pow(coords[1], 2) + Math.pow(coords[2], 2));
          const distanceThreshold = 5.0 * scaleDistance;
          if (distanceFromCenter > distanceThreshold) {
            const scaleFactor = distanceThreshold / distanceFromCenter;
            return [
              coords[0] * scaleFactor / scaleDistance,
              coords[1] * scaleFactor / scaleDistance,
              coords[2] * scaleFactor / scaleDistance
            ];
          }
          return [
            coords[0] / scaleDistance,
            coords[1] / scaleDistance,
            coords[2] / scaleDistance
          ];
        });
      }
      normalizedFrames.push(scaledLandmarks);
    } catch (error) {
      console.error('Error normalizing landmarks in applyDistanceNormalization:', error);
      normalizedFrames.push(frameLandmarks);
    }
  }
  return normalizedFrames;
};

export const mapScoreToClassDetails = (score, classLabels = null) => {
  const config = currentModelConfig || getActiveModelConfig();
  const labels = classLabels || config.outputFormat.classLabels || {
    0: 'Not Engaged', 1: 'Barely Engaged', 2: 'Engaged', 3: 'Highly Engaged', 4: 'SNP' // Default fallback
  };
  const details = { index: -1, name: "Prediction Failed", score };
  if (score === null || score === undefined) return details;

  let classIndex = -1;
  // This mapping is specific to an engagement model.
  // For the AffectNet emotion model, this function might not be directly applicable
  // unless the 'score' represents a specific emotion's confidence or similar.
  // The AffectNet model outputs logits for 8 classes.
  if (!(0.0 <= score && score <= 1.0)) details.name = "Invalid Score Range";
  else if (0.0 <= score && score < 0.175) classIndex = 4;  // SNP
  else if (0.175 <= score && score < 0.40) classIndex = 0;  // Not Engaged
  else if (0.40 <= score && score < 0.60) classIndex = 1;  // Barely Engaged
  else if (0.60 <= score && score < 0.825) classIndex = 2;  // Engaged
  else if (0.825 <= score && score <= 1.0) classIndex = 3;  // Highly Engaged
  else details.name = "Score Mapping Error";
  
  if (classIndex !== -1 && labels[classIndex]) {
    details.index = classIndex;
    details.name = labels[classIndex];
  } else if (classIndex !== -1) {
    details.index = classIndex;
    details.name = "Unknown Index";
  }
  return details;
};

export const mapClassificationLogitsToClassDetails = (logits, classLabels = null) => {
  const config = currentModelConfig || getActiveModelConfig();
  const labels = classLabels || config.outputFormat.classLabels || { /* default labels */ };
  const details = { index: -1, name: "Classification Failed", raw_logits: null, probabilities: null };
  if (!logits || !Array.isArray(logits)) return details;

  details.raw_logits = logits;
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map(l => Math.exp(l - maxLogit));
  const sumExp = expLogits.reduce((sum, val) => sum + val, 0);
  const probabilities = expLogits.map(exp => exp / sumExp);
  details.probabilities = probabilities;

  let maxProb = -Infinity;
  let classIndex = -1;
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      classIndex = i;
    }
  }
  
  if (labels[classIndex]) {
    details.index = classIndex;
    details.name = labels[classIndex];
  } else if (classIndex !== -1) {
    details.index = classIndex;
    details.name = "Unknown Index";
  }
  return details;
};

export const predictEngagement = async (landmarksData) => {
  try {
    console.log('predictEngagement: raw landmarksData length:', landmarksData.length, landmarksData);
    if (!onnxSession) {
      console.log("ONNX session not initialized. Attempting to initialize...");
      const initialized = await initializeOnnxModel(); // Uses current active model ID
      if (!initialized) throw new Error('ONNX model initialization failed');
    }
    
    const activeConfig = currentModelConfig || getActiveModelConfig(); // Ensure we have the latest active config
    const { SEQ_LEN, NUM_LANDMARKS, NUM_COORDS } = getModelDimensions(); // Uses currentModelConfig
    
    const preprocessedLandmarks = preprocessLandmarks(landmarksData); // Uses currentModelConfig via getModelDimensions
    console.log('predictEngagement: preprocessedLandmarks first 30 values:', preprocessedLandmarks.slice ? preprocessedLandmarks.slice(0, 30) : preprocessedLandmarks);
    console.log('predictEngagement: input tensor shape:', activeConfig.inputFormat.tensorShape);
    if (!preprocessedLandmarks) throw new Error('Failed to preprocess landmarks');
    
    const inputTensor = new ort.Tensor('float32', preprocessedLandmarks, activeConfig.inputFormat.tensorShape);
    const feeds = { [onnxSession.inputNames[0]]: inputTensor };
    const results = await onnxSession.run(feeds);
    
    // --- Custom logic for 'emotion_transformer_v1' ---
    if (activeConfig.id === "emotion_transformer_v1") {
      // Use named outputs for the transformer model
      const logitsOutputName = activeConfig.outputFormat.outputNames.logits; // 'logits'
      const logits = results[logitsOutputName]?.data;

      if (!logits) {
        console.error(`Output '${logitsOutputName}' not found in model results for ${activeConfig.id}. Available:`, Object.keys(results));
        throw new Error(`Output '${logitsOutputName}' not found`);
      }
      
      const predictionDetailsCls = mapClassificationLogitsToClassDetails(
        Array.from(logits), activeConfig.outputFormat.classLabels
      );
      
      // Embedding is available if needed: results[activeConfig.outputFormat.outputNames[1]]?.data
      
      return {
        // Include emotion field for UI
        emotion: predictionDetailsCls.name,
        score: predictionDetailsCls.probabilities ? Math.max(...predictionDetailsCls.probabilities) : 0,
        name: predictionDetailsCls.name,
        index: predictionDetailsCls.index,
        classification_head_name: predictionDetailsCls.name,
        classification_head_index: predictionDetailsCls.index,
        classification_head_probabilities: predictionDetailsCls.probabilities,
        raw_classification_logits: predictionDetailsCls.raw_logits,
        model_used: activeConfig.name,
        model_id: activeConfig.id
      };
    } 
    // --- Fallback to existing logic for other models ---
    else if (activeConfig.outputFormat.outputType === 'classification' && onnxSession.outputNames.length >= 2) {
      let actualRegressionScores, actualClassificationLogits;
      // This logic is for models outputting regression then classification, or specific named outputs
      if (activeConfig.id === 'v4_v2' && activeConfig.outputFormat.outputNames && activeConfig.outputFormat.outputNames.length >= 2) {
        actualRegressionScores = results[activeConfig.outputFormat.outputNames[0]]?.data;
        actualClassificationLogits = results[activeConfig.outputFormat.outputNames[1]]?.data;
      } else { // Positional for other dual-output models
        actualRegressionScores = results[onnxSession.outputNames[0]]?.data;
        actualClassificationLogits = results[onnxSession.outputNames[1]]?.data;
      }

      if (!actualRegressionScores || !actualClassificationLogits) {
          throw new Error("Expected outputs not found for dual-output classification model.");
      }

      const rawRegressionScore = actualRegressionScores[0];
      const regressionScore = Math.max(0.0, Math.min(1.0, rawRegressionScore));
      const predictionDetailsReg = mapScoreToClassDetails(regressionScore, activeConfig.outputFormat.classLabels);
      const predictionDetailsCls = mapClassificationLogitsToClassDetails(
        Array.from(actualClassificationLogits), activeConfig.outputFormat.classLabels
      );
      
      return {
        score: regressionScore,
        name: predictionDetailsReg.name,
        index: predictionDetailsReg.index,
        classification_head_name: predictionDetailsCls.name,
        classification_head_index: predictionDetailsCls.index,
        classification_head_probabilities: predictionDetailsCls.probabilities,
        raw_regression_score: rawRegressionScore,
        raw_classification_logits: predictionDetailsCls.raw_logits,
        model_used: activeConfig.name,
        model_id: activeConfig.id
      };
    } 
    else { // Single output models (either regression or classification)
      const outputData = results[onnxSession.outputNames[0]]?.data;
      if (!outputData) {
          throw new Error("Primary output not found for single-output model.");
      }

      if (outputData.length === 1) { // Assumed to be a single regression score
        const rawScore = outputData[0];
        const regressionScore = Math.max(0.0, Math.min(1.0, rawScore));
        const predictionDetailsReg = mapScoreToClassDetails(regressionScore, activeConfig.outputFormat.classLabels);
        return {
          score: regressionScore,
          name: predictionDetailsReg.name,
          index: predictionDetailsReg.index,
          raw_regression_score: rawScore,
          model_used: activeConfig.name,
          model_id: activeConfig.id
        };
      } else { // Assumed to be classification logits
        const predictionDetailsCls = mapClassificationLogitsToClassDetails(
          Array.from(outputData), activeConfig.outputFormat.classLabels
        );
        return {
          score: predictionDetailsCls.probabilities ? Math.max(...predictionDetailsCls.probabilities) : 0,
          name: predictionDetailsCls.name,
          index: predictionDetailsCls.index,
          classification_head_name: predictionDetailsCls.name,
          classification_head_index: predictionDetailsCls.index,
          classification_head_probabilities: predictionDetailsCls.probabilities,
          raw_classification_logits: predictionDetailsCls.raw_logits,
          model_used: activeConfig.name,
          model_id: activeConfig.id
        };
      }
    }
  } catch (error) {
    console.error('ONNX prediction error:', error);
    return null;
  }
};

export const getCurrentModelInfo = () => {
  return currentModelConfig || getActiveModelConfig();
};

export const switchModel = async (modelId) => {
  try {
    const newModelConfig = getModelConfigUtil(modelId); // Use renamed import
    if (!newModelConfig) {
      console.error(`Model '${modelId}' not found in config.`);
      return false;
    }
    if (!setActiveModelUtil(modelId)) { // Use renamed import
      console.error(`Failed to set active model to '${modelId}'.`);
      return false;
    }
    onnxSession = null; // Force re-initialization on next predict/load
    currentModelConfig = null; // Clear cached current config
    console.log(`Switched to model: ${newModelConfig.name}. Session will re-initialize on next use.`);
    return true;
  } catch (error) {
    console.error('Failed to switch model:', error);
    return false;
  }
};

export const getAvailableModels = async () => { // Made async to match older signature if needed
  try {
    return getAllModelConfigsUtil(); // Use renamed import
  } catch (error) {
    console.error('Failed to get available models:', error);
    return {};
  }
};

export const isModelLoaded = () => {
  return onnxSession !== null;
};

export const reloadCurrentModel = async () => {
  try {
    const activeConfig = getActiveModelConfig(); // Get current active model ID from config
    onnxSession = null;
    currentModelConfig = null;
    console.log(`Reloading model: ${activeConfig.name}`);
    return await initializeOnnxModel(activeConfig.id); // Initialize with the current active ID
  } catch (error) {
    console.error('Failed to reload model:', error);
    return false;
  }
};
