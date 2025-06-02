import * as ort from 'onnxruntime-web';
import { getOnnxModelUri } from './onnxModelLoader';
import { fetchModelFromHooks } from './onnxHooksFallback';
import { getActiveModelConfig, getModelPaths, getModelConfig as getModelConfigUtil, setActiveModel as setActiveModelUtil, getAllModelConfigs as getAllModelConfigsUtil } from '../config/modelConfig'; // Renamed imports to avoid conflict

// Constant to enable/disable FERPlus specific normalization
const ENABLE_FERPLUS_NORMALIZATION = true;

// Landmark indices based on the Python training script for FERPlus normalization
// These are indices into the 478 landmarks array
const FERPLUS_NOSE_TIP_IDX = 1;         // Nose tip
const FERPLUS_LEFT_EYE_INNER_IDX = 133; // Left eye inner corner
const FERPLUS_RIGHT_EYE_INNER_IDX = 362;// Right eye inner corner

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
    let targetModelId = modelId;
    // If no modelId is provided, try to default to a FERPlus model if available
    if (!targetModelId) {
        const allConfigs = getAllModelConfigsUtil();
        const ferplusModel = allConfigs.find(m => m.filename === 'emotion_transformer_small.onnx');
        if (ferplusModel) {
            targetModelId = ferplusModel.id;
            console.log(`Defaulting to FERPlus model: ${targetModelId}`);
        }
    }

    if (targetModelId) {
      if (!setActiveModelUtil(targetModelId)) {
        console.error(`Failed to set active model to ${targetModelId}. It might not exist.`);
        currentModelConfig = getActiveModelConfig(); // Fallback
      } else {
        currentModelConfig = getModelConfigUtil(targetModelId);
      }
    } else {
      currentModelConfig = getActiveModelConfig(); // Fallback to current active or default
    }

    if (!currentModelConfig) {
        throw new Error("No model configuration found or could be set.");
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

// FERPlus specific normalization (ported from the Python script)
const applyFerPlusNormalization = (landmarksArray, imageWidth, imageHeight) => {
  if (!landmarksArray || landmarksArray.length === 0) return landmarksArray;

  // landmarksArray is expected to be [NUM_LANDMARKS][NUM_COORDS] for a single frame
  // The Python script processes one image (frame) at a time.
  // landmarks input to this function should be the 3D world coordinates from MediaPipe,
  // scaled by image width/height as in the python script's extract_landmarks_from_image
  // before normalization.

  const landmarksAbs3d = landmarksArray.map(lm => ({
    x: lm.x * imageWidth,
    y: lm.y * imageHeight,
    z: lm.z * imageWidth // Python script uses image_width for Z scaling
  }));

  if (landmarksAbs3d.length <= Math.max(FERPLUS_NOSE_TIP_IDX, FERPLUS_LEFT_EYE_INNER_IDX, FERPLUS_RIGHT_EYE_INNER_IDX)) {
    console.warn("Not enough landmarks for FERPlus normalization.");
    return landmarksArray; // Return original if not enough landmarks
  }

  const noseTip3d = { ...landmarksAbs3d[FERPLUS_NOSE_TIP_IDX] };

  const landmarksCentered3d = landmarksAbs3d.map(lm => ({
    x: lm.x - noseTip3d.x,
    y: lm.y - noseTip3d.y,
    z: lm.z - noseTip3d.z
  }));

  const pLeftEyeInnerXY = { x: landmarksCentered3d[FERPLUS_LEFT_EYE_INNER_IDX].x, y: landmarksCentered3d[FERPLUS_LEFT_EYE_INNER_IDX].y };
  const pRightEyeInnerXY = { x: landmarksCentered3d[FERPLUS_RIGHT_EYE_INNER_IDX].x, y: landmarksCentered3d[FERPLUS_RIGHT_EYE_INNER_IDX].y };

  const dx = pLeftEyeInnerXY.x - pRightEyeInnerXY.x;
  const dy = pLeftEyeInnerXY.y - pRightEyeInnerXY.y;
  let interOcularDistance = Math.sqrt(dx * dx + dy * dy);

  if (interOcularDistance < 1e-6) {
    console.warn("Inter-ocular distance is too small, using fallback.");
    interOcularDistance = imageWidth / 4.0; // Fallback from Python script
    if (interOcularDistance < 1e-6) interOcularDistance = 1.0; // Further fallback
  }

  const landmarksNormalized3d = landmarksCentered3d.map(lm => ({
    x: lm.x / interOcularDistance,
    y: lm.y / interOcularDistance,
    z: lm.z / interOcularDistance
  }));

  return landmarksNormalized3d; // This is an array of {x,y,z} objects
};

export const preprocessLandmarks = (landmarks, videoWidth, videoHeight) => {
  // Convert single-frame landmarks (array of objects) into an array of frames
  const frames = Array.isArray(landmarks) && landmarks.length > 0 && typeof landmarks[0].x === 'number'
    ? [landmarks] // Input is a single frame of landmarks
    : Array.isArray(landmarks)
      ? landmarks // Input is already an array of frames (though typically we process one by one)
      : [];

  const { SEQ_LEN, NUM_LANDMARKS, NUM_COORDS } = getModelDimensions();
  const processedFramesData = []; // This will hold the flat Float32Array data

  for (let i = 0; i < SEQ_LEN; i++) {
    let frameLandmarks = frames[i] || []; // Get current frame's landmarks or empty if no frame

    // Apply FERPlus normalization if enabled and model requires it
    if (ENABLE_FERPLUS_NORMALIZATION && currentModelConfig && currentModelConfig.normalizationType === 'ferplus') {
        console.log("Applying FERPlus Normalization in preprocessLandmarks");
        // Ensure frameLandmarks are in the {x,y,z} format expected by applyFerPlusNormalization
        // The landmarks from MediaPipe are already in this format.
        // videoWidth and videoHeight are passed for scaling inside applyFerPlusNormalization
        frameLandmarks = applyFerPlusNormalization(frameLandmarks, videoWidth, videoHeight);
    } else if (currentModelConfig && currentModelConfig.inputFormat.requiresNormalization) {
        // Apply old normalization if configured and FERPlus is not active for this model
        // This part might need review if applyDistanceNormalization is different
        console.log("Applying legacy distance normalization");
        // frameLandmarks = applyDistanceNormalization(frameLandmarks); // Assuming applyDistanceNormalization takes similar input
    }


    // Initialize flat array for the current frame's landmark data
    const frameArray = new Float32Array(NUM_LANDMARKS * NUM_COORDS).fill(-1.0);

    for (let j = 0; j < Math.min(frameLandmarks.length, NUM_LANDMARKS); j++) {
      const lm = frameLandmarks[j];
      if (lm && typeof lm.x === 'number' && typeof lm.y === 'number' && typeof lm.z === 'number') {
        frameArray[j * NUM_COORDS + 0] = lm.x;
        frameArray[j * NUM_COORDS + 1] = lm.y;
        frameArray[j * NUM_COORDS + 2] = lm.z;
      }
    }
    processedFramesData.push(...frameArray); // Add current frame's flat data
  }
  // Create a single Float32Array for all frames
  return new Float32Array(processedFramesData);
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

export const predictEngagement = async (landmarks, videoWidth, videoHeight) => { // Added videoWidth, videoHeight
  if (!onnxSession || !currentModelConfig) {
    console.error('ONNX session or model config not initialized.');
    return null;
  }

  // Preprocess landmarks: this now receives videoWidth and videoHeight for FER+ norm
  const processedInput = preprocessLandmarks(landmarks, videoWidth, videoHeight);

  const { SEQ_LEN, NUM_LANDMARKS, NUM_COORDS } = getModelDimensions();
  const tensorShape = [1, SEQ_LEN, NUM_LANDMARKS, NUM_COORDS];
  
  // If the model expects [batch_size, num_landmarks, num_coords] (e.g. [1, 478, 3])
  // and SEQ_LEN is 1, adjust shape. The FERPlus model from script expects [1, 478, 3]
  let finalShape = tensorShape;
  if (currentModelConfig.id === 'ferplus_transformer_small_v1' || currentModelConfig.filename === 'emotion_transformer_small.onnx') {
      if (SEQ_LEN === 1) { // Common case for real-time single frame prediction
          finalShape = [1, NUM_LANDMARKS, NUM_COORDS];
      } else {
          // If SEQ_LEN > 1, the shape [1, SEQ_LEN, NUM_LANDMARKS, NUM_COORDS] might be for models expecting sequences.
          // The provided Python model seems to take (1, NUM_LANDMARKS, LANDMARK_DIM)
          // For now, assuming SEQ_LEN=1 for this specific model.
          // If your model truly expects a sequence like [1, N, 478, 3], this needs adjustment.
          console.warn(`Model ${currentModelConfig.id} is assumed to take [1, NUM_LANDMARKS, NUM_COORDS], but SEQ_LEN is ${SEQ_LEN}. Adjusting shape assumption.`);
          finalShape = [1, NUM_LANDMARKS, NUM_COORDS]; // Overriding for FERPlus model
      }
  }


  const tensor = new ort.Tensor('float32', processedInput, finalShape);
  const feeds = { [onnxSession.inputNames[0]]: tensor };

  try {
    const results = await onnxSession.run(feeds);
    const outputTensor = results[onnxSession.outputNames[0]]; // Assuming first output is logits
    const probabilities = outputTensor.data; // This should be the raw logits/probabilities

    // The FERPlus model from script has two outputs: 'logits' and 'embedding'
    // We are interested in 'logits' for classification
    // outputNames: ["logits", "embedding"]

    let classificationProbabilities = probabilities; // Default to the first output's data

    if (onnxSession.outputNames.includes('logits')) {
        classificationProbabilities = results['logits'].data;
    } else {
        console.warn("Output 'logits' not found, using the first output tensor by default.");
    }


    // Softmax application if model output is raw logits
    let finalProbabilities = Array.from(classificationProbabilities);
    if (currentModelConfig.outputFormat.applySoftmax) {
      finalProbabilities = softmax(Array.from(classificationProbabilities));
    }
    
    const classLabels = currentModelConfig.outputFormat.classLabels;
    if (!classLabels || Object.keys(classLabels).length !== finalProbabilities.length) {
        console.error("Class labels mismatch or not defined for the current model.");
        // Return raw probabilities if labels are problematic
        return {
            emotion: "Error: Label mismatch",
            score: 0,
            classification_head_probabilities: finalProbabilities 
        };
    }

    let maxScore = -Infinity;
    let detectedEmotion = 'N/A';
    finalProbabilities.forEach((score, index) => {
      if (score > maxScore) {
        maxScore = score;
        detectedEmotion = classLabels[index] || `Class ${index}`;
      }
    });

    return {
      emotion: detectedEmotion,
      score: maxScore,
      classification_head_probabilities: finalProbabilities // Return all probabilities
    };
  } catch (error) {
    console.error('Error during ONNX inference:', error);
    return null;
  }
};

// Helper function for softmax (if needed, some models output logits)
const softmax = (arr) => {
  const maxLogit = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b);
  return exps.map(x => x / sumExps);
};

export const getCurrentModelInfo = () => {
  return currentModelConfig ? 
    { 
      id: currentModelConfig.id,
      name: currentModelConfig.name,
      filename: currentModelConfig.filename,
      inputFormat: currentModelConfig.inputFormat,
      outputFormat: currentModelConfig.outputFormat,
      normalizationType: currentModelConfig.normalizationType 
    } : 
    null;
};

export const getAllModels = () => {
    return getAllModelConfigsUtil();
};

export const switchModel = async (modelId) => {
    console.log(`Attempting to switch model to: ${modelId}`);
    const newModelConfig = getModelConfigUtil(modelId);
    if (!newModelConfig) {
        console.error(`Cannot switch: Model with ID '${modelId}' not found in configuration.`);
        return false;
    }
    if (currentModelConfig && currentModelConfig.id === modelId && onnxSession) {
        console.log(`Model ${modelId} is already active.`);
        return true; // Already active
    }
    return initializeOnnxModel(modelId); // Re-initialize with the new model ID
};
