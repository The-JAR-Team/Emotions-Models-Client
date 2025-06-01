import * as ort from 'onnxruntime-web';
import { ONNX_MODEL_PATH, EMOTION_CLASSES, LANDMARK_SETTINGS } from '../config/config';

let session;

export const initializeOnnxModel = async () => {
  try {
    ort.env.wasm.wasmPaths = {
      'ort-wasm.wasm': '/onnxruntime-web/ort-wasm.wasm',
      'ort-wasm-simd.wasm': '/onnxruntime-web/ort-wasm-simd.wasm',
      'ort-wasm-threaded.wasm': '/onnxruntime-web/ort-wasm-threaded.wasm',
    };

    session = await ort.InferenceSession.create(ONNX_MODEL_PATH, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    console.log('ONNX session initialized successfully');
    return true;
  } catch (error) {
    console.error('Error initializing ONNX session:', error);
    return false;
  }
};

const normalizeLandmarks = (landmarks, imageWidth, imageHeight) => {
  if (!landmarks || landmarks.length !== LANDMARK_SETTINGS.NUM_LANDMARKS) {
    console.warn('Invalid landmarks for normalization');
    return null;
  }

  const landmarksAbs3d = landmarks.map(lm => [
    lm.x * imageWidth,
    lm.y * imageHeight,
    lm.z * imageWidth, // As per Python script
  ]);

  const noseTip = landmarksAbs3d[LANDMARK_SETTINGS.NOSE_TIP_INDEX];
  const landmarksCentered3d = landmarksAbs3d.map(lm => [
    lm[0] - noseTip[0],
    lm[1] - noseTip[1],
    lm[2] - noseTip[2],
  ]);

  const pLeftEyeInnerXy = landmarksCentered3d[LANDMARK_SETTINGS.LEFT_EYE_INNER_CORNER_INDEX].slice(0, 2);
  const pRightEyeInnerXy = landmarksCentered3d[LANDMARK_SETTINGS.RIGHT_EYE_INNER_CORNER_INDEX].slice(0, 2);

  const dx = pLeftEyeInnerXy[0] - pRightEyeInnerXy[0];
  const dy = pLeftEyeInnerXy[1] - pRightEyeInnerXy[1];
  let interOcularDistance = Math.sqrt(dx * dx + dy * dy);

  if (interOcularDistance < 1e-6) {
    interOcularDistance = imageWidth / 4.0; // Fallback
    if (interOcularDistance < 1e-6) interOcularDistance = 1.0;
  }

  const normalizedLandmarks = landmarksCentered3d.map(lm => [
    lm[0] / interOcularDistance,
    lm[1] / interOcularDistance,
    lm[2] / interOcularDistance,
  ]);

  return normalizedLandmarks;
};

export const predictEmotion = async (faceLandmarks, imageWidth, imageHeight) => {
  if (!session) {
    console.error('ONNX session not initialized');
    return null;
  }

  const normalizedLandmarks = normalizeLandmarks(faceLandmarks, imageWidth, imageHeight);
  if (!normalizedLandmarks) {
    console.error('Landmark normalization failed');
    return null;
  }

  const tensorInput = new ort.Tensor('float32', Float32Array.from(normalizedLandmarks.flat()), [1, LANDMARK_SETTINGS.NUM_LANDMARKS, LANDMARK_SETTINGS.LANDMARK_DIM]);

  try {
    const feeds = { [session.inputNames[0]]: tensorInput };
    const results = await session.run(feeds);
    const logits = results[session.outputNames[0]].data;
    
    // Get the index of the max logit
    let maxLogit = -Infinity;
    let predictedIndex = -1;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) {
        maxLogit = logits[i];
        predictedIndex = i;
      }
    }

    return {
      emotion: EMOTION_CLASSES[predictedIndex] || 'Unknown',
      score: maxLogit, // This is the logit value, not a probability. Softmax would be needed for probability.
      index: predictedIndex,
      allLogits: Array.from(logits) // For debugging or further analysis
    };
  } catch (error) {
    console.error('Error during ONNX inference:', error);
    return null;
  }
};

export const getCurrentModelInfo = () => {
  // This is a placeholder. In a real scenario, this might come from the model file or a config.
  // For the emotion_transformer_v1.onnx, it expects a single frame of landmarks.
  return {
    name: 'emotion_transformer_v1',
    inputFormat: {
      sequenceLength: 1, // This model processes one frame at a time
      numLandmarks: LANDMARK_SETTINGS.NUM_LANDMARKS,
      landmarkDim: LANDMARK_SETTINGS.LANDMARK_DIM
    },
    outputClasses: EMOTION_CLASSES
  };
};
