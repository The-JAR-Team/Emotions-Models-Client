export const ONNX_MODEL_PATH = '/models/emotion_transformer_v1.onnx';

export const EMOTION_CLASSES = {
  0: 'Neutral',
  1: 'Happy',
  2: 'Sad',
  3: 'Surprise',
  4: 'Fear',
  5: 'Disgust',
  6: 'Anger',
  7: 'Contempt',
};

// Landmark indices for normalization (from the Python script)
export const LANDMARK_SETTINGS = {
  NUM_LANDMARKS: 478,
  LANDMARK_DIM: 3,
  NOSE_TIP_INDEX: 1,
  LEFT_EYE_INNER_CORNER_INDEX: 133, // MediaPipe landmark index
  RIGHT_EYE_INNER_CORNER_INDEX: 362, // MediaPipe landmark index
};
