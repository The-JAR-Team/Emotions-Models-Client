import React, { useRef, useState, useEffect } from 'react';
import useFaceMesh from '../hooks/useFaceMesh';
import { initializeOnnxModel, predictEngagement, getCurrentModelInfo } from '../services/emotionOnnxService'; // Changed predictEmotion to predictEngagement
import '../styles/EmotionMonitor.css';

// Constant to enable/disable John Normalization
const ENABLE_JOHN_NORMALIZATION = false;

const EmotionMonitor = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [faceMeshStatus, setFaceMeshStatus] = useState('Initializing...');
  const [isActive, setIsActive] = useState(true);
  const [detectedEmotion, setDetectedEmotion] = useState(null);
  const [emotionScore, setEmotionScore] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [onnxModelReady, setOnnxModelReady] = useState(false);
  const [onnxStatus, setOnnxStatus] = useState('Loading ONNX model...');
  // State for John Normalization
  const [johnNormalizationEnabled, setJohnNormalizationEnabled] = useState(ENABLE_JOHN_NORMALIZATION);
  // Store raw class probabilities for status display
  const [allProbabilities, setAllProbabilities] = useState([]);
  // List of emotions to ignore when selecting top result
  const [ignoredEmotions, setIgnoredEmotions] = useState([]);
  // Toggle ignore for a given emotion label
  const handleToggleIgnore = (label) => {
    setIgnoredEmotions(prev =>
      prev.includes(label) ? prev.filter(l => l !== label) : [...prev, label]
    );
  };
  // State for emotions to ignore

  // For FPS calculation (optional, but good for debugging)
  const frameCountRef = useRef(0);
  const lastFpsLogTimeRef = useRef(Date.now());
  // Throttle ONNX inference to once every 1.5 seconds
  const lastInferenceTimeRef = useRef(0);

  // Initialize ONNX model
  useEffect(() => {
    const initModel = async () => {
      try {
        setOnnxStatus('Starting ONNX model initialization...');
        const initialized = await initializeOnnxModel();
        setOnnxModelReady(initialized);
        if (!initialized) {
          setOnnxStatus('Failed to initialize ONNX model');
          setErrorMessage("Failed to initialize ONNX model");
        } else {
          setOnnxStatus('ONNX model initialized successfully');
          console.log("ONNX model initialized successfully");
        }
      } catch (error) {
        setOnnxStatus(`Error initializing ONNX model: ${error.message}`);
        console.error("Error initializing ONNX model:", error);
        setErrorMessage("Error initializing ONNX model: " + (error.message || "Unknown error"));
      }
    };
    initModel();
  }, []);

  // Handle FaceMesh results
  const handleResults = async (results) => {
    // FPS Calculation
    frameCountRef.current++;
    const now = Date.now();
    if (now - lastFpsLogTimeRef.current >= 5000) { // Log every 5 seconds
      const elapsedSeconds = (now - lastFpsLogTimeRef.current) / 1000;
      const fps = frameCountRef.current / elapsedSeconds;
      console.log(`🎥 Camera FPS: ${fps.toFixed(2)}`);
      frameCountRef.current = 0;
      lastFpsLogTimeRef.current = now;
    }

    if (isLoading) setIsLoading(false);
    if (errorMessage) setErrorMessage(null);

    if (!canvasRef.current || !videoRef.current || !results || !results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      // Clear previous emotion if no face is detected
      setDetectedEmotion(null);
      setEmotionScore(null);
      // Clear canvas if no face
      if (canvasRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const videoWidth = videoRef.current.videoWidth;
    const videoHeight = videoRef.current.videoHeight;

    canvas.width = videoWidth;
    canvas.height = videoHeight;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    try {
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    } catch (e) {
      console.error("Error drawing video to canvas:", e);
      ctx.restore();
      return;
    }

    const landmarks = results.multiFaceLandmarks[0]; // Assuming one face

    if (isActive && landmarks && onnxModelReady) {
      const now = Date.now();
      // Only run inference every 1.5 seconds (1500 ms)
      if (now - lastInferenceTimeRef.current >= 1500) {
        lastInferenceTimeRef.current = now;
        console.log(`[${new Date().toISOString()}] Running inference...`);

        let landmarksForPrediction = landmarks;
        let widthForPrediction = videoWidth;
        let heightForPrediction = videoHeight;

        if (johnNormalizationEnabled) {
          // Calculate tight bounding box of the face in pixel coordinates
          let faceBoxMinX = Infinity, faceBoxMinY = Infinity;
          let faceBoxMaxX = -Infinity, faceBoxMaxY = -Infinity;
          landmarks.forEach(lm => {
            const px = lm.x * videoWidth;
            const py = lm.y * videoHeight;
            faceBoxMinX = Math.min(faceBoxMinX, px);
            faceBoxMinY = Math.min(faceBoxMinY, py);
            faceBoxMaxX = Math.max(faceBoxMaxX, px);
            faceBoxMaxY = Math.max(faceBoxMaxY, py);
          });

          const roiX = faceBoxMinX;
          const roiY = faceBoxMinY;
          const roiWidth = faceBoxMaxX - faceBoxMinX;
          const roiHeight = faceBoxMaxY - faceBoxMinY;

          // Check for valid ROI dimensions
          if (roiWidth > 0 && roiHeight > 0) {
            landmarksForPrediction = landmarks.map(lm => ({
              x: (lm.x * videoWidth - roiX) / roiWidth,
              y: (lm.y * videoHeight - roiY) / roiHeight,
              z: lm.z // Pass z through
            }));
            widthForPrediction = roiWidth;
            heightForPrediction = roiHeight;
            console.log(`[${new Date().toISOString()}] John Normalization applied. ROI: x:${roiX.toFixed(0)}, y:${roiY.toFixed(0)}, w:${roiWidth.toFixed(0)}, h:${roiHeight.toFixed(0)}`);
          } else {
            console.log(`[${new Date().toISOString()}] John Normalization skipped: Invalid ROI dimensions (w:${roiWidth.toFixed(0)}, h:${roiHeight.toFixed(0)})`);
            // Fallback to original landmarks and dimensions (already set by default)
          }
        }

        try {
          const prediction = await predictEngagement(landmarksForPrediction, widthForPrediction, heightForPrediction);
          console.log(`[${new Date().toISOString()}] Prediction result:`, prediction);
        if (prediction) {
          setDetectedEmotion(prediction.emotion);
          setEmotionScore(prediction.score);
          // Update all class probabilities
          const modelInfo = getCurrentModelInfo();
          const labels = modelInfo.outputFormat.classLabels;
          const probs = prediction.classification_head_probabilities || [];
          // Map labels and probabilities
          const mapped = probs.map((p, idx) => ({ label: labels[idx], probability: p }));
          setAllProbabilities(mapped);
          // Filter out ignored emotions, then re-normalize
          const remaining = mapped.filter(item => !ignoredEmotions.includes(item.label));
          const total = remaining.reduce((sum, item) => sum + item.probability, 0) || 1;
          const normalized = remaining.map(item => ({ label: item.label, probability: item.probability / total }));
          // Choose top normalized
          const best = normalized.reduce((maxItem, item) => item.probability > maxItem.probability ? item : maxItem, { label: '', probability: 0 });
          setDetectedEmotion(best.label);
          setEmotionScore(best.probability);
          } else {
            setDetectedEmotion('Error');
            setEmotionScore(null);
          }
        } catch (error) {
          console.error('Error predicting emotion:', error);
          setErrorMessage('Error predicting emotion: ' + error.message);
          setDetectedEmotion('Error');
          setEmotionScore(null);
        }
      }
    }

    // Draw face bounding box (optional, but good for visualization)
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    landmarks.forEach(landmark => {
      const x = landmark.x * canvas.width;
      const y = landmark.y * canvas.height;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    });

    const padding = 20;
    minX = Math.max(0, minX - padding);
    minY = Math.max(0, minY - padding);
    maxX = Math.min(canvas.width, maxX + padding);
    maxY = Math.min(canvas.height, maxY + padding);    ctx.strokeStyle = '#FFCC00'; // Default yellow
    if (detectedEmotion) {
      if (['Happy', 'Surprise'].includes(detectedEmotion)) ctx.strokeStyle = '#00FF00'; // Green
      else if (['Sad', 'Fear', 'Disgust', 'Anger', 'Contempt'].includes(detectedEmotion)) ctx.strokeStyle = '#FF0000'; // Red
    }
    ctx.lineWidth = 4;
    ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);

    if (detectedEmotion) {
      const boxWidth = maxX - minX;
      const boxHeight = maxY - minY;
      
      // Calculate dynamic font size based on box size
      const baseFontSize = Math.max(16, Math.min(28, Math.min(boxWidth / 8, boxHeight / 12)));
      const labelHeight = baseFontSize + 20;
      
      // Draw background for emotion label
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(minX, minY - labelHeight, boxWidth, labelHeight);
      
      // Draw emotion text with enhanced styling
      ctx.font = `bold ${baseFontSize}px 'Segoe UI', Arial, sans-serif`;
      ctx.fillStyle = ctx.strokeStyle; // Use the same color as the box for text
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      
      // Add text shadow effect
      ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
      ctx.shadowBlur = 3;
      ctx.shadowOffsetX = 1;
      ctx.shadowOffsetY = 1;
      
      const emotionText = `${detectedEmotion}`;
      const scoreText = `${emotionScore !== null ? (emotionScore * 100).toFixed(0) + '%' : 'N/A'}`;
      
      // Draw emotion name
      ctx.fillText(emotionText, minX + (boxWidth / 2), minY - labelHeight + baseFontSize/2 + 5);
      
      // Draw score with smaller font
      ctx.font = `600 ${Math.max(12, baseFontSize * 0.7)}px 'Segoe UI', Arial, sans-serif`;
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(scoreText, minX + (boxWidth / 2), minY - labelHeight + baseFontSize + 8);
      
      // Reset shadow
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
    }
    ctx.restore();
  };

  const handleFaceMeshStatus = (status) => {
    setFaceMeshStatus(status);
    if (status.toLowerCase().includes('error') || status.toLowerCase().includes('failed') || status.toLowerCase().includes('denied')) {
      setErrorMessage(status);
      setIsLoading(false);
    }
    if (status === 'FaceMesh Ready') {
      setErrorMessage(null);
      // setIsLoading(false); // isLoading is handled by first result now
    }
  };

  const toggleActive = () => setIsActive(!isActive);

  const handleRetry = async () => {
    setErrorMessage(null);
    setIsLoading(true);
    setFaceMeshStatus('Initializing...');
    setDetectedEmotion(null);
    setEmotionScore(null);

    if (!onnxModelReady) {
      try {
        setOnnxStatus('Re-initializing ONNX model...');
        const initialized = await initializeOnnxModel();
        setOnnxModelReady(initialized);
        if (!initialized) setErrorMessage("Failed to re-initialize ONNX model");
        else setOnnxStatus('ONNX model initialized successfully');
      } catch (error) {
        console.error("Error re-initializing ONNX model:", error);
        setErrorMessage("Error re-initializing ONNX model: " + error.message);
      }
    }
    // Force a browser repaint/reflow to help with webcam issues
    if (videoRef.current) {
      videoRef.current.style.display = 'none';
      setTimeout(() => {
        if (videoRef.current) videoRef.current.style.display = 'block';
      }, 50);
    }
  };
  return (
    <div className="emotion-monitor">
      <div className="status-bar">
        <div className="status-header">
          <div className="status-text">
            <span>FaceMesh: {faceMeshStatus}</span>
            <span>ONNX: {onnxStatus}</span>
            <span className="top-emotion">Top Emotion: {detectedEmotion || 'Detecting...'}</span>
            {errorMessage && <span className="error-message">Error: {errorMessage}</span>}
          </div>
          <div className="button-group">
            <button onClick={toggleActive} className={`toggle-button ${isActive ? 'active' : 'inactive'}`}>
              {isActive ? 'Pause' : 'Resume'}
            </button>
            {errorMessage && (
              <button onClick={handleRetry} className="retry-button">
                Retry
              </button>
            )}
          </div>
        </div>

        {/* Enhanced probabilities display */}
        {allProbabilities.length > 0 && (
          <div className="probabilities-section">
            <div className="probabilities-title">Emotion Probabilities</div>
            <div className="probabilities-list">
              {allProbabilities.map(({ label, probability }) => (
                <div key={label} className="probability-item">
                  <span className="probability-label">{label}</span>
                  <span className="probability-value">{(probability * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Styled emotion filter buttons */}
        {allProbabilities.length > 0 && (
          <div className="emotion-filters">
            <div className="filters-title">Emotion Filters</div>
            <div className="emotion-toggle-grid">
              {Object.keys(getCurrentModelInfo().outputFormat.classLabels).map(key => {
                const label = getCurrentModelInfo().outputFormat.classLabels[key];
                const isIgnored = ignoredEmotions.includes(label);
                return (
                  <div
                    key={label}
                    className={`emotion-toggle-btn ${isIgnored ? 'disabled' : 'enabled'}`}
                    onClick={() => handleToggleIgnore(label)}
                  >
                    {label}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      <div className="video-container">
        <video ref={videoRef} className="webcam" muted playsInline autoPlay style={{ objectFit: 'contain' }} />
        <canvas ref={canvasRef} className="overlay" style={{ objectFit: 'contain' }} />
        {isLoading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <div className="loading-text">{faceMeshStatus}</div>
          </div>
        )}
      </div>

      {/* FaceMesh hook */}
      {useFaceMesh(isActive, videoRef, handleResults, handleFaceMeshStatus)}
    </div>
  );
};

export default EmotionMonitor;
