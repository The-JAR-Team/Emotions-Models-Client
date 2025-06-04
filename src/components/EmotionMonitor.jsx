import React, { useRef, useState, useEffect } from 'react';
import useFaceMesh from '../hooks/useFaceMesh';
import FaceCloseUpStage from '../services/faceCloseUpStage';
import PreprocessDebugView from './PreprocessDebugView';
import { initializeOnnxModel, predictEngagement, getCurrentModelInfo, getAllModels, switchModel } from '../services/emotionOnnxService'; // Added model loader functions
import '../styles/EmotionMonitor.css';

// Constant to enable/disable John Normalization
const ENABLE_JOHN_NORMALIZATION = false;
// Enable separate zoomed-face emotion prediction display
const ENABLE_ZOOM_PREDICTIONS = true;

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
  // State for model loader
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(getCurrentModelInfo()?.id || '');
  // Store raw class probabilities for status display
  const [allProbabilities, setAllProbabilities] = useState([]);
  // Grab current model info (may be null on failure)
  const modelInfo = getCurrentModelInfo();  // List of emotions to ignore when selecting top result
  const [ignoredEmotions, setIgnoredEmotions] = useState([]);
  // State for face close-up image data URL
  const [closeUpDataUrl, setCloseUpDataUrl] = useState(null);
  // State for zoomed face emotion probabilities
  const [zoomProbabilities, setZoomProbabilities] = useState([]);
  // Debug: log when close-up data URL updates
  useEffect(() => {
    console.debug('[Preprocess Debug] closeUpDataUrl updated:', closeUpDataUrl);
  }, [closeUpDataUrl]);
  // State for softmax filter toggle (full frame)
  const [showFilteredProbabilities, setShowFilteredProbabilities] = useState(true);
  // State for zoomed-face probabilities display toggle
  const [showFilteredZoom, setShowFilteredZoom] = useState(true);
  
  // Toggle ignore for a given emotion label
  const handleToggleIgnore = (label) => {
    setIgnoredEmotions(prev =>
      prev.includes(label) ? prev.filter(l => l !== label) : [...prev, label]
    );
  };

  // Custom softmax function for filtered probabilities
  const applySoftmax = (probabilities) => {
    const filtered = probabilities.filter(item => !ignoredEmotions.includes(item.label));
    if (filtered.length === 0) return probabilities;
    
    // Apply softmax to filtered probabilities
    const maxProb = Math.max(...filtered.map(item => item.probability));
    const expValues = filtered.map(item => ({
      ...item,
      probability: Math.exp(item.probability - maxProb)
    }));
    
    const sumExp = expValues.reduce((sum, item) => sum + item.probability, 0);
    const normalized = expValues.map(item => ({
      ...item,
      probability: item.probability / sumExp
    }));
    
    return normalized.sort((a, b) => b.probability - a.probability);
  };

  // Get probabilities to display based on toggle state
  const getDisplayProbabilities = () => {
    if (!allProbabilities.length) return [];
    
    if (showFilteredProbabilities) {
      return applySoftmax(allProbabilities);
    } else {
      return allProbabilities.sort((a, b) => b.probability - a.probability);
    }
  };// Color mapping for emotions
  const getEmotionColor = (emotion) => {
    const emotionColors = {
      'Happiness': '#fbbf24',
      'Sadness': '#3b82f6', 
      'Anger': '#ef4444',
      'Fear': '#8b5cf6',
      'Surprise': '#06b6d4',
      'Disgust': '#84cc16',
      'Contempt': '#6b7280',
      'Neutral': '#64748b'
    };
    return emotionColors[emotion] || '#64748b';
  };
  // Get zoomed-face probabilities to display based on toggle state
  const getDisplayZoom = () => {
    if (!zoomProbabilities.length) return [];
    if (showFilteredZoom) {
      return applySoftmax(zoomProbabilities);
    }
    return [...zoomProbabilities].sort((a, b) => b.probability - a.probability);
  };

  // Get emotion background color for bounding box
  const getEmotionBgColor = (emotion) => {
    const emotionBgColors = {
      'Happiness': 'rgba(251, 191, 36, 0.2)',
      'Sadness': 'rgba(59, 130, 246, 0.2)',
      'Anger': 'rgba(239, 68, 68, 0.2)',
      'Fear': 'rgba(139, 92, 246, 0.2)',
      'Surprise': 'rgba(6, 182, 212, 0.2)',
      'Disgust': 'rgba(132, 204, 22, 0.2)',
      'Contempt': 'rgba(107, 114, 128, 0.2)',
      'Neutral': 'rgba(100, 116, 139, 0.2)'
    };
    return emotionBgColors[emotion] || 'rgba(100, 116, 139, 0.2)';
  };

  // For FPS calculation (optional, but good for debugging)
  const frameCountRef = useRef(0);
  const lastFpsLogTimeRef = useRef(Date.now());
  // Throttle ONNX inference to once every 1.5 seconds
  const lastInferenceTimeRef = useRef(0);
  // Throttle zoomed-face predictions separately
  const lastZoomTimeRef = useRef(0);
  const ZOOM_INFERENCE_INTERVAL_MS = 1500;

  // Initialize ONNX model
  useEffect(() => {
    const initModel = async () => {
      try {
        setOnnxStatus('Starting ONNX model initialization...');
        const initialized = await initializeOnnxModel();
        setOnnxModelReady(initialized);
        setSelectedModel(getCurrentModelInfo()?.id || '');
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
    // Load available models
    setAvailableModels(getAllModels() || []);
  }, []);
  
  // Prepare FaceCloseUpStage
  const stageRef = useRef(new FaceCloseUpStage(256, 0.2, '/models', true));
  useEffect(() => {
    // Load any required models (no-op for FaceMesh-based)
    stageRef.current.loadModels().catch(err => console.error('FaceCloseUpStage load error:', err));
  }, []);

  // Handle model selection
  const handleModelChange = async (e) => {
    const modelId = e.target.value;
    setSelectedModel(modelId);
    setOnnxStatus(`Loading model ${modelId}...`);
    const ok = await switchModel(modelId);
    setOnnxModelReady(ok);
    setOnnxStatus(ok ? `Model loaded: ${getCurrentModelInfo().name}` : `Failed to load model: ${modelId}`);
  };

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
      // Clear zoom and debug views
      setZoomProbabilities([]);
      setCloseUpDataUrl(null);
      // Clear canvas
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
    // Debug: print raw landmarks from FaceMesh
    console.log('[Landmarks Raw]', landmarks);

    if (isActive && landmarks && onnxModelReady) {
      const now = Date.now();
      // Only run inference every 1.5 seconds (1500 ms)
      if (now - lastInferenceTimeRef.current >= 1000) {
        lastInferenceTimeRef.current = now;
        console.log(`[${new Date().toISOString()}] Running inference...`);
        // Initialize and debug: print landmarks used for full-frame prediction
        let landmarksForPrediction = landmarks;
        console.log('[Landmarks For Prediction]', landmarksForPrediction);
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
          // Log full-frame raw landmarks
          console.log('--- FULL FRAME INPUT ---');
          console.log('Raw landmarks:', landmarksForPrediction);
          const prediction = await predictEngagement(landmarksForPrediction, widthForPrediction, heightForPrediction, { context: 'FULL' });
          console.log(`[${new Date().toISOString()}] FULL FRAME PREDICTION:`, prediction);
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
    // After drawing, extract cropped face
    // Compute tight face box and apply relative padding
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    landmarks.forEach(landmark => {
      const x = landmark.x * canvas.width;
      const y = landmark.y * canvas.height;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    });
    // Relative padding: use same factor as FaceCloseUpStage
    const boxW = maxX - minX;
    const boxH = maxY - minY;
    const padW = boxW * stageRef.current.paddingFactor;
    const padH = boxH * stageRef.current.paddingFactor;
    minX = Math.max(0, minX - padW);
    minY = Math.max(0, minY - padH);
    maxX = Math.min(canvas.width, maxX + padW);
    maxY = Math.min(canvas.height, maxY + padH);
    
    // Use emotion-specific colors
    const emotionColor = getEmotionColor(detectedEmotion);
    const emotionBgColor = getEmotionBgColor(detectedEmotion);
    
    ctx.strokeStyle = emotionColor;
    ctx.lineWidth = 6;
    ctx.shadowColor = emotionColor;
    ctx.shadowBlur = 8;
    ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
    
    // Reset shadow for fill operations
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;    if (detectedEmotion) {
      const boxWidth = maxX - minX;
      const boxHeight = maxY - minY;
      
      // Calculate dynamic font size based on box size
      const baseFontSize = Math.max(18, Math.min(32, Math.min(boxWidth / 8, boxHeight / 12)));
      const labelHeight = baseFontSize + 30;
      
      // Draw background for emotion label with gradient
      const gradient = ctx.createLinearGradient(minX, minY - labelHeight, minX, minY);
      gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
      ctx.fillStyle = gradient;
      ctx.fillRect(minX, minY - labelHeight, boxWidth, labelHeight);
      
      // Add subtle border to label background
      ctx.strokeStyle = emotionColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(minX, minY - labelHeight, boxWidth, labelHeight);
      
      // Draw emotion text with enhanced styling
      ctx.font = `bold ${baseFontSize}px 'Inter', 'Segoe UI', Arial, sans-serif`;
      ctx.fillStyle = emotionColor;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      
      // Add text glow effect
      ctx.shadowColor = emotionColor;
      ctx.shadowBlur = 8;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
      
      const emotionText = `${detectedEmotion}`;
      const scoreText = `${emotionScore !== null ? (emotionScore * 100).toFixed(1) + '%' : 'N/A'}`;
      
      // Draw emotion name with glow
      ctx.fillText(emotionText, minX + (boxWidth / 2), minY - labelHeight + baseFontSize/2 + 8);
      
      // Draw score with smaller font and white color
      ctx.font = `600 ${Math.max(14, baseFontSize * 0.75)}px 'Inter', 'Segoe UI', Arial, sans-serif`;
      ctx.fillStyle = '#FFFFFF';
      ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
      ctx.shadowBlur = 4;
      ctx.fillText(scoreText, minX + (boxWidth / 2), minY - labelHeight + baseFontSize + 12);
      
      // Reset shadow
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
    }
    ctx.restore();
    // Manual crop & zoom using detected box
    const boxWidth = maxX - minX;
    const boxHeight = maxY - minY;
    if (boxWidth > 0 && boxHeight > 0) {
      // Use FaceCloseUpStage for consistent padding, crop and resize
      const faceBox = { x: minX, y: minY, width: boxWidth, height: boxHeight };
      try {
        const dataUrl = await stageRef.current.processCanvas(canvas, faceBox);
        console.debug('[Preprocess Debug] cropped faceUrl via stage:', dataUrl ? 'OK' : 'null');
        setCloseUpDataUrl(dataUrl);
      } catch (err) {
        console.error('FaceCloseUpStage crop error:', err);
      }
      // Prepare zoomed-face prediction with throttle
      const nowZoom = Date.now();
      console.log(`[ZoomPrediction] check: enabled=${ENABLE_ZOOM_PREDICTIONS}, ready=${onnxModelReady}, delta=${nowZoom - lastZoomTimeRef.current}ms`);
      if (ENABLE_ZOOM_PREDICTIONS && onnxModelReady && nowZoom - lastZoomTimeRef.current >= ZOOM_INFERENCE_INTERVAL_MS) {
        lastZoomTimeRef.current = nowZoom;
        // normalize landmarks relative to box, rescale z to full-frame pixel space
        const zoomLandmarks = landmarks.map(lm => ({
          x: (lm.x * canvas.width - minX) / boxWidth,
          y: (lm.y * canvas.height - minY) / boxHeight,
          z: lm.z * (canvas.width / boxWidth)
        }));
        console.log('[ZoomPrediction] landmarks:', zoomLandmarks);
        try {
          // Run zoomed-face prediction with same preprocessing as full-frame
          const zoomPred = await predictEngagement(zoomLandmarks, boxWidth, boxHeight);
          console.log('[ZoomPrediction] raw result:', zoomPred);
          const labels = getCurrentModelInfo().outputFormat.classLabels;
          const zp = (zoomPred?.classification_head_probabilities || []).map((p, i) => ({ label: labels[i], probability: p }));
          console.log('[ZoomPrediction] probabilities:', zp);
          setZoomProbabilities(zp);
        } catch (e) {
          console.error('[ZoomPrediction] error:', e);
        }
      }
    }
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
          <div className="model-loader">
            <label htmlFor="model-select">Model: </label>
            <select
              id="model-select"
              value={selectedModel}
              onChange={handleModelChange}
              disabled={!onnxModelReady || !modelInfo}
            >
              {availableModels.map(m => (
                <option key={m.id} value={m.id}>{m.name}</option>
              ))}
            </select>
            {!modelInfo && <span className="error-message">No model loaded</span>}
          </div>
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
          </div>        </div>
      </div>

      {/* Main content layout - side by side */}
      <div className="main-content">
        {/* Video and preprocess debug stack */}
        <div className="video-area">
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
          {/* Preprocessed face debug view below webcam */}
          <PreprocessDebugView dataUrl={closeUpDataUrl} />
        </div>
        {/* Probabilities sidebar */}
        <div className="probabilities-sidebar">          {/* Enhanced probabilities display */}
          {allProbabilities.length > 0 && (
            <div className="probabilities-section">
              <div className="probabilities-header">
                <div className="probabilities-title">🎭 Emotion Probabilities (Full Frame)</div>
                <button
                  className={`softmax-toggle-btn ${showFilteredProbabilities ? 'filtered' : 'raw'}`}
                  onClick={() => setShowFilteredProbabilities(!showFilteredProbabilities)}
                  title={showFilteredProbabilities ? 'Filtered softmax' : 'Raw probabilities'}
                >
                  {showFilteredProbabilities ? '🧮 Filtered' : '📊 Raw'}
                </button>
              </div>
              <div className="probabilities-list">
                {getDisplayProbabilities().map(({ label, probability }) => (
                  <div key={label} className="probability-item" data-emotion={label}>
                    <span className="probability-label">{label}</span>
                    <span className="probability-value">{(probability * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {ENABLE_ZOOM_PREDICTIONS && zoomProbabilities.length > 0 && (
            <div className="probabilities-section zoomed">
              <div className="probabilities-header">
                <div className="probabilities-title">🎯 Zoomed Face Probabilities</div>
                <button
                  className={`softmax-toggle-btn ${showFilteredZoom ? 'filtered' : 'raw'}`}
                  onClick={() => setShowFilteredZoom(!showFilteredZoom)}
                  title={showFilteredZoom ? 'Filtered softmax (zoom)' : 'Raw probabilities (zoom)'}
                >
                  {showFilteredZoom ? '🧮 Filtered' : '📊 Raw'}
                </button>
              </div>
              <div className="probabilities-list">
                {getDisplayZoom().map(({ label, probability }) => (
                  <div key={label} className="probability-item" data-emotion={label}>
                    <span className="probability-label">{label}</span>
                    <span className="probability-value">{(probability * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Styled emotion filter buttons */}
          {allProbabilities.length > 0 && modelInfo && (
            <div className="emotion-filters">
              <div className="filters-title">🎛️ Emotion Filters</div>
              <div className="emotion-toggle-grid">
                {Object.keys(modelInfo.outputFormat.classLabels).map(key => {
                  const label = modelInfo.outputFormat.classLabels[key];
                  const isIgnored = ignoredEmotions.includes(label);
                  return (
                    <div
                      key={label}
                      className={`emotion-toggle-btn ${isIgnored ? 'disabled' : 'enabled'}`}
                      data-emotion={label}
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
      </div>

      {/* FaceMesh hook */}
      {useFaceMesh(isActive, videoRef, handleResults, handleFaceMeshStatus)}
    </div>
  );
};

export default EmotionMonitor;
