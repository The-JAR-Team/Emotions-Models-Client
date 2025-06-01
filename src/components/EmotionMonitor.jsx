import React, { useRef, useState, useEffect } from 'react';
import useFaceMesh from '../hooks/useFaceMesh';
import { initializeOnnxModel, predictEmotion, getCurrentModelInfo } from '../services/emotionOnnxService';
import '../styles/EmotionMonitor.css';

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

  // For FPS calculation (optional, but good for debugging)
  const frameCountRef = useRef(0);
  const lastFpsLogTimeRef = useRef(Date.now());

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
      try {
        // The emotion ONNX model processes single frames, not sequences like the engagement one.
        const prediction = await predictEmotion(landmarks, videoWidth, videoHeight);
        if (prediction) {
          setDetectedEmotion(prediction.emotion);
          setEmotionScore(prediction.score); // This is a logit, not probability
          // console.log('Emotion Prediction:', prediction);
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
    maxY = Math.min(canvas.height, maxY + padding);

    ctx.strokeStyle = '#FFCC00'; // Default yellow
    if (detectedEmotion) {
      if (['Happy', 'Surprise'].includes(detectedEmotion)) ctx.strokeStyle = '#00FF00'; // Green
      else if (['Sad', 'Fear', 'Disgust', 'Anger', 'Contempt'].includes(detectedEmotion)) ctx.strokeStyle = '#FF0000'; // Red
    }
    ctx.lineWidth = 3;
    ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);

    if (detectedEmotion) {
      const boxWidth = maxX - minX;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(minX, minY - 30, boxWidth, 30);
      const labelFontSize = Math.max(14, Math.min(18, boxWidth / 10));
      ctx.font = `bold ${labelFontSize}px Arial`;
      ctx.fillStyle = ctx.strokeStyle; // Use the same color as the box for text
      ctx.textAlign = "center";
      ctx.fillText(`${detectedEmotion} (${emotionScore !== null ? emotionScore.toFixed(2) : 'N/A'})`, minX + (boxWidth / 2), minY - 10);
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
        <div className="status-text">
          <span>FaceMesh: {faceMeshStatus}</span>
          <span>ONNX: {onnxStatus}</span>
          <span>Emotion: {detectedEmotion ? `${detectedEmotion} (${emotionScore !== null ? emotionScore.toFixed(2) : '...'})` : 'Detecting...'}</span>
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
