import { useEffect, useRef } from 'react';
import { FaceMesh } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';

// Create a singleton FaceMesh instance that can be reused
let globalFaceMeshInstance = null;
let globalFaceMeshInitialized = false;
let globalInitializationInProgress = false;
let globalCameraInstance = null;

// Ensure proper Module setup before any FaceMesh initialization
if (typeof window !== 'undefined') {
  window.Module = window.Module || {};
  window.Module.arguments_ = window.Module.arguments_ || [];
  // Prevent setting Module.arguments directly
  Object.defineProperty(window.Module, 'arguments', {
    get: function() { return this.arguments_; },
    set: function(v) { this.arguments_ = v; }
  });
}

export default function useFaceMesh(enabled, videoRef, onResults, onStatusChange, onErrorState) {
  const errorCount = useRef(0);
  const streamRef = useRef(null);
  const initializedRef = useRef(false);
  const resultCallbackRef = useRef(onResults);
  const statusCallbackRef = useRef(onStatusChange);
  const errorStateCallbackRef = useRef(onErrorState);
  const MAX_ERRORS = 10;

  // Update refs when callbacks change
  useEffect(() => {
    resultCallbackRef.current = onResults;
    statusCallbackRef.current = onStatusChange;
    errorStateCallbackRef.current = onErrorState;
  }, [onResults, onStatusChange, onErrorState]);

  useEffect(() => {
    if (!enabled) return;

    let isComponentMounted = true;
    
    const MAX_INIT_RETRIES = 3;
    const INIT_RETRY_DELAY_MS = 1000;
    
    // Update status if component is still mounted
    const updateStatus = (status) => {
      if (isComponentMounted && statusCallbackRef.current) {
        statusCallbackRef.current(status);
      }
    };
    
    // Function to retry initialization with delay
    const retryInitialization = async (fn) => {
      let lastError;
      for (let i = 1; i <= MAX_INIT_RETRIES; i++) {
        try {
          return await fn();
        } catch (err) {
          lastError = err;
          updateStatus(`Initializing FaceMesh… retry ${i}/${MAX_INIT_RETRIES}`);
          console.warn(`FaceMesh init error (attempt ${i}):`, err);
          await new Promise(r => setTimeout(r, INIT_RETRY_DELAY_MS));
        }
      }
      throw lastError;
    };

    // Initialize webcam stream
    const initializeWebcam = async () => {
      try {
        // Clean up existing stream
        if (streamRef.current) {
          try { 
            streamRef.current.getTracks().forEach(track => track.stop()); 
          } catch (e) {
            console.error("Error stopping existing stream:", e);
          }
          streamRef.current = null;
        }

        // Request camera with explicit dimensions
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          }
        });
        
        // Set stream to video element and play
        if (videoRef.current && isComponentMounted) {
          videoRef.current.srcObject = stream;
          streamRef.current = stream;
          
          // Wait for video to be ready
          await new Promise((resolve) => {
            videoRef.current.onloadedmetadata = () => {
              if (videoRef.current) {
                videoRef.current.play().then(resolve).catch(e => {
                  console.error("Error playing video:", e);
                  resolve(); // Continue anyway
                });
              } else {
                resolve();
              }
            };
          });
          
          return true;
        }
        return false;
      } catch (err) {
        console.error("Webcam initialization error:", err);
        updateStatus('Camera access denied');
        return false;
      }
    };

    // Main FaceMesh initialization function
    const initializeFaceMesh = async () => {
      errorCount.current = 0;
      
      // Step 1: Initialize webcam first
      const webcamReady = await initializeWebcam();
      if (!webcamReady || !isComponentMounted) return false;

      // Step 2: Set up or reuse FaceMesh
      try {
        // If another initialization is already in progress, wait for it
        if (globalInitializationInProgress) {
          updateStatus('Waiting for existing initialization...');
          let waitCount = 0;
          while (globalInitializationInProgress && waitCount < 50) { // max 5 seconds wait
            await new Promise(r => setTimeout(r, 100));
            waitCount++;
          }
          
          // If initialization completed successfully, reuse it
          if (globalFaceMeshInitialized && globalFaceMeshInstance) {
            // Just set a new results callback
            globalFaceMeshInstance.onResults(results => {
              if (!isComponentMounted) return;
              
              if (!initializedRef.current) {
                initializedRef.current = true;
                updateStatus('FaceMesh Ready');
              }
              
              if (resultCallbackRef.current) {
                resultCallbackRef.current(results);
              }
              errorCount.current = 0;
              
              // Notify about error state being cleared
              if (errorStateCallbackRef.current) {
                errorStateCallbackRef.current(false);
              }
            });
            
            // Start camera with existing FaceMesh
            if (!globalCameraInstance && videoRef.current) {
              const camera = new Camera(videoRef.current, {
                onFrame: async () => {
                  if (!videoRef.current?.srcObject || videoRef.current.paused || !globalFaceMeshInstance || !isComponentMounted) return;
                  
                  try {
                    await globalFaceMeshInstance.send({ image: videoRef.current });
                  } catch (error) {
                    errorCount.current++;
                    console.error('FaceMesh send error:', error);
                    
                    // Notify about error state
                    if (errorStateCallbackRef.current) {
                      errorStateCallbackRef.current(errorCount.current > MAX_ERRORS);
                    }
                    
                    if (errorCount.current > MAX_ERRORS) {
                      updateStatus('Too many errors - Click Retry');
                    }
                  }
                },
                width: 640,
                height: 480
              });
              
              globalCameraInstance = camera;
              await camera.start();
            } else if (globalCameraInstance && !globalCameraInstance._running) {
              // If camera exists but not running, restart it
              await globalCameraInstance.start();
            }
            
            return true;
          }
        }
        
        // Mark initialization as in progress
        globalInitializationInProgress = true;
        
        try {
          // Initialize new FaceMesh only if we don't have a working one
          if (!globalFaceMeshInstance || !globalFaceMeshInitialized) {
            updateStatus('Creating new FaceMesh instance...');
            
            // Create new FaceMesh instance
            const fm = new FaceMesh({
              locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${f}`
            });
            
            // Configure FaceMesh
            fm.setOptions({
              maxNumFaces: 1,
              refineLandmarks: true,
              minDetectionConfidence: 0.5,
              minTrackingConfidence: 0.5
            });
            
            // Set results callback
            fm.onResults(results => {
              if (!isComponentMounted) return;
              
              if (!initializedRef.current) {
                initializedRef.current = true;
                updateStatus('FaceMesh Ready');
              }
              
              if (resultCallbackRef.current) {
                resultCallbackRef.current(results);
              }
              errorCount.current = 0;
              
              // Notify about error state being cleared
              if (errorStateCallbackRef.current) {
                errorStateCallbackRef.current(false);
              }
            });
            
            // Initialize FaceMesh (this loads the model)
            updateStatus('Loading FaceMesh model...');
            await fm.initialize();
            
            // Store in global reference
            globalFaceMeshInstance = fm;
            globalFaceMeshInitialized = true;
          } else {
            // Just update the callback for the existing instance
            globalFaceMeshInstance.onResults(results => {
              if (!isComponentMounted) return;
              
              if (!initializedRef.current) {
                initializedRef.current = true;
                updateStatus('FaceMesh Ready');
              }
              
              if (resultCallbackRef.current) {
                resultCallbackRef.current(results);
              }
              errorCount.current = 0;
              
              // Notify about error state being cleared
              if (errorStateCallbackRef.current) {
                errorStateCallbackRef.current(false);
              }
            });
          }
          
          // Set up or reuse the camera
          if (!globalCameraInstance && videoRef.current) {
            const camera = new Camera(videoRef.current, {
              onFrame: async () => {
                if (!videoRef.current?.srcObject || videoRef.current.paused || !globalFaceMeshInstance || !isComponentMounted) return;
                
                try {
                  await globalFaceMeshInstance.send({ image: videoRef.current });
                } catch (error) {
                  errorCount.current++;
                  console.error('FaceMesh send error:', error);
                  
                  // Notify about error state
                  if (errorStateCallbackRef.current) {
                    errorStateCallbackRef.current(errorCount.current > MAX_ERRORS);
                  }
                  
                  if (errorCount.current > MAX_ERRORS) {
                    updateStatus('Too many errors - Click Retry');
                  }
                }
              },
              width: 640,
              height: 480
            });
            
            globalCameraInstance = camera;
            await camera.start();
          } else if (globalCameraInstance && !globalCameraInstance._running) {
            // If camera exists but not running, restart it
            await globalCameraInstance.start();
          }
          
          return true;
        } finally {
          // Mark initialization as complete
          globalInitializationInProgress = false;
        }
      } catch (err) {
        console.error('FaceMesh initialization failed:', err);
        updateStatus('Initialization Error - Click Retry');
        return false;
      }
    };

    // Start initialization process
    retryInitialization(initializeFaceMesh)
      .catch(err => {
        console.error("FaceMesh init failed after retries:", err);
        initializedRef.current = false;
        updateStatus('Initialization Failed - Click Retry');
      });

    // Cleanup function
    return () => {
      isComponentMounted = false;
      initializedRef.current = false;
      
      // Stop stream - but DON'T stop the camera or close FaceMesh
      if (streamRef.current) {
        try { 
          // Just clear our local references, don't stop the actual tracks
          streamRef.current = null;
        } catch (e) {}
      }
      
      // We intentionally DON'T clean up these resources on unmount:
      // - Don't close the global faceMesh instance
      // - Don't stop the global camera instance
      // - Don't stop any media tracks
      
      // Just clear video source reference from our component
      if (videoRef.current) {
        try { 
          // Don't actually clear the video source, just detach our reference
          // videoRef.current.srcObject = null; 
        } catch (e) {}
      }
    };
  }, [enabled, videoRef]);
}
