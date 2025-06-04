import React from 'react';
import '../styles/PreprocessDebugView.css';

/**
 * Displays the preprocessed face image for debugging
 */
const PreprocessDebugView = ({ dataUrl }) => {
  // Debug: log dataUrl updates
  React.useEffect(() => {
    console.debug('[PreprocessDebugView] dataUrl:', dataUrl);
  }, [dataUrl]);
  // Show placeholder while waiting for data
  if (!dataUrl) {
    return (
      <div className="preprocess-debug-view placeholder">
        <div className="preprocess-title">🛠️ Preprocessed Face (waiting...)</div>
      </div>
    );
  }
  return (
    <div className="preprocess-debug-view">
      <div className="preprocess-title">🛠️ Preprocessed Face</div>
      <img src={dataUrl} alt="Preprocessed face" className="preprocess-image" />
    </div>
  );
};

export default PreprocessDebugView;
