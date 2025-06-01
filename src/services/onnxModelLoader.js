// src/services/onnxModelLoader.js

export const loadOnnxModelFromUrl = async (url) => {
  try {
    console.log(`Attempting to fetch ONNX model from: ${url}`);
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/octet-stream',
        'Content-Type': 'application/octet-stream'
      },
      cache: 'no-cache', // Consider 'default' or 'force-cache' for production if appropriate
      mode: 'cors', // Ensure server allows CORS if fetching from different origin
      credentials: 'same-origin' // Or 'include' if needed
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status} ${response.statusText} from ${url}`);
    }
    
    console.log(`Successfully fetched model from: ${url}`);
    return await response.arrayBuffer();
  } catch (error) {
    console.error(`Error fetching model from ${url}:`, error);
    return null;
  }
};

export const getOnnxModelUri = async (modelFilename) => {
  // Try loading from URL first with various paths that might work
  const baseUrl = window.location.origin;
  // Ensure basePath correctly reflects the deployment subdirectory if any.
  // For a standard Vite app, pathname might be '/' or '/subpath/'.
  let basePath = window.location.pathname;
  if (!basePath.endsWith('/')) {
    // If path is like '/subpath/index.html', get '/subpath/'
    basePath = basePath.substring(0, basePath.lastIndexOf('/') + 1);
  }

  // Construct URLs relative to the current deployment.
  // Vite serves 'public' directory contents from the root.
  const urlsToTry = [
    new URL(`models/${modelFilename}`, baseUrl).href, // Absolute path from domain root + /models/
    // The following might be redundant if baseUrl already includes the subpath, or useful if not.
    // new URL(basePath + `models/${modelFilename}`, baseUrl).href, 
  ];

  // Add relative paths that might work depending on how the app is served
  // These are relative to the current page's URL path.
  if (typeof window !== 'undefined') { // Ensure this runs in browser context
    urlsToTry.push(new URL(`models/${modelFilename}`, window.location.href).href);
    urlsToTry.push(new URL(`../models/${modelFilename}`, window.location.href).href); // If service is one level down
  }


  // Add direct paths often used in Vite for public assets
  urlsToTry.push(`/models/${modelFilename}`);      // Path from root, common for Vite
  urlsToTry.push(`./models/${modelFilename}`);     // Relative path

  // Deduplicate URLs
  const uniqueUrls = [...new Set(urlsToTry)];
  
  console.log(`Attempting to load ONNX model '${modelFilename}' from the following URLs:`, uniqueUrls);
  
  for (const url of uniqueUrls) {
    try {
      const modelBuffer = await loadOnnxModelFromUrl(url);
      if (modelBuffer) {
        const blob = new Blob([modelBuffer], { type: 'application/octet-stream' });
        const modelBlobUrl = URL.createObjectURL(blob);
        console.log(`Created ONNX model blob URL from successfully fetched: ${url}`);
        return modelBlobUrl;
      }
    } catch (error) {
      // Error is logged in loadOnnxModelFromUrl, continue to next URL
      // console.warn(`Skipping URL ${url} due to error: ${error.message}`);
    }
  }
  
  console.error(`Failed to load ONNX model '${modelFilename}' from any of the attempted URLs.`);
  return null;
};
