// renderer.js
// This script is executed in the Electron renderer process (the web page).
// It handles user interactions, media capture (screen, audio), and communication with the main process via IPC.

const { ipcRenderer } = require('electron'); // Module to send messages to the main process

// --- Global variables for media capture and processing ---
let mediaStream = null; // Holds the MediaStream object for screen and/or audio capture
let screenshotInterval = null; // Timer for periodically capturing screenshots
let audioContext = null; // Web Audio API AudioContext for processing audio
let audioProcessor = null; // ScriptProcessorNode for handling audio data (primarily for Windows/Linux loopback/mic)
let micAudioProcessor = null; // Separate audio processor if microphone input is handled distinctly (e.g. Linux)
// let audioBuffer = []; // This global audioBuffer seems unused, local buffers are used in functions.

const SAMPLE_RATE = 24000; // Desired sample rate for audio processing
const AUDIO_CHUNK_DURATION = 0.1; // Duration of audio chunks to process (in seconds)
const BUFFER_SIZE = 4096; // Buffer size for ScriptProcessorNode, affects latency and processing frequency

// Variables for offscreen screenshot rendering
let hiddenVideo = null; // Hidden <video> element to play the screen capture stream
let offscreenCanvas = null; // Offscreen <canvas> for drawing video frames
let offscreenContext = null; // 2D rendering context for the offscreen canvas

// Platform detection flags
const isLinux = process.platform === 'linux';
const isMacOS = process.platform === 'darwin';

/**
 * Gets a reference to the main custom element <jefe-app>.
 * @returns {HTMLElement | null} The jefe-app element or null if not found.
 */
function jefeElement() {
    return document.getElementById('jefe');
}

/**
 * Converts a Float32Array (typically from Web Audio API) to an Int16Array (PCM data).
 * This is often needed for audio formats expected by speech-to-text services.
 * @param {Float32Array} float32Array - The input array with samples ranging from -1.0 to 1.0.
 * @returns {Int16Array} The output array with PCM samples.
 */
function convertFloat32ToInt16(float32Array) {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
        // Improved scaling to prevent clipping
        const s = Math.max(-1, Math.min(1, float32Array[i]));
        int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return int16Array;
}

/**
 * Converts an ArrayBuffer containing binary data to a Base64 encoded string.
 * @param {ArrayBuffer} buffer - The ArrayBuffer to convert.
 * @returns {string} The Base64 encoded string.
 */
function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

/**
 * Initializes the Gemini AI session by sending API key and configuration to the main process.
 * Updates the UI element's status based on success or failure.
 * @param {string} [profile='interview'] - The selected user profile for AI interaction.
 * @param {string} [language='en-US'] - The language for speech recognition.
 */
async function initializeGemini(profile = 'interview', language = 'en-US') {
    const apiKey = localStorage.getItem('apiKey')?.trim(); // Get API key from local storage
    if (apiKey) {
        // Invoke 'initialize-gemini' IPC handler in the main process
        const success = await ipcRenderer.invoke('initialize-gemini', apiKey, localStorage.getItem('customPrompt') || '', profile, language);
        const appElement = jefeElement();
        if (appElement) {
            if (success) {
                appElement.setStatus('Live');
            } else {
                appElement.setStatus('Error: Gemini Init Failed');
            }
        }
    } else {
        const appElement = jefeElement();
        if (appElement) {
            appElement.setStatus('Error: API Key Missing');
        }
        console.error("API Key is missing from local storage.");
    }
}

// --- IPC Event Listeners from Main Process ---

// Listen for 'update-status' messages from the main process to update the UI.
ipcRenderer.on('update-status', (event, status) => {
    console.log('Status update from main:', status);
    const appElement = jefeElement();
    if (appElement) {
        appElement.setStatus(status);
    }
});

// Listen for 'update-response' messages from the main process (AI responses) to update the UI.
ipcRenderer.on('update-response', (event, response) => {
    console.log('Gemini response from main:', response);
    const appElement = jefeElement();
    if (appElement) {
        appElement.setResponse(response);
    }
});

// Listener for mouse event status changes from main process (e.g., click-through mode toggled)
ipcRenderer.on('mouse-events-status', (event, ignored) => {
    console.log('Mouse events ignored status from main:', ignored);
    // The UI might want to reflect this, e.g., by changing an icon or style.
    // For now, just logging. The jefe-element itself doesn't show this status.
});

// Listener for a shortcut-triggered window close request from the main process.
ipcRenderer.on('shortcut-close-window', () => {
    const appElement = jefeElement();
    if (appElement && typeof appElement.handleClose === 'function') {
        appElement.handleClose(); // Call the web component's close handler
    }
});


/**
 * Starts the screen and audio capture process based on the operating system.
 * - macOS: Uses SystemAudioDump (via IPC to main) for system audio and getDisplayMedia for screen.
 * - Linux: Uses getDisplayMedia for screen and getUserMedia for microphone.
 * - Windows: Uses getDisplayMedia with loopback audio for screen and system audio.
 * It also starts an interval for capturing screenshots.
 */
async function startCapture() {
    try {
        if (isMacOS) {
            // On macOS, use SystemAudioDump for audio and getDisplayMedia for screen
            console.log('Starting macOS capture with SystemAudioDump...');

            // Start macOS audio capture
            const audioResult = await ipcRenderer.invoke('start-macos-audio');
            if (!audioResult.success) {
                throw new Error('Failed to start macOS audio capture: ' + audioResult.error);
            }

            // Get screen capture for screenshots
            mediaStream = await navigator.mediaDevices.getDisplayMedia({
                video: {
                    frameRate: 1,
                    width: { ideal: 1920 },
                    height: { ideal: 1080 },
                },
                audio: false, // Don't use browser audio on macOS
            });

            console.log('macOS screen capture started - audio handled by SystemAudioDump');
        } else if (isLinux) {
            // Linux - use display media for screen capture and getUserMedia for microphone
            mediaStream = await navigator.mediaDevices.getDisplayMedia({
                video: {
                    frameRate: 1,
                    width: { ideal: 1920 },
                    height: { ideal: 1080 },
                },
                audio: false, // Don't use system audio loopback on Linux
            });

            // Get microphone input for Linux
            let micStream = null;
            try {
                micStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: SAMPLE_RATE,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                    },
                    video: false,
                });

                console.log('Linux microphone capture started');

                // Setup audio processing for microphone on Linux
                setupLinuxMicProcessing(micStream);
            } catch (micError) {
                console.warn('Failed to get microphone access on Linux:', micError);
                // Continue without microphone if permission denied
            }

            console.log('Linux screen capture started');
        } else {
            // Windows - use display media with loopback for system audio
            mediaStream = await navigator.mediaDevices.getDisplayMedia({
                video: {
                    frameRate: 1,
                    width: { ideal: 1920 },
                    height: { ideal: 1080 },
                },
                audio: {
                    sampleRate: SAMPLE_RATE,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });

            console.log('Windows capture started with loopback audio');

            // Setup audio processing for Windows loopback audio only
            setupWindowsLoopbackProcessing();
        }

        console.log('MediaStream obtained:', {
            hasVideo: mediaStream.getVideoTracks().length > 0,
            hasAudio: mediaStream.getAudioTracks().length > 0,
            videoTrack: mediaStream.getVideoTracks()[0]?.getSettings(),
        });

        // Start capturing screenshots every second
        screenshotInterval = setInterval(captureScreenshot, 1000);

        // Capture first screenshot immediately
        setTimeout(captureScreenshot, 100);
    } catch (err) {
        console.error('Error starting capture:', err);
        jefe.e().setStatus('error'); // Renamed
    }
}

function setupLinuxMicProcessing(micStream) {
    // Setup microphone audio processing for Linux
    const micAudioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    const micSource = micAudioContext.createMediaStreamSource(micStream);
    const micProcessor = micAudioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);

    let audioBuffer = [];
    const samplesPerChunk = SAMPLE_RATE * AUDIO_CHUNK_DURATION; // Number of samples in one chunk

    micProcessor.onaudioprocess = async e => {
        const inputData = e.inputBuffer.getChannelData(0); // Get mono audio data
        // It seems a global audioBuffer was intended here, but it's shadowed.
        // Using a local one for this function scope or ensure the global one is used if intended.
        let localAudioBuffer = []; // Assuming local buffer for this processing logic.
        localAudioBuffer.push(...inputData);

        // Process audio in defined chunks
        while (localAudioBuffer.length >= samplesPerChunk) {
            const chunk = localAudioBuffer.splice(0, samplesPerChunk);
            const pcmData16 = convertFloat32ToInt16(chunk); // Convert to 16-bit PCM
            const base64Data = arrayBufferToBase64(pcmData16.buffer); // Encode as Base64

            // Send audio data to the main process
            await ipcRenderer.invoke('send-audio-content', {
                data: base64Data,
                mimeType: `audio/pcm;rate=${SAMPLE_RATE}`,
            });
        }
    };

    micSource.connect(micProcessor); // Connect microphone source to processor
    micProcessor.connect(micAudioContext.destination); // Connect processor to destination (e.g., speakers, though often not needed for capture)

    // Store processor reference for cleanup, potentially on a global or class variable if needed elsewhere
    // For now, this assignment to a global `audioProcessor` might conflict if Windows also uses it.
    // Consider renaming or scoping this if both Linux mic and Windows loopback are simultaneously possible (they aren't in current logic).
    micAudioProcessor = micProcessor; // Use a distinct variable for Linux mic processor
}

/**
 * Sets up audio processing for Windows loopback audio obtained via getDisplayMedia.
 * It creates an AudioContext, connects the mediaStream to a ScriptProcessorNode,
 * and processes audio data in chunks, sending it to the main process.
 */
function setupWindowsLoopbackProcessing() {
    if (!mediaStream || mediaStream.getAudioTracks().length === 0) {
        console.warn("Windows loopback processing called without an audio track in mediaStream.");
        return;
    }
    audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    const source = audioContext.createMediaStreamSource(mediaStream);
    audioProcessor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1); // 1 input channel, 1 output channel

    let localAudioBuffer = []; // Local buffer for accumulating audio samples
    const samplesPerChunk = SAMPLE_RATE * AUDIO_CHUNK_DURATION; // Number of samples in one chunk

    audioProcessor.onaudioprocess = async e => {
        const inputData = e.inputBuffer.getChannelData(0); // Get mono audio data
        localAudioBuffer.push(...inputData);

        // Process audio in defined chunks
        while (localAudioBuffer.length >= samplesPerChunk) {
            const chunk = localAudioBuffer.splice(0, samplesPerChunk);
            const pcmData16 = convertFloat32ToInt16(chunk); // Convert to 16-bit PCM
            const base64Data = arrayBufferToBase64(pcmData16.buffer); // Encode as Base64

            // Send audio data to the main process
            await ipcRenderer.invoke('send-audio-content', {
                data: base64Data,
                mimeType: `audio/pcm;rate=${SAMPLE_RATE}`,
            });
        }
    };

    source.connect(audioProcessor); // Connect media stream source to processor
    audioProcessor.connect(audioContext.destination); // Connect processor to context destination (speakers, etc.)
                                                    // This is often done even if output is not desired, to keep the graph active.
}


/**
 * Captures a screenshot from the current mediaStream (screen capture).
 * It uses a hidden video element and an offscreen canvas to draw and encode the image.
 * The captured image (JPEG, base64 encoded) is sent to the main process.
 */
async function captureScreenshot() {
    // console.log('Capturing screenshot...'); // Can be too verbose for interval logging
    if (!mediaStream || mediaStream.getVideoTracks().length === 0) {
        // console.warn("Screenshot capture skipped: No active video mediaStream.");
        return;
    }

    // Lazy initialization of the hidden video element used to play the screen capture stream
    if (!hiddenVideo) {
        hiddenVideo = document.createElement('video');
        hiddenVideo.srcObject = mediaStream;
        hiddenVideo.muted = true;
        hiddenVideo.playsInline = true;
        await hiddenVideo.play();

        await new Promise(resolve => {
            if (hiddenVideo.readyState >= 2) return resolve();
            hiddenVideo.onloadedmetadata = () => resolve();
        });

        // Lazy init of canvas based on video dimensions
        offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = hiddenVideo.videoWidth;
        offscreenCanvas.height = hiddenVideo.videoHeight;
        offscreenContext = offscreenCanvas.getContext('2d');
    }

    // Check if video is ready
    if (hiddenVideo.readyState < 2) {
        console.warn('Video not ready yet, skipping screenshot');
        return;
    }

    offscreenContext.drawImage(hiddenVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

    // Check if image was drawn properly by sampling a pixel
    const imageData = offscreenContext.getImageData(0, 0, 1, 1);
    const isBlank = imageData.data.every((value, index) => {
        // Check if all pixels are black (0,0,0) or transparent
        return index === 3 ? true : value === 0;
    });

    if (isBlank) {
        console.warn('Screenshot appears to be blank/black');
    }

    offscreenCanvas.toBlob(
        async blob => {
            if (!blob) {
                console.error('Failed to create blob from canvas');
                return;
            }

            const reader = new FileReader();
            reader.onloadend = async () => {
                const base64data = reader.result.split(',')[1];

                // Validate base64 data
                if (!base64data || base64data.length < 100) {
                    console.error('Invalid base64 data generated');
                    return;
                }

                const result = await ipcRenderer.invoke('send-image-content', {
                    data: base64data,
                });

                if (!result.success) {
                    console.error('Failed to send image:', result.error);
                }
            };
            reader.readAsDataURL(blob);
        },
        'image/jpeg',
        0.8
    );
}

function stopCapture() {
    if (screenshotInterval) {
        clearInterval(screenshotInterval);
        screenshotInterval = null;
    }

    if (audioProcessor) {
        audioProcessor.disconnect();
        audioProcessor = null;
    }

    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    // Stop macOS audio capture if running
    if (isMacOS) {
        ipcRenderer.invoke('stop-macos-audio').catch(err => {
            console.error('Error stopping macOS audio:', err);
        });
    }

    // Clean up hidden elements
    if (hiddenVideo) {
        hiddenVideo.pause();
        hiddenVideo.srcObject = null;
        hiddenVideo = null;
    }
    offscreenCanvas = null;
    offscreenContext = null; // Clear canvas context reference
}

/**
 * Sends a text message from the UI to the main process to be relayed to the Gemini AI.
 * @param {string} text - The text message to send.
 * @returns {Promise<object>} A promise that resolves with the result of the IPC call.
 */
async function sendTextMessage(text) {
    if (!text || text.trim().length === 0) {
        console.warn('Cannot send empty text message.');
        return { success: false, error: 'Empty message content.' };
    }

    try {
        // Invoke 'send-text-message' IPC handler in the main process
        const result = await ipcRenderer.invoke('send-text-message', text);
        if (result.success) {
            console.log('Text message sent successfully via IPC.');
        } else {
            console.error('Failed to send text message via IPC:', result.error);
        }
        return result;
    } catch (error) {
        console.error('IPC error sending text message:', error);
        return { success: false, error: error.message };
    }
}

// --- Expose functions to the main world (accessible via window.jefe in jefe-element.js) ---
// This object provides an API for the web component (jefe-element.js) to interact with
// this renderer process script, which in turn communicates with the main process.
window.jefe = {
    initializeGemini,    // Function to initialize the AI session
    startCapture,        // Function to start screen and audio capture
    stopCapture,         // Function to stop all captures
    sendTextMessage,     // Function to send a text message to the AI
    isLinux: isLinux,    // Boolean flag indicating if the platform is Linux
    isMacOS: isMacOS,    // Boolean flag indicating if the platform is macOS
    e: jefeElement,      // Function to get the <jefe-app> DOM element
};
