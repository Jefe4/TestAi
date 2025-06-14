// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
    process.exit(0);
}

// Electron modules
const { app, BrowserWindow, desktopCapturer, globalShortcut, session, ipcMain, shell, screen } = require('electron');
// Node.js modules
const path = require('node:path');
const fs = require('node:fs');
const os = require('os');
const { spawn } = require('child_process');
// External dependencies
const { GoogleGenAI } = require('@google/genai');
// Local utilities
const { pcmToWav, analyzeAudioBuffer, saveDebugAudio } = require('./audioUtils');
const { getSystemPrompt } = require('./prompts');

// --- Global state variables ---
let geminiSession = null; // Holds the active Google GenAI session
let loopbackProc = null; // Process for Windows loopback audio capture (not currently used in favor of getDisplayMedia)
let systemAudioProc = null; // Process for macOS SystemAudioDump utility for system audio capture
let audioIntervalTimer = null; // Timer for audio processing intervals (not currently used, but could be for other audio methods)
let mouseEventsIgnored = false; // Flag to track if mouse events for the main window are currently ignored (click-through mode)
let messageBuffer = ''; // Buffer for accumulating text parts from Gemini streaming responses

/**
 * Ensures that necessary data directories for the application exist.
 * Creates them if they don't. Specifically:
 * - ~/.jefe/data/image (for screenshots or image data)
 * - ~/.jefe/data/audio (for debug audio recordings)
 */
function ensureDataDirectories() {
    const homeDir = os.homedir();
    const jefeDir = path.join(homeDir, 'jefe'); // Renamed
    const dataDir = path.join(jefeDir, 'data');
    const imageDir = path.join(dataDir, 'image');
    const audioDir = path.join(dataDir, 'audio');

    [jefeDir, dataDir, imageDir, audioDir].forEach(dir => { // Renamed
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
    });

    return { imageDir, audioDir };
}

/**
 * Creates and configures the main application window (BrowserWindow).
 * This window is transparent, always on top, and loads `index.html`.
 * It also sets up display media request handling and global shortcuts.
 */
function createWindow() {
    // Create the browser window.
    const mainWindow = new BrowserWindow({
        width: 900, // Default width
        height: 400, // Default height
        frame: false,
        transparent: true,
        hasShadow: false,
        alwaysOnTop: true,
        skipTaskbar: true,
        hiddenInMissionControl: true,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            backgroundThrottling: false,
            enableBlinkFeatures: 'GetDisplayMedia',
            webSecurity: true,
            allowRunningInsecureContent: false,
        },
        backgroundColor: '#00000000', // Transparent background for an overlay experience
    });

    // Custom handler for display media requests from the renderer process (e.g., screen capture).
    // It automatically selects the first available screen source.
    // For Windows, 'loopback' audio is requested to capture system audio along with the screen.
    // For other platforms, audio capture is handled differently (e.g., SystemAudioDump on macOS, mic on Linux).
    session.defaultSession.setDisplayMediaRequestHandler(
        (request, callback) => {
            desktopCapturer.getSources({ types: ['screen'] }).then(sources => {
                if (sources && sources.length > 0) {
                    callback({ video: sources[0], audio: process.platform === 'win32' ? 'loopback' : undefined });
                } else {
                    // Handle error or no sources found
                    callback({ error: 'No screen sources found' });
                }
            });
        },
        { useSystemPicker: false } // Set to false to bypass the system picker dialog, true to show it
    );

    // Enable content protection to prevent screen capture of this window by other apps (where supported by OS).
    mainWindow.setContentProtection(true);
    // Make the window visible on all workspaces/virtual desktops.
    mainWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
    
    // Specific always-on-top behavior for Windows to ensure it stays above other windows.
    if (process.platform === 'win32') {
        mainWindow.setAlwaysOnTop(true, 'screen-saver', 1);
    }

    // Load the index.html of the app into the main window.
    mainWindow.loadFile(path.join(__dirname, 'index.html'));

    // --- Global Shortcut Registration ---
    // These shortcuts control window movement, click-through mode, and other app actions.
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width, height } = primaryDisplay.workAreaSize;
    // Define window movement increment based on a percentage of the smaller screen dimension.
    const moveIncrement = Math.floor(Math.min(width, height) * 0.05); // Adjusted to 5% for finer control

    const isMac = process.platform === 'darwin';
    // Modifier key for window movement shortcuts (Alt on macOS, Ctrl on Windows/Linux).
    const modifier = isMac ? 'Alt' : 'Ctrl';

    // Register shortcuts for moving the window Up, Down, Left, Right.
    const shortcuts = [`${modifier}+Up`, `${modifier}+Down`, `${modifier}+Left`, `${modifier}+Right`];
    shortcuts.forEach(accelerator => {
        globalShortcut.register(accelerator, () => {
            // Only move if the window is focused or if it's in click-through mode
            // (allowing movement even if an underlying window is focused).
            if (!mainWindow.isFocused() && !mouseEventsIgnored) return;

            const [currentX, currentY] = mainWindow.getPosition();
            let newX = currentX;
            let newY = currentY;

            switch (accelerator) {
                case `${modifier}+Up`:
                    newY -= moveIncrement;
                    break;
                case `${modifier}+Down`:
                    newY += moveIncrement;
                    break;
                case `${modifier}+Left`:
                    newX -= moveIncrement;
                    break;
                case `${modifier}+Right`:
                    newX += moveIncrement;
                    break;
            }

            mainWindow.setPosition(newX, newY);
        });
    });

    // Register shortcut to trigger window close action (Cmd+\ on macOS, Ctrl+\ elsewhere).
    // This sends an IPC message to the renderer, which then calls the 'window-close' handler.
    const closeShortcut = isMac ? 'Cmd+\\' : 'Ctrl+\\';
    globalShortcut.register(closeShortcut, () => {
        // mainWindow.close(); // Direct close. Consider sending message to renderer if pre-close logic is needed there.
        mainWindow.webContents.send('shortcut-close-window');
    });

    // Register shortcut to toggle mouse event ignoring (click-through mode) (Cmd+M on macOS, Ctrl+M elsewhere).
    const toggleShortcut = isMac ? 'Cmd+M' : 'Ctrl+M';
    globalShortcut.register(toggleShortcut, () => {
        mouseEventsIgnored = !mouseEventsIgnored;
        // When ignoring mouse events, `forward: true` passes them to content below the window.
        mainWindow.setIgnoreMouseEvents(mouseEventsIgnored, { forward: mouseEventsIgnored });
        console.log(mouseEventsIgnored ? 'Mouse events ignored (click-through enabled)' : 'Mouse events enabled (interactive)');
        sendToRenderer('mouse-events-status', mouseEventsIgnored); // Notify renderer of status change
    });

    // Register shortcut for "next step" functionality (Cmd+Enter on macOS, Ctrl+Enter elsewhere).
    // This sends a predefined text message to the active Gemini session.
    const nextStepShortcut = isMac ? 'Cmd+Enter' : 'Ctrl+Enter';
    globalShortcut.register(nextStepShortcut, async () => {
        console.log('Next step shortcut triggered');
        if (geminiSession) {
            try {
                await geminiSession.sendRealtimeInput({ text: 'What should be the next step here' });
                console.log('Sent "next step" message to Gemini');
            } catch (error) {
                console.error('Error sending next step message:', error);
            }
        } else {
            console.log('No active Gemini session for next step command.');
        }
    });

    // --- IPC Event Handlers for Renderer Process Communication ---

    // Listener for 'view-changed' event from renderer.
    // Could be used to adjust main process behavior based on current view in renderer.
    // Currently, it ensures mouse events are re-enabled if not in 'assistant' view,
    // but this behavior might be better handled by the toggle shortcut exclusively.
    ipcMain.on('view-changed', (event, view) => {
        // Example: if (view !== 'assistant' && mouseEventsIgnored) { mainWindow.setIgnoreMouseEvents(false); }
        // This is commented out to let user explicitly control click-through via shortcut.
    });

    // Handles 'window-close' request from renderer (e.g., from a custom close button in the UI).
    ipcMain.handle('window-close', () => {
         mainWindow.close(); // This will trigger 'window-all-closed' if it's the last window.
    });

    // Example IPC handler for window minimization (if a custom button in renderer calls it).
    // ipcMain.handle('window-minimize', () => {
    //     mainWindow.minimize();
    // });
}


/**
 * Initializes a new Google GenAI session for real-time interaction.
 * Configures the session with API key, system prompt (based on profile and custom input),
 * and callbacks for handling messages, errors, and session lifecycle events.
 * @param {string} apiKey - The Google API key for Gemini.
 * @param {string} [customPrompt=''] - Custom instructions to be included in the system prompt.
 * @param {string} [profile='interview'] - The selected user profile (e.g., 'interview', 'jefe') to determine system prompt base.
 * @param {string} [language='en-US'] - The language code for speech recognition.
 * @returns {Promise<boolean>} True if the session was initialized successfully, false otherwise.
 */
async function initializeGeminiSession(apiKey, customPrompt = '', profile = 'interview', language = 'en-US') {
    const client = new GoogleGenAI({
        vertexai: false,
        apiKey: apiKey,
    });

    const systemPrompt = getSystemPrompt(profile, customPrompt);

    try {
        const session = await client.live.connect({
            model: 'gemini-2.0-flash-live-001',
            callbacks: {
                onopen: function () {
                    sendToRenderer('update-status', 'Connected to Gemini - Starting recording...');
                },
                onmessage: function (message) {
                    console.log(message);
                    if (message.serverContent?.modelTurn?.parts) {
                        for (const part of message.serverContent.modelTurn.parts) {
                            console.log(part);
                            if (part.text) {
                                messageBuffer += part.text;
                            }
                        }
                    }

                    if (message.serverContent?.generationComplete) {
                        sendToRenderer('update-response', messageBuffer);
                        messageBuffer = '';
                    }

                    if (message.serverContent?.turnComplete) {
                        sendToRenderer('update-status', 'Listening...');
                    }
                },
                onerror: function (e) {
                    console.debug('Error:', e.message);
                    sendToRenderer('update-status', 'Error: ' + e.message);
                },
                onclose: function (e) {
                    console.debug('Session closed:', e.reason);
                    sendToRenderer('update-status', 'Session closed');
                },
            },
            config: {
                responseModalities: ['TEXT'],
                speechConfig: { languageCode: language },
                systemInstruction: {
                    parts: [{ text: systemPrompt }],
                },
            },
        });

        geminiSession = session;
        return true;
    } catch (error) {
        console.error('Failed to initialize Gemini session:', error);
        return false;
    }
}

/**
 * Sends data to the renderer process of the main window.
 * @param {string} channel - The IPC channel to send data on.
 * @param {any} data - The data to send.
 */
function sendToRenderer(channel, data) {
    const windows = BrowserWindow.getAllWindows();
    if (windows.length > 0) {
        // Assumes the first window is the main one.
        windows[0].webContents.send(channel, data);
    }
}

/**
 * Starts audio capture on macOS using the bundled SystemAudioDump utility.
 * This function spawns SystemAudioDump as a child process and pipes its stdout.
 * Audio data is processed in chunks and sent to Gemini.
 * @returns {boolean} True if audio capture started successfully, false otherwise.
 */
function startMacOSAudioCapture() {
    if (process.platform !== 'darwin') return false; // macOS specific

    console.log('Starting macOS audio capture with SystemAudioDump...');

    let systemAudioPath;
    // Determine path to SystemAudioDump based on whether app is packaged.
    if (app.isPackaged) {
        systemAudioPath = path.join(process.resourcesPath, 'SystemAudioDump');
    } else {
        systemAudioPath = path.join(__dirname, 'SystemAudioDump');
    }

    console.log('SystemAudioDump path:', systemAudioPath);

    systemAudioProc = spawn(systemAudioPath, [], {
        stdio: ['ignore', 'pipe', 'pipe'],
    });

    if (!systemAudioProc.pid) {
        console.error('Failed to start SystemAudioDump');
        return false;
    }

    console.log('SystemAudioDump started with PID:', systemAudioProc.pid);

    const CHUNK_DURATION = 0.1;
    const SAMPLE_RATE = 24000;
    const BYTES_PER_SAMPLE = 2;
    const CHANNELS = 2; // SystemAudioDump outputs stereo
    const CHUNK_SIZE = SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * CHUNK_DURATION;

    let audioBuffer = Buffer.alloc(0); // Buffer to accumulate raw audio data

    // Handle data from SystemAudioDump's stdout
    systemAudioProc.stdout.on('data', data => {
        audioBuffer = Buffer.concat([audioBuffer, data]);

        // Process audio in fixed-size chunks
        while (audioBuffer.length >= CHUNK_SIZE) {
            const chunk = audioBuffer.slice(0, CHUNK_SIZE);
            audioBuffer = audioBuffer.slice(CHUNK_SIZE); // Remaining data for next iteration

            const monoChunk = convertStereoToMono(chunk); // Convert to mono for Gemini
            const base64Data = monoChunk.toString('base64');
            sendAudioToGemini(base64Data); // Send processed chunk to AI

            // Optional: Save audio chunk for debugging
            if (process.env.DEBUG_AUDIO) {
                console.log(`Processed audio chunk: ${chunk.length} bytes`);
                saveDebugAudio(monoChunk, 'system_audio');
            }
        }

        // Prevent buffer from growing indefinitely if chunks are not perfectly aligned
        const maxBufferSize = SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * 1; // Max 1 second buffer
        if (audioBuffer.length > maxBufferSize) {
            console.warn(`Audio buffer trimming from ${audioBuffer.length} to ${maxBufferSize}`);
            audioBuffer = audioBuffer.slice(-maxBufferSize);
        }
    });

    // Handle errors from SystemAudioDump
    systemAudioProc.stderr.on('data', data => {
        console.error('SystemAudioDump stderr:', data.toString());
    });

    systemAudioProc.on('close', code => {
        console.log('SystemAudioDump process closed with code:', code);
        systemAudioProc = null;
    });

    systemAudioProc.on('error', err => {
        console.error('SystemAudioDump process error:', err);
        systemAudioProc = null;
    });

    return true;
}

/**
 * Converts a stereo PCM audio buffer to mono by taking the left channel.
 * Assumes 16-bit little-endian samples.
 * @param {Buffer} stereoBuffer - The stereo audio buffer.
 * @returns {Buffer} The mono audio buffer.
 */
function convertStereoToMono(stereoBuffer) {
    const samples = stereoBuffer.length / 4; // 2 bytes per sample, 2 channels
    const monoBuffer = Buffer.alloc(samples * 2); // 2 bytes per sample, 1 channel

    for (let i = 0; i < samples; i++) {
        const leftSample = stereoBuffer.readInt16LE(i * 4); // Read left channel sample
        monoBuffer.writeInt16LE(leftSample, i * 2);
    }
    return monoBuffer;
}

/**
 * Stops the macOS audio capture process if it's running.
 */
function stopMacOSAudioCapture() {
    if (systemAudioProc) {
        console.log('Stopping SystemAudioDump...');
        systemAudioProc.kill('SIGTERM'); // Send termination signal
        systemAudioProc = null;
    }
}

/**
 * Sends a base64 encoded audio chunk to the active Gemini session.
 * @param {string} base64Data - The base64 encoded PCM audio data.
 */
async function sendAudioToGemini(base64Data) {
    if (!geminiSession) {
        // console.warn('No active Gemini session to send audio.');
        return;
    }
    try {
        // process.stdout.write('.'); // Minimal logging for frequent calls
        await geminiSession.sendRealtimeInput({
            audio: {
                data: base64Data,
                mimeType: 'audio/pcm;rate=24000',
            },
        });
    } catch (error) {
        // console.error('Error sending audio to Gemini:', error); // Can be very verbose
    }
}

// --- Electron App Lifecycle Event Handlers ---

// This method will be called when Electron has finished initialization
// and is ready to create browser windows.
app.whenReady().then(createWindow);

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
    stopMacOSAudioCapture(); // Ensure audio capture stops
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// Handle app exit more gracefully
app.on('before-quit', () => {
    console.log('Application before-quit event');
    stopMacOSAudioCapture();
    if (geminiSession) {
        // Attempt to disconnect session, but don't wait indefinitely
        geminiSession.disconnect().catch(e => console.error("Error disconnecting Gemini on quit:", e));
        geminiSession = null;
    }
});


// On macOS, re-create a window in the app when the dock icon is clicked
// and there are no other windows open.
app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

// --- IPC Handlers for Renderer Process Communication ---

/**
 * Handles renderer request to initialize Gemini session.
 * Exposed to renderer via `window.jefe.initializeGemini` (through preload).
 */
ipcMain.handle('initialize-gemini', async (event, apiKey, customPrompt, profile = 'interview', language = 'en-US') => {
    return await initializeGeminiSession(apiKey, customPrompt, profile, language);
});

/**
 * Handles renderer request to send generic audio content (e.g., from Windows/Linux loopback/mic).
 */
ipcMain.handle('send-audio-content', async (event, { data, mimeType }) => {
    if (!geminiSession) return { success: false, error: 'No active Gemini session' };
    try {
        // process.stdout.write('.'); // Minimal logging
        await geminiSession.sendRealtimeInput({
            audio: { data: data, mimeType: mimeType },
        });
        return { success: true };
    } catch (error) {
        console.error('Error sending audio:', error);
        return { success: false, error: error.message };
    }
});

/**
 * Handles renderer request to send image content (screenshots).
 */
ipcMain.handle('send-image-content', async (event, { data, debug }) => {
    if (!geminiSession) return { success: false, error: 'No active Gemini session' };

    try {
        if (!data || typeof data !== 'string') {
            console.error('Invalid image data received for send-image-content');
            return { success: false, error: 'Invalid image data for send-image-content' };
        }

        // Basic check for empty or very small base64 string
        if (data.length < 100) { // Arbitrary small length
            console.error('Image data too small for send-image-content');
            return { success: false, error: 'Image data too small for send-image-content' };
        }

        // process.stdout.write('!'); // Minimal logging for image send
        await geminiSession.sendRealtimeInput({
            media: { data: data, mimeType: 'image/jpeg' },
        });

        return { success: true };
    } catch (error) {
        console.error('Error sending image:', error);
        return { success: false, error: error.message };
    }
});

/**
 * Handles renderer request to send a text message to the AI.
 */
ipcMain.handle('send-text-message', async (event, text) => {
    if (!geminiSession) return { success: false, error: 'No active Gemini session' };

    try {
        if (!text || typeof text !== 'string' || text.trim().length === 0) {
            console.warn('Attempted to send invalid or empty text message.');
            return { success: false, error: 'Invalid text message provided.' };
        }

        console.log('Sending text message to Gemini:', text);
        await geminiSession.sendRealtimeInput({ text: text.trim() });
        return { success: true };
    } catch (error) {
        console.error('Error sending text:', error);
        return { success: false, error: error.message };
    }
});

/**
 * Handles renderer request to start macOS specific audio capture.
 */
ipcMain.handle('start-macos-audio', async event => {
    if (process.platform !== 'darwin') {
        return { success: false, error: 'macOS audio capture only available on macOS' };
    }
    try {
        const success = startMacOSAudioCapture();
        return { success };
    } catch (error) {
        console.error('Error starting macOS audio capture:', error);
        return { success: false, error: error.message };
    }
});

/**
 * Handles renderer request to stop macOS specific audio capture.
 */
ipcMain.handle('stop-macos-audio', async event => {
    try {
        stopMacOSAudioCapture();
        return { success: true };
    } catch (error) {
        console.error('Error stopping macOS audio capture:', error);
        return { success: false, error: error.message };
    }
});

/**
 * Handles renderer request to close the current AI session (disconnects Gemini, stops audio).
 */
ipcMain.handle('close-session', async event => {
    try {
        stopMacOSAudioCapture(); // Stop macOS audio if running
        // Add similar stops for Windows/Linux loopback/mic if they were separate processes

        if (geminiSession) {
            await geminiSession.disconnect();
            geminiSession = null;
            console.log('Gemini session disconnected.');
        }
        return { success: true };
    } catch (error) {
        console.error('Error closing session:', error);
        return { success: false, error: error.message };
    }
});

/**
 * Handles renderer request to quit the entire application.
 */
ipcMain.handle('quit-application', async event => {
    try {
        // Perform any cleanup before quitting
        await ipcMain.invoke('close-session'); // Ensure session is closed
        app.quit();
        return { success: true }; // Note: app might quit before this is received
    } catch (error) {
        console.error('Error quitting application:', error);
        return { success: false, error: error.message };
    }
});

/**
 * Handles renderer request to open an external URL in the default browser.
 */
ipcMain.handle('open-external', async (event, url) => {
    try {
        await shell.openExternal(url);
        return { success: true };
    } catch (error) {
        console.error('Error opening external URL:', error);
        return { success: false, error: error.message };
    }
});
