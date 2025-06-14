import { html, css, LitElement } from './lit-core-2.7.4.min.js';

/**
 * @customElement jefe-app
 * @summary Main application web component for Jefe.
 *
 * This component manages the UI, state, and interactions for the Jefe application.
 * It handles different views (main, customize, help, assistant), user inputs,
 * communication with the main Electron process (via `window.jefe` exposed by renderer.js),
 * and displays AI responses.
 */
class JefeApp extends LitElement {
    static styles = css`
        /* Component styles are defined here */
        * {
            box-sizing: border-box;
            font-family:
                'Inter',
                -apple-system,
                BlinkMacSystemFont,
                sans-serif;
            margin: 0px;
            padding: 0px;
            cursor: default;
        }

        :host {
            display: block;
            width: 100%;
            height: 100vh;
            background-color: var(--background-transparent);
            color: var(--text-color);
        }

        .window-container {
            height: 100vh;
            border-radius: 7px;
            overflow: hidden;
        }

        .container {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .header {
            -webkit-app-region: drag;
            display: flex;
            align-items: center;
            padding: 10px 20px;
            border: 1px solid var(--border-color);
            background: var(--header-background);
            border-radius: 7px;
        }

        .header-title {
            flex: 1;
            font-size: 16px;
            font-weight: 600;
            -webkit-app-region: drag;
        }

        .header-actions {
            display: flex;
            gap: 12px;
            align-items: center;
            -webkit-app-region: no-drag;
        }

        .header-actions span {
            font-size: 13px;
            color: var(--header-actions-color);
        }

        .main-content {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            margin-top: 10px;
            border: 1px solid var(--border-color);
            background: var(--main-content-background);
            border-radius: 7px;
        }

        .button {
            background: var(--button-background);
            color: var(--text-color);
            border: 1px solid var(--button-border);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
        }

        .icon-button {
            background: none;
            color: var(--icon-button-color);
            border: none;
            padding: 8px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            display: flex;
        }

        .icon-button:hover {
            background: var(--hover-background);
        }

        .button:hover {
            background: var(--hover-background);
        }

        button:disabled {
            opacity: 0.5;
        }

        input,
        textarea,
        select {
            background: var(--input-background);
            color: var(--text-color);
            border: 1px solid var(--button-border);
            padding: 10px 14px;
            width: 100%;
            border-radius: 8px;
            font-size: 14px;
        }

        input:focus,
        textarea:focus,
        select:focus {
            outline: none;
            border-color: var(--focus-border-color);
            box-shadow: 0 0 0 3px var(--focus-box-shadow);
            background: var(--input-focus-background);
        }

        input::placeholder,
        textarea::placeholder {
            color: var(--placeholder-color);
        }

        .input-group {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
        }

        .input-group input {
            flex: 1;
        }

        .response-container {
            height: calc(100% - 60px);
            overflow-y: auto;
            white-space: pre-wrap;
            border-radius: 10px;
            font-size: 20px;
            line-height: 1.6;
        }

        .response-container::-webkit-scrollbar {
            width: 8px;
        }

        .response-container::-webkit-scrollbar-track {
            background: var(--scrollbar-track);
            border-radius: 4px;
        }

        .response-container::-webkit-scrollbar-thumb {
            background: var(--scrollbar-thumb);
            border-radius: 4px;
        }

        .response-container::-webkit-scrollbar-thumb:hover {
            background: var(--scrollbar-thumb-hover);
        }

        textarea {
            height: 120px;
            resize: vertical;
            line-height: 1.5;
        }

        .welcome {
            font-size: 24px;
            margin-bottom: 8px;
            font-weight: 600;
            margin-top: auto;
        }

        @keyframes pulse {
            0% {
                opacity: 1;
            }
            50% {
                opacity: 0.8;
            }
            100% {
                opacity: 1;
            }
        }

        .option-group {
            margin-bottom: 24px;
        }

        .option-label {
            display: block;
            margin-bottom: 8px;
            color: var(--option-label-color);
            font-weight: 500;
            font-size: 14px;
        }

        .option-group .description {
            margin-top: 8px;
            margin-bottom: 0;
            font-size: 13px;
            color: var(--description-color);
        }

        .screen-preview {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 12px;
        }

        .screen-option {
            border: 2px solid transparent;
            padding: 8px;
            text-align: center;
            border-radius: 12px;
            background: var(--screen-option-background);
        }

        .screen-option:hover {
            background: var(--screen-option-hover-background);
            border-color: var(--button-border);
        }

        .screen-option.selected {
            border-color: var(--focus-border-color);
            background: var(--screen-option-selected-background);
        }

        .screen-option img {
            width: 150px;
            height: 100px;
            object-fit: contain;
            background: var(--screen-option-background);
            border-radius: 8px;
        }

        .screen-option div {
            font-size: 12px;
            margin-top: 6px;
            color: var(--screen-option-text);
        }

        .selected .screen-option div {
            color: var(--focus-border-color);
            font-weight: 500;
        }

        .description {
            color: var(--description-color);
            font-size: 14px;
            margin-bottom: 24px;
            line-height: 1.5;
        }

        .start-button {
            background: var(--start-button-background);
            color: var(--start-button-color);
            border: 1px solid var(--start-button-border);
        }

        .start-button:hover {
            background: var(--start-button-hover-background);
            border-color: var(--start-button-hover-border);
        }

        .text-input-container {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            align-items: center;
        }

        .text-input-container input {
            flex: 1;
        }

        .text-input-container button {
            background: var(--text-input-button-background);
            color: var(--start-button-background);
            border: none;
        }

        .text-input-container button:hover {
            background: var(--text-input-button-hover);
        }

        .nav-button {
            background: var(--button-background);
            color: var(--text-color);
            border: 1px solid var(--button-border);
            padding: 8px;
            border-radius: 8px;
            font-size: 12px;
            display: flex;
            align-items: center;
            min-width: 32px;
            justify-content: center;
        }

        .nav-button:hover {
            background: var(--hover-background);
        }

        .nav-button:disabled {
            opacity: 0.3;
        }

        .response-counter {
            font-size: 12px;
            color: var(--description-color);
            white-space: nowrap;
            min-width: 60px;
            text-align: center;
        }

        .link {
            color: var(--link-color);
            text-decoration: underline;
        }

        .key {
            background: var(--key-background);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            margin: 0px;
        }

        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: var(--scrollbar-background);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--scrollbar-thumb);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--scrollbar-thumb-hover);
        }
    `;

    // Defines the properties of the component that will trigger re-renders when changed.
    static properties = {
        /** The currently active view (e.g., 'main', 'customize', 'help', 'assistant'). */
        currentView: { type: String },
        /** Text to display in the status bar, typically reflecting AI or system status. */
        statusText: { type: String },
        /** Timestamp (ms) when the assistant session started, for calculating elapsed time. */
        startTime: { type: Number },
        /** Boolean indicating if audio/screen recording is active. */
        isRecording: { type: Boolean }, // Note: This property is declared but not explicitly used in the provided code.
                                        // `startTime` is used to infer session activity for display.
        /** Boolean indicating if an AI session is currently active. */
        sessionActive: { type: Boolean }, // Note: Also seems implicitly managed by view/startTime rather than direct set.
        /** The value of the selected AI profile (e.g., 'interview', 'jefe'). */
        selectedProfile: { type: String },
        /** The selected language for speech recognition and AI responses (e.g., 'en-US'). */
        selectedLanguage: { type: String },
        /** Array storing the AI's responses during a session. */
        responses: { type: Array },
        /** Index of the currently viewed response in the `responses` array. */
        currentResponseIndex: { type: Number },
    };

    constructor() {
        super();
        // Initialize component state
        this.currentView = 'main'; // Default view on load
        this.statusText = ''; // Initial status text
        this.startTime = null; // Time when assistant session begins
        this.isRecording = false; // Reflects recording state (though not directly driving logic in this snippet)
        this.sessionActive = false; // Reflects if an AI session is active (though not directly driving logic)

        // Load saved preferences from localStorage or use defaults
        this.selectedProfile = localStorage.getItem('selectedProfile') || 'interview';
        this.selectedLanguage = localStorage.getItem('selectedLanguage') || 'en-US';

        this.responses = []; // Array to hold AI responses
        this.currentResponseIndex = -1; // Index for navigating responses, -1 means no responses yet or not viewing one

        // Note: Event listeners for IPC messages ('update-response', 'update-status')
        // are set up in renderer.js and directly call methods like this.setStatus, this.setResponse.
        // Consider moving those listeners into connectedCallback if more complex lifecycle management is needed.
    }

    /**
     * Handles the custom close button action by invoking an IPC call to the main process.
     * This is typically used by the 'x' button in the custom header.
     */
    async handleWindowClose() {
        const { ipcRenderer } = window.require('electron');
        await ipcRenderer.invoke('window-close'); // Asks main process to close the window
    }

    /**
     * LitElement lifecycle method called when the element is connected to the DOM.
     * Used for setup tasks.
     */
    connectedCallback() {
        super.connectedCallback();
        // Event listeners for IPC are handled in renderer.js currently.
        // If this component needed to directly listen to IPC without the global `jefe.e()` proxy,
        // listeners could be added here. Example:
        // window.electronIPC.on('update-status', this.handleStatusUpdate.bind(this));
    }

    /**
     * Updates the status text displayed in the UI.
     * @param {string} t - The new status text.
     */
    setStatus(t) {
        this.statusText = t;
    }

    setResponse(r) {
        this.responses.push(r);

        // If user is viewing the latest response (or no responses yet), auto-navigate to new response
        if (this.currentResponseIndex === this.responses.length - 2 || this.currentResponseIndex === -1) {
            this.currentResponseIndex = this.responses.length - 1;
        }

        this.requestUpdate();
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        // It's good practice to remove listeners here if they were added in connectedCallback.
        // However, the current IPC listeners ('update-response', 'update-status') are global
        // and managed in renderer.js, not directly on this component instance.
        // If they were instance-specific, they'd be removed here:
        // window.electronIPC.removeListener('update-status', this.handleStatusUpdate.bind(this));

        // The provided code has ipcRenderer.removeAllListeners in the web component,
        // which might be unexpected if renderer.js also sets them up.
        // This could lead to conflicts if not carefully managed.
        // Assuming these are for global listeners that the component *itself* might have set up
        // if it were directly using ipcRenderer (which it isn't for receiving, only for sending via handleWindowClose).
        // For clarity, if these listeners are meant for `jefe.e().setStatus` etc., they are managed by `ipcRenderer.on` in `renderer.js`.
        // This component's methods are called by those global handlers.
        // const { ipcRenderer } = window.require('electron');
        // ipcRenderer.removeAllListeners('update-response'); // These might be redundant or misplaced
        // ipcRenderer.removeAllListeners('update-status');   // if listeners are in renderer.js
    }

    /**
     * Generic handler for input changes that save the value to localStorage.
     * @param {Event} e - The input event from an input element.
     * @param {string} property - The localStorage key to save the value under.
     */
    handleInput(e, property) {
        localStorage.setItem(property, e.target.value);
    }

    /**
     * Handles changes to the profile selection dropdown.
     * Updates the `selectedProfile` property and saves it to localStorage.
     * @param {Event} e - The change event from the select element.
     */
    handleProfileSelect(e) {
        this.selectedProfile = e.target.value;
        localStorage.setItem('selectedProfile', this.selectedProfile);
    }

    /**
     * Handles changes to the language selection dropdown.
     * Updates the `selectedLanguage` property and saves it to localStorage.
     * @param {Event} e - The change event from the select element.
     */
    handleLanguageSelect(e) {
        this.selectedLanguage = e.target.value;
        localStorage.setItem('selectedLanguage', this.selectedLanguage);
    }

    /**
     * Handles the "Start Session" button click.
     * Initializes the Gemini AI session via `window.jefe.initializeGemini`,
     * starts media capture via `window.jefe.startCapture`,
     * resets responses, and switches to the 'assistant' view.
     */
    async handleStart() {
        // Calls functions exposed by renderer.js on the window.jefe object
        await window.jefe.initializeGemini(this.selectedProfile, this.selectedLanguage);
        window.jefe.startCapture();
        this.responses = []; // Clear previous responses
        this.currentResponseIndex = -1; // Reset response navigation
        this.startTime = Date.now(); // Set session start time for elapsed time display
        this.currentView = 'assistant'; // Switch to the assistant view
    }

    /**
     * Handles close/back actions based on the current view.
     * - If in 'customize' or 'help' view, navigates back to 'main' view.
     * - If in 'assistant' view, stops media capture, closes the AI session (via IPC),
     *   and navigates back to 'main' view.
     * - If in 'main' view (or any other unexpected view), invokes IPC to quit the application.
     */
    async handleClose() {
        if (this.currentView === 'customize' || this.currentView === 'help') {
            this.currentView = 'main';
        } else if (this.currentView === 'assistant') {
            window.jefe.stopCapture(); // Stop media capture
            this.startTime = null; // Reset start time

            // Close the AI session via IPC call to the main process
            const { ipcRenderer } = window.require('electron');
            try {
                await ipcRenderer.invoke('close-session');
                this.sessionActive = false; // Update session status (though not directly used elsewhere yet)
                console.log('AI session closed via IPC.');
            } catch (error) {
                console.error("Error invoking 'close-session':", error);
            }
            this.currentView = 'main';
        } else {
            // If in main view or an unknown state, quit the application
            const { ipcRenderer } = window.require('electron');
            await ipcRenderer.invoke('quit-application');
        }
    }

    /**
     * Switches the view to 'help'.
     */
    async openHelp() {
        this.currentView = 'help';
    }

    /**
     * Opens a help link for obtaining an API key in the default external browser.
     * Uses IPC to request the main process to open the URL.
     */
    async openAPIKeyHelp() {
        const { ipcRenderer } = window.require('electron');
        // TODO: Update this URL if a specific help page is created for Jefe
        await ipcRenderer.invoke('open-external', 'https://jefe.com/help/api-key');
    }

    /**
     * Opens an arbitrary external URL in the default browser.
     * Uses IPC to request the main process to open the URL.
     * @param {string} url - The URL to open.
     */
    async openExternalLink(url) {
        const { ipcRenderer } = window.require('electron');
        await ipcRenderer.invoke('open-external', url);
    }

    /**
     * Scrolls the response container to the bottom.
     * This is typically called after new content is added or view changes.
     * Uses a `setTimeout` to ensure the DOM has updated before scrolling.
     */
    scrollToBottom() {
        setTimeout(() => {
            const container = this.shadowRoot.querySelector('.response-container');
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        }, 0); // Timeout helps ensure DOM update before scroll calculation
    }

    /**
     * LitElement lifecycle method called after the element's DOM has been updated.
     * Used here to scroll to bottom if the view changes and to notify the main
     * process about view changes.
     * @param {Map} changedProperties - A Map of properties that changed.
     */
    updated(changedProperties) {
        super.updated(changedProperties);
        if (changedProperties.has('currentView')) {
            this.scrollToBottom();
        }

        // Notify main process of view change. This helps the main process
        // potentially adjust behavior (e.g., mouse event handling, though currently disabled there).
        if (changedProperties.has('currentView')) {
            const { ipcRenderer } = window.require('electron');
            ipcRenderer.send('view-changed', this.currentView);
        }
    }

    /**
     * Handles sending a text message typed by the user to the AI.
     * It retrieves the text from the input field, clears the field,
     * and uses the `window.jefe.sendTextMessage` (exposed by renderer.js)
     * to send the message via IPC to the main process.
     */
    async handleSendText() {
        const textInput = this.shadowRoot.querySelector('#textInput');
        if (textInput && textInput.value.trim()) {
            const message = textInput.value.trim();
            textInput.value = ''; // Clear the input field

            // Send the message using the exposed jefe API from renderer.js
            const result = await window.jefe.sendTextMessage(message);

            if (!result.success) {
                // Display error if sending failed
                console.error('Failed to send message:', result.error);
                this.setStatus('Error sending message: ' + result.error);
            } else {
                this.setStatus('Message sent...'); // Update status on successful send
            }
        }
    }

    /**
     * Handles the 'keydown' event on the text input field.
     * If the Enter key is pressed without Shift, it prevents default newline behavior
     * and calls `handleSendText` to send the message.
     * @param {KeyboardEvent} e - The keyboard event.
     */
    handleTextKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent default action (e.g., adding a newline)
            this.handleSendText(); // Send the message
        }
    }

    /**
     * Renders the header section of the application.
     * The title and available actions change based on the `currentView`.
     * @returns {import('lit-html').TemplateResult} The HTML template for the header.
     */
    renderHeader() {
        const titles = {
            main: 'Jefe',             // Title for the main/welcome view
            customize: 'Customize',   // Title for the settings/customization view
            help: 'Help & Shortcuts', // Title for the help view
            assistant: 'Jefe',        // Title when the assistant session is active
        };

        let elapsedTime = '';
        // Calculate and display elapsed time if in assistant view and session has started
        if (this.currentView === 'assistant' && this.startTime) {
            const elapsedSeconds = Math.floor((Date.now() - this.startTime) / 1000);
            // Simple HH:MM:SS formattter (can be more sophisticated if needed)
            const h = Math.floor(elapsedSeconds / 3600).toString().padStart(2, '0');
            const m = Math.floor((elapsedSeconds % 3600) / 60).toString().padStart(2, '0');
            const s = (elapsedSeconds % 60).toString().padStart(2, '0');
            if (elapsedSeconds >= 3600) elapsedTime = `${h}:${m}:${s}`;
            else elapsedTime = `${m}:${s}`;
        }

        return html`
            <div class="header">
                <div class="header-title">${titles[this.currentView]}</div>
                <div class="header-actions">
                    ${this.currentView === 'assistant' // Display elapsed time and status only in assistant view
                        ? html`
                              <span>${elapsedTime}</span>
                              <span>${this.statusText}</span>
                          `
                        : ''}
                    ${this.currentView === 'main' // Display customize and help buttons only in main view
                        ? html`
                              <button class="icon-button" title="Customize" @click=${() => (this.currentView = 'customize')}>
                                  <!-- Settings/Customize Icon SVG -->
                                  <?xml version="1.0" encoding="UTF-8"?><svg
                                      width="24px"
                                      height="24px"
                                      stroke-width="1.7"
                                      viewBox="0 0 24 24"
                                      fill="none"
                                      xmlns="http://www.w3.org/2000/svg"
                                      color="currentColor"
                                  >
                                      <path
                                          d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z"
                                          stroke="currentColor"
                                          stroke-width="1.7"
                                          stroke-linecap="round"
                                          stroke-linejoin="round"
                                      ></path>
                                      <path
                                          d="M19.6224 10.3954L18.5247 7.7448L20 6L18 4L16.2647 5.48295L13.5578 4.36974L12.9353 2H10.981L10.3491 4.40113L7.70441 5.51596L6 4L4 6L5.45337 7.78885L4.3725 10.4463L2 11V13L4.40111 13.6555L5.51575 16.2997L4 18L6 20L7.79116 18.5403L10.397 19.6123L11 22H13L13.6045 19.6132L16.2551 18.5155C16.6969 18.8313 18 20 18 20L20 18L18.5159 16.2494L19.6139 13.598L21.9999 12.9772L22 11L19.6224 10.3954Z"
                                          stroke="currentColor"
                                          stroke-width="1.7"
                                          stroke-linecap="round"
                                          stroke-linejoin="round"
                                      ></path>
                                  </svg>
                              </button>
                              <button class="icon-button" title="Help" @click=${this.openHelp}>
                                  <!-- Help/Question Icon SVG -->
                                  <?xml version="1.0" encoding="UTF-8"?><svg
                                      width="24px"
                                      height="24px"
                                      stroke-width="1.7"
                                      viewBox="0 0 24 24"
                                      fill="none"
                                      xmlns="http://www.w3.org/2000/svg"
                                      color="currentColor"
                                  >
                                      <path
                                          d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z"
                                          stroke="currentColor"
                                          stroke-width="1.7"
                                          stroke-linecap="round"
                                          stroke-linejoin="round"
                                      ></path>
                                      <path
                                          d="M9 9C9 5.49997 14.5 5.5 14.5 9C14.5 11.5 12 10.9999 12 13.9999"
                                          stroke="currentColor"
                                          stroke-width="1.7"
                                          stroke-linecap="round"
                                          stroke-linejoin="round"
                                      ></path>
                                      <path
                                          d="M12 18.01L12.01 17.9989"
                                          stroke="currentColor"
                                          stroke-width="1.7"
                                          stroke-linecap="round"
                                          stroke-linejoin="round"
                                      ></path>
                                  </svg>
                              </button>
                          `
                        : ''}
                    ${this.currentView === 'assistant' // Display "Back" button in assistant view
                        ? html`
                              <button @click=${this.handleClose} class="button window-close" title="End Session (Cmd/Ctrl + \\)">
                                  Back&nbsp;&nbsp;<span class="key" style="pointer-events: none;">${jefe.isMacOS ? 'Cmd' : 'Ctrl'}</span>&nbsp;&nbsp;<span class="key"
                                      >&bsol;</span
                                  >
                              </button>
                          `
                        : html`
                              <!-- Display "Close" (X) icon button in other views (main, customize, help) -->
                              <button @click=${this.handleClose} class="icon-button window-close" title="Close Window or Go Back (Cmd/Ctrl + \\)">
                                  <!-- Close/X Icon SVG -->
                                  <?xml version="1.0" encoding="UTF-8"?><svg
                                      width="24px"
                                      height="24px"
                                      stroke-width="1.7"
                                      viewBox="0 0 24 24"
                                      fill="none"
                                      xmlns="http://www.w3.org/2000/svg"
                                      color="currentColor"
                                  >
                                      <path
                                          d="M6.75827 17.2426L12.0009 12M17.2435 6.75736L12.0009 12M12.0009 12L6.75827 6.75736M12.0009 12L17.2435 17.2426"
                                          stroke="currentColor"
                                          stroke-width="1.7"
                                          stroke-linecap="round"
                                          stroke-linejoin="round"
                                      ></path>
                                  </svg>
                              </button>
                          `}
                </div>
            </div>
        `;
    }

    /**
     * Renders the main view of the application.
     * This view typically includes an API key input and a "Start Session" button.
     * @returns {import('lit-html').TemplateResult} The HTML template for the main view.
     */
    renderMainView() {
        return html`
            <div style="height: 100%; display: flex; flex-direction: column; width: 100%; max-width: 500px;">
                <div class="welcome">Welcome</div>

                <div class="input-group">
                    <input
                        type="password"
                        placeholder="Enter your Gemini API Key"
                        .value=${localStorage.getItem('apiKey') || ''}
                        @input=${e => this.handleInput(e, 'apiKey')}
                    />
                    <button @click=${this.handleStart} class="button start-button">Start Session</button>
                </div>
                <p class="description">
                    dont have an api key?
                    <span @click=${this.openAPIKeyHelp} class="link">get one here</span>
                </p>
            </div>
        `;
    }

    /**
     * Renders the customization view.
     * This view allows users to select AI profiles, language, and set custom AI behavior prompts.
     * @returns {import('lit-html').TemplateResult} The HTML template for the customize view.
     */
    renderCustomizeView() {
        const profiles = [ // Available AI profiles
            {
                value: 'interview',
                name: 'Job Interview',
                description: 'Get help with answering interview questions',
            },
            {
                value: 'jefe',
                name: 'Jefe (Coding Asst.)',
                description: 'Proactive coding assistance based on screen and audio',
            },
            {
                value: 'sales',
                name: 'Sales Call',
                description: 'Assist with sales conversations and objection handling',
            },
            {
                value: 'meeting',
                name: 'Business Meeting',
                description: 'Support for professional meetings and discussions',
            },
            {
                value: 'presentation',
                name: 'Presentation',
                description: 'Help with presentations and public speaking',
            },
            {
                value: 'negotiation',
                name: 'Negotiation',
                description: 'Guidance for business negotiations and deals',
            },
        ];

        const languages = [
            { value: 'en-US', name: 'English (US)' },
            { value: 'en-GB', name: 'English (UK)' },
            { value: 'en-AU', name: 'English (Australia)' },
            { value: 'en-IN', name: 'English (India)' },
            { value: 'de-DE', name: 'German (Germany)' },
            { value: 'es-US', name: 'Spanish (United States)' },
            { value: 'es-ES', name: 'Spanish (Spain)' },
            { value: 'fr-FR', name: 'French (France)' },
            { value: 'fr-CA', name: 'French (Canada)' },
            { value: 'hi-IN', name: 'Hindi (India)' },
            { value: 'pt-BR', name: 'Portuguese (Brazil)' },
            { value: 'ar-XA', name: 'Arabic (Generic)' },
            { value: 'id-ID', name: 'Indonesian (Indonesia)' },
            { value: 'it-IT', name: 'Italian (Italy)' },
            { value: 'ja-JP', name: 'Japanese (Japan)' },
            { value: 'tr-TR', name: 'Turkish (Turkey)' },
            { value: 'vi-VN', name: 'Vietnamese (Vietnam)' },
            { value: 'bn-IN', name: 'Bengali (India)' },
            { value: 'gu-IN', name: 'Gujarati (India)' },
            { value: 'kn-IN', name: 'Kannada (India)' },
            { value: 'ml-IN', name: 'Malayalam (India)' },
            { value: 'mr-IN', name: 'Marathi (India)' },
            { value: 'ta-IN', name: 'Tamil (India)' },
            { value: 'te-IN', name: 'Telugu (India)' },
            { value: 'nl-NL', name: 'Dutch (Netherlands)' },
            { value: 'ko-KR', name: 'Korean (South Korea)' },
            { value: 'cmn-CN', name: 'Mandarin Chinese (China)' },
            { value: 'pl-PL', name: 'Polish (Poland)' },
            { value: 'ru-RU', name: 'Russian (Russia)' },
            { value: 'th-TH', name: 'Thai (Thailand)' },
        ];

        const profileNames = {
            jefe: 'Jefe (Coding Asst.)',
            interview: 'Job Interview',
            sales: 'Sales Call',
            meeting: 'Business Meeting',
            presentation: 'Presentation',
            negotiation: 'Negotiation',
        };

        return html`
            <div>
                <div class="option-group">
                    <label class="option-label">Select Profile</label>
                    <select .value=${this.selectedProfile} @change=${this.handleProfileSelect}>
                        ${profiles.map(
                            profile => html`
                                <option value=${profile.value} ?selected=${this.selectedProfile === profile.value}>${profile.name}</option>
                            `
                        )}
                    </select>
                    <div class="description">${profiles.find(p => p.value === this.selectedProfile)?.description || ''}</div>
                </div>

                <div class="option-group">
                    <label class="option-label">Select Language</label>
                    <select .value=${this.selectedLanguage} @change=${this.handleLanguageSelect}>
                        ${languages.map(
                            language => html`
                                <option value=${language.value} ?selected=${this.selectedLanguage === language.value}>${language.name}</option>
                            `
                        )}
                    </select>
                    <div class="description">Choose the language for speech recognition and AI responses.</div>
                </div>

                <div class="option-group">
                    <span class="option-label">AI Behavior for ${profileNames[this.selectedProfile] || 'Selected Profile'}</span>
                    <textarea
                        placeholder="Describe how you want the AI to behave..."
                        .value=${localStorage.getItem('customPrompt') || ''}
                        class="custom-prompt-textarea"
                        rows="4"
                        @input=${e => this.handleInput(e, 'customPrompt')}
                    ></textarea>
                    <div class="description">
                        This custom prompt will be added to the ${profileNames[this.selectedProfile] || 'selected profile'} instructions to
                        personalize the AI's behavior.
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Renders the help view.
     * This view displays information about community links, keyboard shortcuts, how to use,
     * supported profiles, and audio input methods.
     * @returns {import('lit-html').TemplateResult} The HTML template for the help view.
     */
    renderHelpView() {
        return html`
            <div>
                <div class="option-group">
                    <span class="option-label">Community & Support</span>
                    <div class="description">
                        <span @click=${() => this.openExternalLink('https://github.com/sohzm/jefe')} class="link">📂 GitHub Repository</span
                        ><br />
                        <span @click=${() => this.openExternalLink('https://discord.gg/GCBdubnXfJ')} class="link">💬 Join Discord Community</span>
                    </div>
                </div>

                <div class="option-group">
                    <span class="option-label">Keyboard Shortcuts</span>
                    <div class="description">
                        <strong>Window Movement:</strong><br />
                        <span class="key">${jefe.isMacOS ? 'Option' : 'Ctrl'}</span> + Arrow Keys - Move the window in 45px increments<br /><br />

                        <strong>Window Control:</strong><br />
                        <span class="key">${jefe.isMacOS ? 'Cmd' : 'Ctrl'}</span> + <span class="key">M</span> - Toggle mouse events (click-through
                        mode)<br />
                        <span class="key">${jefe.isMacOS ? 'Cmd' : 'Ctrl'}</span> + <span class="key">&bsol;</span> - Close window or go back<br /><br />

                        <strong>Text Input:</strong><br />
                        <span class="key">Enter</span> - Send text message to AI<br />
                        <span class="key">Shift</span> + <span class="key">Enter</span> - New line in text input
                    </div>
                </div>

                <div class="option-group">
                    <span class="option-label">How to Use</span>
                    <div class="description">
                        1. <strong>Start a Session:</strong> Enter your Gemini API key and click "Start Session"<br />
                        2. <strong>Customize:</strong> Choose your profile and language in the settings<br />
                        3. <strong>Position Window:</strong> Use keyboard shortcuts to move the window to your desired location<br />
                        4. <strong>Click-through Mode:</strong> Use <span class="key">${jefe.isMacOS ? 'Cmd' : 'Ctrl'}</span> +
                        <span class="key">M</span> to make the window click-through<br />
                        5. <strong>Get AI Help:</strong> The AI will analyze your screen and audio to provide assistance<br />
                        6. <strong>Text Messages:</strong> Type questions or requests to the AI using the text input
                    </div>
                </div>

                <div class="option-group">
                    <span class="option-label">Supported Profiles</span>
                    <div class="description">
                        <strong>Job Interview:</strong> Get help with interview questions and responses<br />
                        <strong>Sales Call:</strong> Assistance with sales conversations and objection handling<br />
                        <strong>Business Meeting:</strong> Support for professional meetings and discussions<br />
                        <strong>Presentation:</strong> Help with presentations and public speaking<br />
                        <strong>Negotiation:</strong> Guidance for business negotiations and deals
                    </div>
                </div>

                <div class="option-group">
                    <span class="option-label">Audio Input</span>
                    <div class="description">
                        ${jefe.isMacOS
                            ? html`<strong>macOS:</strong> Uses SystemAudioDump for system audio capture`
                            : jefe.isLinux
                              ? html`<strong>Linux:</strong> Uses microphone input`
                              : html`<strong>Windows:</strong> Uses loopback audio capture`}<br />
                        The AI listens to conversations and provides contextual assistance based on what it hears.
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Renders the assistant view, which is active during an AI session.
     * Displays the AI's responses, provides navigation for multiple responses,
     * and includes a text input for sending messages to the AI.
     * @returns {import('lit-html').TemplateResult} The HTML template for the assistant view.
     */
    renderAssistantView() {
        // Determine audio input type string for display (informational)
        let audioInputType = 'Microphone'; // Default
        if (jefe.isMacOS) {
            audioInputType = 'System Audio';
        } else if (jefe.isLinux) {
            audioInputType = 'Microphone'; // Explicitly microphone for Linux as per current setup
        } else { // Windows
            audioInputType = 'System Audio (Loopback)';
        }

        const profileNames = { // TODO: This could be part of a shared config or profile definition
            interview: 'Job Interview',
            sales: 'Sales Call',
            meeting: 'Business Meeting',
            presentation: 'Presentation',
            negotiation: 'Negotiation',
            jefe: 'Jefe (Coding Asst.)'
        };

        // const activeInputs = [audioInputType, 'Screen']; // Informational, not directly used in this render

        // Determine the current response to display, or a default welcome/status message
        const currentResponse =
            this.responses.length > 0 && this.currentResponseIndex >= 0 && this.currentResponseIndex < this.responses.length
                ? this.responses[this.currentResponseIndex]
                : `Hey, I'm listening to your ${profileNames[this.selectedProfile] || this.selectedProfile || 'session'}.`;

        // Format the response counter (e.g., "1/5")
        const responseCounterText = this.responses.length > 0
            ? `${this.currentResponseIndex + 1}/${this.responses.length}`
            : '';

        return html`
            <div style="height: 100%; display: flex; flex-direction: column;">
                <!-- Container for AI responses -->
                <div class.response-container">${currentResponse}</div>

                <!-- Text input and navigation controls -->
                <div class="text-input-container">
                    <button
                        class="nav-button"
                        @click=${this.navigateToPreviousResponse}
                        ?disabled=${this.currentResponseIndex <= 0}
                        title="Previous response (Cmd/Ctrl + Left Arrow - if implemented globally)"
                    >
                        ←
                    </button>

                    <!-- Display response counter if there are responses -->
                    ${this.responses.length > 0 ? html` <span class="response-counter">${responseCounterText}</span> ` : ''}

                    <input type="text" id="textInput" placeholder="Type a message to the AI..." @keydown=${this.handleTextKeydown} />

                    <button
                        class="nav-button"
                        @click=${this.navigateToNextResponse}
                        ?disabled=${this.currentResponseIndex >= this.responses.length - 1}
                        title="Next response (Cmd/Ctrl + Right Arrow - if implemented globally)"
                    >
                        →
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Navigates to the previous AI response in the `responses` array.
     * Decrements `currentResponseIndex` if not already at the first response.
     */
    navigateToPreviousResponse() {
        if (this.currentResponseIndex > 0) {
            this.currentResponseIndex--;
            // this.requestUpdate(); // LitElement usually handles this automatically for property changes
        }
    }

    /**
     * Navigates to the next AI response in the `responses` array.
     * Increments `currentResponseIndex` if not already at the last response.
     */
    navigateToNextResponse() {
        if (this.currentResponseIndex < this.responses.length - 1) {
            this.currentResponseIndex++;
            // this.requestUpdate(); // LitElement usually handles this automatically
        }
    }

    /**
     * Main render method for the component.
     * It determines which view to render based on the `currentView` property.
     * @returns {import('lit-html').TemplateResult} The HTML template for the currently active view.
     */
    render() {
        const views = { // Map view names to their render methods
            main: this.renderMainView(),
            customize: this.renderCustomizeView(),
            help: this.renderHelpView(),
            assistant: this.renderAssistantView(),
        };

        return html`
            <div class="window-container">
                <div class="container">
                    ${this.renderHeader()}
                    <div class="main-content">${views[this.currentView]}</div>
                </div>
            </div>
        `;
    }
}

customElements.define('jefe-app', JefeApp);
