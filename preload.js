// preload.js
// This script runs before the renderer process is fully loaded and has access to both
// the DOM/window context and Node.js APIs from the main process side of the IPC bridge.
// It's typically used to expose specific Node.js/Electron functionalities to the renderer
// process in a secure way using `contextBridge.exposeInMainWorld`.

// Currently, this preload script is not exposing any custom APIs to the renderer process.
// If specific main process functionalities or Node.js modules were needed in the renderer
// in a context-isolated environment (contextIsolation: true in BrowserWindow webPreferences),
// they would be exposed here. For example:
//
// const { contextBridge, ipcRenderer } = require('electron');
// contextBridge.exposeInMainWorld('myAPI', {
//   doSomething: () => ipcRenderer.invoke('do-something'),
//   send: (channel, data) => ipcRenderer.send(channel, data),
//   on: (channel, func) => ipcRenderer.on(channel, (event, ...args) => func(...args))
// });
//
// Since contextIsolation is currently false in this project's BrowserWindow settings,
// the renderer process has direct access to Node.js and Electron APIs like `require('electron').ipcRenderer`,
// making explicit context bridging less critical for basic IPC. However, for security best practices,
// enabling contextIsolation and using a preload script for IPC is recommended.

// Default Electron comments:
// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts
