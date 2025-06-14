// audioUtils.js
// This module provides utility functions for audio processing,
// such as converting PCM data to WAV format, analyzing audio buffers,
// and saving audio data for debugging purposes.

const fs = require('fs'); // Node.js File System module
const path = require('path'); // Node.js Path module

/**
 * Converts raw PCM (Pulse Code Modulation) audio data to WAV format.
 * This is useful for making raw audio data playable in standard audio players
 * and for verification or debugging.
 *
 * @param {Buffer} pcmBuffer - The buffer containing raw PCM audio data.
 * @param {string} outputPath - The file path where the WAV file will be saved.
 * @param {number} [sampleRate=24000] - The sample rate of the audio (samples per second).
 * @param {number} [channels=1] - The number of audio channels (1 for mono, 2 for stereo).
 * @param {number} [bitDepth=16] - The number of bits per sample (e.g., 16-bit).
 * @returns {string} The outputPath where the WAV file was saved.
 */
function pcmToWav(pcmBuffer, outputPath, sampleRate = 24000, channels = 1, bitDepth = 16) {
    const byteRate = sampleRate * channels * (bitDepth / 8); // Bytes per second
    const blockAlign = channels * (bitDepth / 8); // Bytes per sample frame (all channels)
    const dataSize = pcmBuffer.length; // Size of the PCM data in bytes

    // Create WAV header (44 bytes)
    const header = Buffer.alloc(44);

    // RIFF chunk descriptor
    header.write('RIFF', 0); // ChunkID
    header.writeUInt32LE(dataSize + 36, 4); // ChunkSize (file size - 8 bytes for RIFF and ChunkSize)
    header.write('WAVE', 8); // Format

    // "fmt " sub-chunk (describes the sound data's format)
    header.write('fmt ', 12); // Subchunk1ID
    header.writeUInt32LE(16, 16); // Subchunk1Size (16 for PCM)
    header.writeUInt16LE(1, 20); // AudioFormat (1 for PCM, other numbers indicate compression)
    header.writeUInt16LE(channels, 22); // NumChannels
    header.writeUInt32LE(sampleRate, 24); // SampleRate
    header.writeUInt32LE(byteRate, 28); // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
    header.writeUInt16LE(blockAlign, 32); // BlockAlign (NumChannels * BitsPerSample/8)
    header.writeUInt16LE(bitDepth, 34); // BitsPerSample

    // "data" sub-chunk (contains the actual sound data)
    header.write('data', 36); // Subchunk2ID
    header.writeUInt32LE(dataSize, 40); // Subchunk2Size (NumSamples * NumChannels * BitsPerSample/8)

    // Combine header and PCM data to form the WAV buffer
    const wavBuffer = Buffer.concat([header, pcmBuffer]);

    // Write the WAV buffer to the specified output file
    fs.writeFileSync(outputPath, wavBuffer);

    return outputPath; // Return the path for confirmation
}

/**
 * Analyzes an audio buffer (assumed to be 16-bit PCM) and logs various metrics.
 * This is primarily for debugging audio issues.
 *
 * @param {Buffer} buffer - The audio buffer to analyze.
 * @param {string} [label='Audio'] - A label for the console output.
 * @returns {object} An object containing analysis metrics (minValue, maxValue, avgValue, rmsValue, silencePercentage, sampleCount).
 */
function analyzeAudioBuffer(buffer, label = 'Audio') {
    // Create an Int16Array view over the buffer to read 16-bit samples
    // Assumes buffer.buffer is the underlying ArrayBuffer, buffer.byteOffset is its offset,
    // and buffer.length is the byte length of the view.
    const int16Array = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.length / 2);

    let minValue = 32767; // Max possible Int16 value
    let maxValue = -32768;
    let avgValue = 0;
    let rmsValue = 0;
    let silentSamples = 0;

    for (let i = 0; i < int16Array.length; i++) {
        const sample = int16Array[i];
        minValue = Math.min(minValue, sample);
        maxValue = Math.max(maxValue, sample);
        avgValue += sample;
        rmsValue += sample * sample;

        if (Math.abs(sample) < 100) {
            silentSamples++;
        }
    }

    avgValue /= int16Array.length;
    rmsValue = Math.sqrt(rmsValue / int16Array.length);

    const silencePercentage = (silentSamples / int16Array.length) * 100;

    console.log(`${label} Analysis:`);
    console.log(`  Samples: ${int16Array.length}`);
    console.log(`  Min: ${minValue}, Max: ${maxValue}`);
    console.log(`  Average: ${avgValue.toFixed(2)}`);
    console.log(`  RMS: ${rmsValue.toFixed(2)}`);
    console.log(`  Silence: ${silencePercentage.toFixed(1)}%`);
    console.log(`  Dynamic Range: ${20 * Math.log10(maxValue / (rmsValue || 1))} dB`);

    return {
        minValue,
        maxValue,
        avgValue,
        rmsValue,
        silencePercentage,
        sampleCount: int16Array.length,
    };
}

/**
 * Saves an audio buffer (raw PCM and converted WAV) along with metadata for debugging.
 * Files are saved to a 'debug' subdirectory in the application's data directory (e.g., ~/.jefe/debug).
 *
 * @param {Buffer} buffer - The audio buffer to save.
 * @param {string} type - A type descriptor for the audio (e.g., 'system_audio', 'mic_audio'), used in filenames.
 * @param {number} [timestamp=Date.now()] - Timestamp for the filenames.
 * @returns {object} An object containing paths to the saved PCM, WAV, and metadata JSON files.
 */
function saveDebugAudio(buffer, type, timestamp = Date.now()) {
    const homeDir = require('os').homedir(); // Get user's home directory
    // Construct path to the debug directory within the app's data folder
    const debugDir = path.join(homeDir, 'jefe', 'debug'); // Corrected 'cheddar' to 'jefe'

    // Create the debug directory if it doesn't exist
    if (!fs.existsSync(debugDir)) {
        fs.mkdirSync(debugDir, { recursive: true });
    }

    // Define file paths for PCM, WAV, and metadata
    const pcmPath = path.join(debugDir, `${type}_${timestamp}.pcm`);
    const wavPath = path.join(debugDir, `${type}_${timestamp}.wav`);
    const metaPath = path.join(debugDir, `${type}_${timestamp}.json`);

    // Save the raw PCM buffer directly
    fs.writeFileSync(pcmPath, buffer);

    // Convert the PCM buffer to WAV format and save it
    // Assumes default sampleRate, channels, bitDepth if not specified, which matches typical use.
    pcmToWav(buffer, wavPath);

    // Analyze the audio buffer to get metrics
    const analysis = analyzeAudioBuffer(buffer, type);

    // Create metadata object
    const metadata = {
        timestamp,
        type,
        bufferSize: buffer.length, // Size in bytes
        analysis, // Results from analyzeAudioBuffer
        format: { // Assumed format for the saved PCM/WAV
            sampleRate: 24000, // Default sample rate used in the app
            channels: 1,       // Assuming mono after processing for Gemini
            bitDepth: 16,      // Standard bit depth
        },
    };

    // Save metadata as a JSON file
    fs.writeFileSync(metaPath, JSON.stringify(metadata, null, 2)); // Pretty print JSON

    console.log(`Debug audio saved: ${wavPath}`); // Log path to the playable WAV file

    return { pcmPath, wavPath, metaPath }; // Return paths for potential further use
}

// Export the utility functions for use in other modules
module.exports = {
    pcmToWav,
    analyzeAudioBuffer,
    saveDebugAudio,
};
