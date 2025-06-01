#!/usr/bin/env node

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { pcmToWav, analyzeAudioBuffer, saveDebugAudio } = require('./audioUtils');

console.log('Audio Test Utility');
console.log('==================\n');

// Test SystemAudioDump for 5 seconds
function testSystemAudio() {
    console.log('Testing SystemAudioDump for 5 seconds...');

    const systemAudioPath = path.join(__dirname, 'SystemAudioDump');

    if (!fs.existsSync(systemAudioPath)) {
        console.error('SystemAudioDump not found at:', systemAudioPath);
        return;
    }

    const systemAudioProc = spawn(systemAudioPath, [], {
        stdio: ['ignore', 'pipe', 'pipe'],
    });

    if (!systemAudioProc.pid) {
        console.error('Failed to start SystemAudioDump');
        return;
    }

    console.log('SystemAudioDump started with PID:', systemAudioProc.pid);

    let audioBuffer = Buffer.alloc(0);
    let chunkCount = 0;
    const startTime = Date.now();

    systemAudioProc.stdout.on('data', data => {
        audioBuffer = Buffer.concat([audioBuffer, data]);
        chunkCount++;
        process.stdout.write(`\rReceived ${chunkCount} chunks, ${audioBuffer.length} bytes`);
    });

    systemAudioProc.stderr.on('data', data => {
        console.error('\nSystemAudioDump stderr:', data.toString());
    });

    // Stop after 5 seconds
    setTimeout(() => {
        systemAudioProc.kill('SIGTERM');

        const duration = (Date.now() - startTime) / 1000;
        console.log(`\n\nCapture completed in ${duration.toFixed(1)} seconds`);
        console.log(`Total data: ${audioBuffer.length} bytes`);
        console.log(`Expected: ~${Math.floor(24000 * 2 * duration)} bytes for ${duration.toFixed(1)}s at 24kHz 16-bit mono`);

        if (audioBuffer.length > 0) {
            // Analyze the captured audio
            console.log('\nAnalyzing captured audio...');
            analyzeAudioBuffer(audioBuffer, 'SystemAudioDump Output');

            // Save as WAV for playback
            const outputDir = path.join(process.env.HOME || process.env.USERPROFILE, 'cheddar', 'test');
            if (!fs.existsSync(outputDir)) {
                fs.mkdirSync(outputDir, { recursive: true });
            }

            const timestamp = Date.now();
            const wavPath = path.join(outputDir, `test_system_${timestamp}.wav`);
            const pcmPath = path.join(outputDir, `test_system_${timestamp}.pcm`);

            fs.writeFileSync(pcmPath, audioBuffer);
            pcmToWav(audioBuffer, wavPath);

            console.log(`\nAudio saved to:`);
            console.log(`  PCM: ${pcmPath}`);
            console.log(`  WAV: ${wavPath}`);
            console.log('\nYou can play the WAV file to verify the audio quality.');

            // Check for common issues
            const int16Array = new Int16Array(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.length / 2);
            let maxValue = 0;
            let silentSamples = 0;

            for (let i = 0; i < int16Array.length; i++) {
                const absValue = Math.abs(int16Array[i]);
                maxValue = Math.max(maxValue, absValue);
                if (absValue < 100) silentSamples++;
            }

            const silencePercentage = (silentSamples / int16Array.length) * 100;

            console.log('\nDiagnostics:');
            if (maxValue < 1000) {
                console.log('⚠️  Audio level is very low. System audio might not be captured properly.');
            } else if (maxValue < 5000) {
                console.log('⚠️  Audio level is low. Consider checking system volume.');
            } else {
                console.log('✓ Audio level appears normal.');
            }

            if (silencePercentage > 90) {
                console.log('⚠️  Audio is mostly silent. Make sure audio is playing during capture.');
            } else if (silencePercentage > 50) {
                console.log('⚠️  Audio has significant silence.');
            } else {
                console.log('✓ Audio contains sufficient signal.');
            }
        } else {
            console.error('\nNo audio data captured!');
        }
    }, 5000);

    systemAudioProc.on('error', err => {
        console.error('SystemAudioDump error:', err);
    });
}

// Run the test
testSystemAudio();
