INJECT_JS = """
function initWebSocketFunctions() {
    window.audioCtx = null;
    window.currentWs = null;
    window.nextStartTime = 0;
    window.lastTextSentIndex = 0;
    
    // metrics variables
    window.reqStartTime = 0;
    window.firstChunkTime = 0;
    window.totalAudioSeconds = 0;
    
    function logMsg(msg) {
        let el = document.getElementById('ws-status-logs');
        if(el) {
            el.innerHTML += '> ' + msg + '<br/>';
            el.scrollTop = el.scrollHeight;
        }
    }
    
    window.stopWsStream = function() {
        if(window.currentWs) {
            window.currentWs.send(JSON.stringify({ op: "stop" }));
            window.currentWs.close();
            window.currentWs = null;
        }
        if(window.audioCtx) {
            window.audioCtx.close();
            window.audioCtx = null;
        }
        let el = document.getElementById('ws-status-logs');
        if(el) el.innerHTML = '> Stream stopped & buffer purged.<br/>';
    };

    let typingTimer = null;
    window.onTextInputChange = function(text) {
        if (window.currentWs && window.currentWs.readyState === WebSocket.OPEN) {
            if (text.length < window.lastTextSentIndex) {
                // User deleted text, reset index
                window.lastTextSentIndex = text.length;
                return;
            }
            let delta = text.substring(window.lastTextSentIndex);
            if (delta.length > 0) {
                window.currentWs.send(JSON.stringify({ op: "text", value: delta }));
                window.lastTextSentIndex = text.length;
            }
            if (typingTimer) clearTimeout(typingTimer);
            typingTimer = setTimeout(() => {
                if (window.currentWs && window.currentWs.readyState === WebSocket.OPEN) {
                    window.flushWsStream();
                }
            }, 1000);
        }
    };

    window.flushWsStream = function() {
        if (window.currentWs && window.currentWs.readyState === WebSocket.OPEN) {
            window.currentWs.send(JSON.stringify({ op: "flush" }));
            logMsg('Sent flush signal to finalize remaining text buffer...');
        }
    };
    
    window.startWebSocketStream = function(text, model, voice, lang, variant, speed, split_choice, custom_regex, temp, top_k, top_p, rep_pen, seed, cfg, exaggeration, extras) {
        window.stopWsStream(); 
        
        window.reqStartTime = performance.now();
        window.firstChunkTime = 0;
        window.totalAudioSeconds = 0;
        window.lastTextSentIndex = text.length;
        
        window.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
        window.nextStartTime = window.audioCtx.currentTime + 0.1;
        
        let protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        let wsUrl = protocol + '//' + window.location.host + '/ws/stream';
        
        logMsg('Initializing Real-time Stream to ' + wsUrl + '...');
        window.currentWs = new WebSocket(wsUrl);
        window.currentWs.binaryType = "arraybuffer";
        
        window.currentWs.onopen = function() {
            logMsg('Protocol: [op: start] | Engine: ' + model + ' | Variant: ' + variant);
            let payload = {
                op: "start",
                model: model, voice: voice, lang: lang, variant: variant, speed: parseFloat(speed),
                split_choice: split_choice, custom_regex: custom_regex,
                temp: parseFloat(temp),
                top_k: parseInt(top_k),
                top_p: parseFloat(top_p),
                rep_pen: parseFloat(rep_pen),
                seed: parseInt(seed),
                cfg: parseFloat(cfg),
                exaggeration: parseFloat(exaggeration)
            };
            // Add extras with indices so backend can map them
            if (extras && extras.length) {
                extras.forEach((val, idx) => {
                    payload["extra_" + idx] = val;
                });
            }
            window.currentWs.send(JSON.stringify(payload));
            
            // Immediately send current text and auto-flush it 
            if (text.length > 0) {
                window.currentWs.send(JSON.stringify({ op: "text", value: text }));
                window.lastTextSentIndex = text.length;
                setTimeout(() => { 
                    window.flushWsStream(); 
                }, 400);
            }
        };
        
        let chunkCount = 0;
        window.currentWs.onmessage = async function(event) {
            if (event.data.byteLength === 0) {
                // b'' from python backend acts as a ping indicating buffer was flushed successfully
                let finalRtf = (performance.now() - window.reqStartTime) / 1000 / window.totalAudioSeconds;
                logMsg(`<b>[Buffer Flushed]</b> Synthesizer caught up. Total Audio Generated: ${window.totalAudioSeconds.toFixed(2)}s | Session RTF: ${finalRtf.toFixed(3)}`);
                return;
            }
            
            if (window.firstChunkTime === 0) {
                window.firstChunkTime = performance.now();
                logMsg('<b>[Metric]</b> Time To First Byte (TTFB): ' + (window.firstChunkTime - window.reqStartTime).toFixed(2) + ' ms');
            }
            
            chunkCount++;
            let float32Data = new Float32Array(event.data);
            let audioBuffer = window.audioCtx.createBuffer(1, float32Data.length, 24000);
            audioBuffer.getChannelData(0).set(float32Data);
            
            window.totalAudioSeconds += audioBuffer.duration;
            let rtf = ((performance.now() - window.reqStartTime) / 1000) / window.totalAudioSeconds;
            
            logMsg(`▶ <b>Chunk ${chunkCount}</b>: ${audioBuffer.duration.toFixed(2)}s audio received. (Current Pipeline RTF: ${rtf.toFixed(3)})`);
            
            let source = window.audioCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(window.audioCtx.destination);
            
            let currentTime = window.audioCtx.currentTime;
            if (window.nextStartTime < currentTime) {
                window.nextStartTime = currentTime + 0.05;
            }
            
            source.start(window.nextStartTime);
            window.nextStartTime += audioBuffer.duration;
        };
        
        window.currentWs.onerror = function() { logMsg('WebSocket Error.'); };
        window.currentWs.onclose = function() { logMsg('Connection closed.'); };
    };
    
    return [];
}
"""
