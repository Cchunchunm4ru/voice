/**
 * Pipecat Voice AI Bot Frontend
 * Connects to the Pipecat bot using Daily.co WebRTC transport
 */

class PipecatClient {
    constructor() {
        this.callFrame = null;
        this.isMuted = false;
        this.isConnected = false;
        
        // UI Elements
        this.statusEl = document.getElementById('status');
        this.connectBtn = document.getElementById('connectBtn');
        this.disconnectBtn = document.getElementById('disconnectBtn');
        this.muteBtn = document.getElementById('muteBtn');
        this.audioBars = document.getElementById('audioBars');
        this.idleText = document.getElementById('idleText');
        this.errorMessage = document.getElementById('errorMessage');
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        this.connectBtn.addEventListener('click', () => this.connect());
        this.disconnectBtn.addEventListener('click', () => this.disconnect());
        this.muteBtn.addEventListener('click', () => this.toggleMute());
    }
    
    updateStatus(status, className) {
        this.statusEl.textContent = status;
        this.statusEl.className = `status ${className}`;
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.classList.remove('hidden');
        setTimeout(() => {
            this.errorMessage.classList.add('hidden');
        }, 5000);
    }
    
    async connect() {
        try {
            this.updateStatus('Connecting...', 'connecting');
            this.connectBtn.disabled = true;
            
            // Request room URL from the bot
            const response = await fetch('http://localhost:7860/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    config: []
                })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to start bot session: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (!data.room_url) {
                throw new Error('No room URL received from bot');
            }
            
            // Create Daily call frame
            this.callFrame = window.DailyIframe.createCallObject({
                audioSource: true,
                videoSource: false,
            });
            
            // Set up call frame event listeners
            this.callFrame
                .on('joined-meeting', () => {
                    console.log('Joined meeting');
                    this.isConnected = true;
                    this.updateStatus('Connected - Speak now!', 'connected');
                    this.connectBtn.classList.add('hidden');
                    this.disconnectBtn.classList.remove('hidden');
                    this.disconnectBtn.disabled = false;
                    this.muteBtn.classList.remove('hidden');
                    this.muteBtn.disabled = false;
                    this.idleText.classList.add('hidden');
                    this.audioBars.classList.remove('hidden');
                })
                .on('left-meeting', () => {
                    console.log('Left meeting');
                    this.handleDisconnect();
                })
                .on('error', (error) => {
                    console.error('Daily error:', error);
                    this.showError(`Connection error: ${error.errorMsg || 'Unknown error'}`);
                    this.handleDisconnect();
                })
                .on('participant-joined', (event) => {
                    console.log('Participant joined:', event.participant);
                })
                .on('participant-left', (event) => {
                    console.log('Participant left:', event.participant);
                })
                .on('track-started', (event) => {
                    console.log('Track started:', event.track.kind);
                });
            
            // Join the room
            await this.callFrame.join({ url: data.room_url });
            
        } catch (error) {
            console.error('Connection error:', error);
            this.showError(error.message);
            this.handleDisconnect();
        }
    }
    
    async disconnect() {
        if (this.callFrame) {
            try {
                await this.callFrame.leave();
                await this.callFrame.destroy();
            } catch (error) {
                console.error('Error during disconnect:', error);
            }
        }
        this.handleDisconnect();
    }
    
    handleDisconnect() {
        this.isConnected = false;
        this.callFrame = null;
        this.updateStatus('Disconnected', 'disconnected');
        this.connectBtn.classList.remove('hidden');
        this.connectBtn.disabled = false;
        this.disconnectBtn.classList.add('hidden');
        this.muteBtn.classList.add('hidden');
        this.audioBars.classList.add('hidden');
        this.idleText.classList.remove('hidden');
        
        if (this.isMuted) {
            this.isMuted = false;
            this.muteBtn.textContent = 'Mute Microphone';
            this.muteBtn.className = 'btn-mute';
        }
    }
    
    toggleMute() {
        if (!this.callFrame || !this.isConnected) return;
        
        this.isMuted = !this.isMuted;
        this.callFrame.setLocalAudio(!this.isMuted);
        
        if (this.isMuted) {
            this.muteBtn.textContent = 'Unmute Microphone';
            this.muteBtn.className = 'btn-unmute';
            this.updateStatus('Connected - Microphone Muted', 'connected');
        } else {
            this.muteBtn.textContent = 'Mute Microphone';
            this.muteBtn.className = 'btn-mute';
            this.updateStatus('Connected - Speak now!', 'connected');
        }
    }
}

// Initialize the client when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const client = new PipecatClient();
    console.log('Pipecat client initialized');
});
