// No import statement at the top!

class PipecatClient {
    constructor() {
        this.room = null;
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

            // Request room URL and token from the bot
            const response = await fetch('http://127.0.0.1:8080/start', {
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
            console.log('Response from /start:', data);

            const roomUrl = data.room_url || data.url;
            const token = data.token;
            if (!roomUrl || !token) {
                throw new Error('No room URL or token received from bot');
            }

            console.log('Joining LiveKit room with URL:', roomUrl);

            // Use livekit from the CDN (all lowercase)
            this.room = new window.livekit.Room();
            const RoomEvent = window.livekit.RoomEvent;

            this.room
                .on(RoomEvent.Connected, () => {
                    console.log('Connected to LiveKit room');
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
                .on(RoomEvent.Disconnected, () => {
                    console.log('Disconnected from LiveKit room');
                    this.handleDisconnect();
                })
                .on(RoomEvent.ParticipantConnected, (participant) => {
                    console.log('Participant joined:', participant.identity);
                })
                .on(RoomEvent.ParticipantDisconnected, (participant) => {
                    console.log('Participant left:', participant.identity);
                })
                .on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
                    console.log('Track started:', track.kind);
                })
                .on(RoomEvent.ConnectionError, (error) => {
                    console.error('LiveKit connection error:', error);
                    this.showError(`Connection error: ${error.message || 'Unknown error'}`);
                    this.handleDisconnect();
                });

            // Connect with audio only
            await this.room.connect(roomUrl, token, {
                autoSubscribe: true,
                audio: true,
                video: false,
            });

        } catch (error) {
            console.error('Connection error:', error);
            this.showError(error.message);
            this.handleDisconnect();
        }
    }

    async disconnect() {
        if (this.room) {
            try {
                await this.room.disconnect();
            } catch (error) {
                console.error('Error during disconnect:', error);
            }
        }
        this.handleDisconnect();
    }

    handleDisconnect() {
        this.isConnected = false;
        this.room = null;
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
        if (!this.room || !this.isConnected) return;

        this.isMuted = !this.isMuted;
        this.room.localParticipant.setMicrophoneEnabled(!this.isMuted);

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
