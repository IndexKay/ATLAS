from RealtimeSTT import AudioToTextRecorder

class STT: 
    def __init__(self):
        self.recorder_config = {
            'model': 'large-v3',
            'spinner': False,
            'language': 'en',
            'silero_sensitivity': 0.01,
            'webrtc_sensitivity': 3,
            'post_speech_silence_duration': 0.1,
            'min_length_of_recording': 0.2,
            'min_gap_between_recordings': 0,

            #'realtime_model_type': 'tiny.en',
            #'enable_realtime_transcription': True,
            #'on_realtime_transcription_update': self.clear_queues,
        }
        try:
            self.recorder = AudioToTextRecorder(**self.recorder_config)
        except Exception as e:
            print(f"Error initializing AudioToTextRecorder: {e}")
            self.recorder = None  # Or handle this appropriately


    def record(self):
        self.recorder.start()
        input("Press Enter to stop recording...")
        self.recorder.stop()
        print("Transcription: ", self.recorder.text())

        return(self.recorder.text())



if __name__ == "__main__":
    stt = STT() 
    stt.record()
