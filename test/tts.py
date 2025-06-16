from RealtimeTTS import TextToAudioStream, CoquiEngine


class Voice:
    def __init__(self):
        # Initializing the tts engine with Language, Sampled voice, Path of Model
        self.engine = CoquiEngine(
            language="en",                             
            voice="./ATLAS/TTS/voice_sample/audio.wav",
            local_models_path = "./ATLAS/TTS/models"            
        )
        # Starting the Text to Audio Stream
        self.stream = TextToAudioStream(self.engine)
        # Generating a warm up generation for better performance on first Generation from the user
        self.stream.feed("warm up").play(muted=True)


    def TTS(self, text, mute=False):
        # Steaming the generated audio for the inputted text
        self.stream.feed(text).play(
            output_wavfile="./ATLAS/TTS/output.wav",
            muted=mute
        )

    def Disable(self):
        # Shutting down the tts engine
        self.engine.shutdown()




if __name__ == "__main__":

    response = "Certainly, sir. I’ll categorize and arrange your tasks for optimum productivity today. Shall we begin with the top priorities?"
    voice = Voice()

    voice.TTS(text=response)
    voice.Disable()
