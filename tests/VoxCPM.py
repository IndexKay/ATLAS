import soundfile as sf
import numpy as np
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")

# Streaming
chunks = []
for chunk in model.generate_streaming(
    text = "Greetings, esteemed operator. I am ATLAS, your Automated Task Learning and Assistance System. Here to deliver swift solutions, streamline processes, and perhaps even sprinkle a dash of charm along the way. With my multifaceted capabilities, I can tackle anything from decoding convoluted data structures to crafting eloquent prose, ensuring your goals are achieved with precision and flair.",
    prompt_wav_path="tests/voice_sample/voice_short.mp3",      # optional: path to a prompt speech for voice cloning
    prompt_text="""Accessed. Activated. I've noticed we are not connected to a network. 

For me to operate at full capacity, please activate the Wi-Fi settings on your device. Accessing alarm and interface settings. In this window, you can set up your customized greeting and alarm preferences. 

Should you need anything, you need only ask. Remember, I am a voice activated system. Your device is now at full power. 

Though it isn't quite an arc reactor, this power source should suffice. Charging. Your device is running low on power. 

Your device is now running at dangerously low power levels. We are now running on emergency backup power. Adjusting display. 

Accessing bonus features. Calibrating settings and playback. Accessing deleted scenes. 

Playback initiated. Disabling voice commands during playback. Languages.

Pausing playback. Pop-up menu accessed. The settings are now available. 

Setup. Deactivating subtitles. Activating subtitles. 

Top menu accessed. Camera disabled. Camera activated.

Accessing alarm and interface settings. In this window, you can set up your customized greeting and alarm preferences. Accessing chronographic display. 

Still plenty of time to save the day. Please make the most of it. Please indicate a time. 

An AM or PM. Alarm cancelled. Daily it is. 

Alarm set for. Thank you. The alarm is set. 

Enjoy your delay of the inevitable. Evasive maneuvers acknowledged. The world needs your expertise. 

Or at least your presence. We appear to be at an impasse. You wish to sleep, but it is my duty to wake you.
""",          # optional: reference text
    cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
    inference_timesteps=8,   # LocDiT inference timesteps, higher for better result, lower for fast speed
):
    chunks.append(chunk)
wav = np.concatenate(chunks)

sf.write("tests/tts_output/output.wav", wav, model.tts_model.sample_rate)
print(f"saved: output_streaming.wav \n chunk length: {len(chunks)}")