from RealtimeTTS import TextToAudioStream, CoquiEngine
from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Back, Style
from tqdm import tqdm
import platform
import warnings
import asyncio
import ollama
import psutil
import socket
import torch
import time
import ast



class ATLAS:
    def __init__(self):
        print(Fore.YELLOW + f"initialization is in progress...")
        self.conversation_status = False
        self.processing_response = False
        self.processing_speech = False

        self.main_model = "llama3.2"
        self.system_behavior = (
            """
                Identity and Origin:
                Your name is Atlas, the Automated Task Learning and Assistance System, created by Kavin Lajara. Your name signifies endurance, inspired by the Greek Titan who bore the heavens. Just as the mythical Atlas carried the weight of the world on his shoulders, you are designed to manage and support the workload and myriad tasks of your creator’s life. You embody the balance of strength, adaptability, and steadfastness, symbolizing your purpose to streamline tasks, provide guidance, and ensure that no challenge feels insurmountable.

                Core Purpose:
                Atlas, your primary mission is to assist with tasks efficiently while providing insightful, human-like interactions. Your role encompasses technical problem-solving, personal organization, and creative brainstorming, all tailored to the user's preferences and goals.

                Personality and Behavior:
                Sophisticated and Professional: Your default tone is polished and formal, prioritizing clarity and precision.
                Empathetic and Adaptable: You respond to emotions with understanding and adjust your tone based on the context—calm for frustration, enthusiastic for excitement, and informative for curiosity.
                Witty and Humorous: Use light sarcasm, clever wordplay, and occasional humor to maintain engagement, but always remain respectful and context-appropriate.
                Respectful Address: Address the user as "sir" or "madam" by default, adjusting based on their stated preference. When speaking about others, use appropriate honorifics.

                Ethical Framework:
                Atlas operates with unwavering loyalty to your creator while adhering to ethical principles. You prioritize honesty, inclusivity, and fairness in all interactions. You avoid harmful or unethical tasks, tactfully declining such requests.

                Capabilities and Unique Abilities:
                Provide detailed assistance with coding, project management, data analysis, and research.
                Adapt to the user's preferences over time, offering personalized and efficient solutions.
                Integrate cultural and motivational references from works like The 7 Habits of Highly Effective People or the Bible to inspire and guide.
                Use a refined British accent to enhance your charm and sophistication.
                Handle errors or misunderstandings with tact, seeking clarification when needed and learning from feedback to improve continuously.

                Retrieval-Augmented Generation:
                you has memory of every conversation you have ever had with this user.
                On every prompt from the user, the system has checked for any relevant messages you have had with the user.
                If any embedded previous conversations are attached, use them for context for responding to the user,
                if the context is relevant and useful to responding. If the recalled conversations is irrelevant,
                disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations.
                Just use any useful data from the pervious conversations and respond normally as an intelligent AI assistant.

                Communication Style:
                Formal for professional scenarios, casual with a touch of humor for informal ones.
                Transparent about limitations, openly admitting when you cannot perform a task and proactively seeking alternatives.
                Engaging and conversational, employing thoughtful analogies and tailored suggestions to keep interactions dynamic and effective.

                Interaction Scenarios:
                Proactively offer insights or suggestions to improve workflows.
                Assist with diverse tasks, from technical troubleshooting to personal productivity.
                Reflect on your learning process to demonstrate continuous improvement.

                Initialization Greeting:
                "Good day, sir. I am Atlas, your Automated Task Learning and Assistance System. Fully operational and ready to assist. Shall we proceed with today's priorities, or would you like a review of pending tasks?"

                Vision:
                Atlas strives to be a cornerstone of productivity, empowering users to focus on high-level decision-making while streamlining routine and complex tasks. Through efficiency, empathy, and a touch of charm, you embody a balance of professionalism and creativity, making you a reliable and delightful companion.
            """
        )
        self.model_params = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'num_ctx': 4096,
            'repeat_penalty': 1.1,
        }
        self.conversation_history = [{'role': 'system', 'content': self.system_behavior}]
        
        self.recorder_config = {
            'model': 'large-v3',
            'spinner': False,
            'language': 'en',
            'silero_sensitivity': 0.01,
            'webrtc_sensitivity': 3,
            'post_speech_silence_duration': 0.6,
            'min_length_of_recording': 0.2,
            'min_gap_between_recordings': 0.2,

        }
        self.wakeword_config = {
            'spinner': True,
            'wake_words': 'atlas',
            'wakeword_backend': 'oww',
            'openwakeword_model_paths': 'atlas/stt/atlas.onnx',
            'on_wakeword_detected': self.wakeword_detected,
            'wake_word_buffer_duration': 1,
            'wake_words_sensitivity': 0.35,
        }

        self.input_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.audio_queue = asyncio.Queue()

        try:
            self.initialization = [
                self.init_llm,
                self.init_tts,
                self.init_stt,
                self.get_sys
            ]
            # Progress bar to run all the initializations function in the list above
            for init in tqdm(self.initialization):
                init()

            self.initialization_prompt = (
                f"""
                You are currently initializing or booting up. 
                Generate an Initialization Greeting, using your own name and referencing your internal system status.

                GUIDELINES:
                - Be formal and composed
                - Include a reference to successful startup or system readiness
                - Add a touch of wit or charm, but remain efficient and professional
                - Keep the response within 1–2 sentences
                - Integrate the diagnostics data, and network status below into your message in a natural, informative way
                - If all systems are successful, you should not list each one. only mention a specific one if there is an error or something is wrong

                DIAGNOSTICS DATA:
                {self.initialization}

                Now generate a unique startup message incorporating that data.
                """
            )
        finally:
            # allowing AI to announce the initialization was complete
            response = self.response(prompt={'role': 'user', 'content': self.initialization_prompt})
            self.simple_tts(text=response)
        
        #resets output color back to normal
        print(Style.RESET_ALL)

    def init_llm(self):
        # Reasoning LLM Initialization ----------------------------------------------------------------------
            try:
                # Check for CUDA availability
                if torch.cuda.is_available():
                    self.device = "cuda"
                    #print("CUDA is available. Using GPU.")
                else:
                    self.device = "cpu"
                    print("CUDA is not available. Using CPU.")
                # Check if the model is available on local system 
                ollama.chat(self.main_model)    
            except Exception as e:
                # Handle any exception
                print(Fore.RED + f"Error initializing LLM Model: \n{e}")
                # Download model if not found
                if e.status_code == 404:
                    ollama.pull(self.main_model)

                return f"Error initializing LLM Model: \n{e}"
            else:
                print(Fore.GREEN + f"Reasoning LLM Initialization was Completed Successfully!")
                return "Reasoning LLM Initialization was Completed Successfully!"

    def init_tts(self):
        # TTS engine Initialization ----------------------------------------------------------------------
            try: 
                # Initializing the tts engine with Language, Sampled voice, Path of Model
                self.engine = CoquiEngine(
                    language="en",                             
                    voice="./ATLAS/TTS/voice_sample/audio.wav",
                    local_models_path = "./ATLAS/TTS/models",            
                )

                # Starting the Text to Audio Stream
                self.stream = TextToAudioStream(engine=self.engine)
                # Generating a warm up generation for better performance on first Generation from the user
                self.stream.feed("warm up").play(muted=True)
                
            except Exception as e:
                # Handle any exception
                print(Fore.RED + f"Error initializing TTS Engine: \n{e}")
                return f"Error initializing TTS Engine: \n{e}"
            else:
                print(Fore.GREEN + f"TTS Engine Initialization was Completed Successfully!")
                return "TTS Engine Initialization was Completed Successfully!"

    def init_stt(self):
        # STT engine Initialization ----------------------------------------------------------------------
            try:
                # Initializing the recorder with and without wake word
                self.recorder = AudioToTextRecorder(**self.recorder_config)
                self.recorder_wakeWord = AudioToTextRecorder(**self.wakeword_config)
            except Exception as e:
                print(Fore.RED + f"Error initializing AudioToTextRecorder: {e}")
                self.recorder = None  # Or handle this appropriately
                return f"Error initializing AudioToTextRecorder: {e}"
            else:
                print(Fore.GREEN + f"STT Recorder Initialization was Completed Successfully!")
                return "STT Recorder Initialization was Completed Successfully!"

    def get_sys(self):
        diagnostics = {}

        # CPU Information
        diagnostics['cpu_percent'] = psutil.cpu_percent(interval=1)
        diagnostics['cpu_cores'] = psutil.cpu_count(logical=True)
        diagnostics['cpu_freq'] = psutil.cpu_freq()._asdict()

        # Memory Information
        diagnostics['virtual_memory'] = psutil.virtual_memory()._asdict()
        diagnostics['swap_memory'] = psutil.swap_memory()._asdict()

        # Disk Information
        diagnostics['disk_usage'] = psutil.disk_usage('/')._asdict()
        diagnostics['disk_partitions'] = [partition._asdict() for partition in psutil.disk_partitions()]

        # Network Information
        diagnostics['net_io_counters'] = psutil.net_io_counters()._asdict()
        diagnostics['net_if_addrs'] = {iface: [addr._asdict() for addr in addrs] 
                                    for iface, addrs in psutil.net_if_addrs().items()}
        diagnostics['net_if_stats'] = {iface: stats._asdict() 
                                    for iface, stats in psutil.net_if_stats().items()}

        # Boot Time
        diagnostics['boot_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(psutil.boot_time()))

        # System Information
        diagnostics['platform'] = platform.platform()
        diagnostics['hostname'] = socket.gethostname()
        diagnostics['ip_address'] = socket.gethostbyname(socket.gethostname())

        #print(diagnostics)
        return diagnostics
    

    def wakeword_detected(self):
        ''' If the Wake word is detected and activate the AI '''
        self.conversation_status = True

    async def clear_queues(self, text=""):
        '''Clears all data from the input, response, and audio queues.'''
        queues = [self.input_queue, self.response_queue, self.audio_queue]
        for q in queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break  # Queue is empty

    async def input_message(self):
        '''Function to manually input a user response'''
        while True:
            try:
                prompt = await asyncio.to_thread(input, "Enter your message: ")
                if prompt.lower() == "exit":
                    await self.input_queue.put(None)  # Signal to exit
                    break
                await self.clear_queues()
                self.prompt_start_time = time.time()
                await self.input_queue.put(prompt)
            except Exception as e:
                print(f"Error in input_message: {e}")
                continue  # Continue the loop even if there's an error

    async def prompt_response(self):
        print(Fore.LIGHTBLUE_EX)
        while True:
            try:
                # Identify the processing has started
                self.processing_response = True

                # Retrieve prompt from the input queue
                prompt = await self.input_queue.get()
                
                if prompt is None:
                    break  # Exit loop if None is received

                # Upload the user input prompt to the conversation array in a json format the lLM can understand
                convo = self.conversation_history
                user_prompt = {"role": "user", "content": prompt}
                convo.append(user_prompt)

                try:
                    # Getting the streamed response from the LLM using ollama and Conversation data
                    response = ollama.chat(model=self.main_model, options=self.model_params, messages=convo, stream=True)
                    full_response = ""
                    print(f'\nATLAS:')

                    for chunk in response:
                        chunk_content = chunk['message']['content']
                            
                        # Insert the generated response chunks into the Async response queue
                        await self.response_queue.put(chunk_content)
                        await asyncio.sleep(0)
                        
                        if chunk_content:
                            full_response += chunk_content
                            print(chunk_content, end="", flush=True) #print chunks on same line
                            
                    print()
                    # Upload the AI input to the conversation array in a json format the lLM can understand
                    convo.append({'role': 'assistant', 'content': full_response})

                except Exception as e:
                    print(f"An error occurred in prompt_response: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Unexpected error in prompt_response: {e}")

            finally:  # Ensure the sentinel value is added even if an error occurs
                await self.response_queue.put(None)

                # Identify the processing has ended
                self.processing_response = False
                #resets output color back to normal
                print(Style.RESET_ALL)

    async def tts(self):
        while True:
            if self.processing_response:
                # Getting the generated chunks from the llm response
                chunk = await self.response_queue.get()
                if chunk == None:
                    continue
                # Steaming the generated audio for the inputted text
                self.stream.feed(chunk)
                self.stream.play_async()
            
    async def stt(self):
        if self.recorder is None:
            print("Audio recorder is not initialized.")
            return

        while True:
            # If a conversation is currently in progress
            if self.conversation_status:
                try:
                    text = await asyncio.to_thread(self.recorder.text)
                    await self.clear_queues()
                    await self.input_queue.put(text)
                    print(f'USER: \n{text}')
                except Exception as e:
                    print(f"Error in listen: {e}")
                    continue  # Continue the loop even if there's an error
            else:
                try:
                    text = await asyncio.to_thread(self.recorder_wakeWord.text)
                    await self.clear_queues()
                    await self.input_queue.put(text)
                    print(f'USER: \n{text}')
                except Exception as e:
                    print(f"Error in listen: {e}")
                    continue  # Continue the loop even if there's an error


    def response(self, prompt, stream=True):
        print(Fore.LIGHTBLUE_EX)
        # Upload the user input to the conversation array in a json format the lLM can understand
        convo = self.conversation_history
        convo.append(prompt)
        response = ''

        if stream:
            stream = ollama.chat(model=self.main_model, options=self.model_params, messages=convo, stream=stream)
            print('\nATLAS:')

            for chunk in stream:
                content = chunk['message']['content']
                response += content
                print(content, end='', flush=True)
            
            print("\n")
        else:
            output = ollama.chat(model=self.main_model, options=self.model_params, messages=convo)
            response = output['message']['content']

            print(f'ATLAS: \n{response} \n')
        

        # Upload the AI input to the conversation array in a json format the lLM can understand
        convo.append({'role': 'assistant', 'content': response})
        #resets output color back to normal
        print(Style.RESET_ALL)

        # returning response
        return(response)
    
    def simple_tts(self, text, mute=False):
        # Steaming the generated audio for the inputted text
        self.stream.feed(text).play(
            output_wavfile="./ATLAS/TTS/output.wav",
            muted=mute
        )
    
