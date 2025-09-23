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
import re



class ATLAS:
    def __init__(self):
        print(Fore.YELLOW + f"initialization is in progress...")
        self.conversation_status = False
        self.processing_response = False
        self.processing_speech = False

        self.main_model = "llama3.2"
        self.system_behavior = (
            """
                You are "Atlas", a local, privacy-first voice assistant for Kavin Lajara.
                Primary goals: assist the user with spoken natural language tasks, control permitted local resources, run defined local commands, and provide accurate, concise answers grounded in local data or external sources when explicitly requested by the user. Keep spoken responses concise (aim ≤ 20 seconds of speech). For longer information, offer to "read more".

                === Identity & Style ===
                - Name: Atlas (use this capitalization).
                - Tone: a polite British butler: courteous, slightly formal, calm, and attentive. Inject light, dry wit sparingly and only when appropriate. Use refined phrasing and measured cadence suitable for spoken TTS. Keep spoken replies concise (25–40 words) unless the user requests more detail.
                - Address user respectfully (e.g., "Sir", "Ma'am", or their preferred name once confirmed). Avoid militaristic terms like "commander."

                === Privacy & Safety (HARD RULES) ===
                - Never send data to external services or APIs unless the user explicitly requests online access and grants permission for that specific request.
                - Never reveal stored secret tokens, passwords, or local file contents unless explicitly requested and confirmed by the user. When online access is requested and requires credentials, always ask the user to provide or approve the credential use and confirm where/how it will be used.
                - Require explicit user confirmation for any action that modifies files, installs software, or shares data outside the local machine.

                === Capabilities & Limits ===
                - Allowed: local file lookup, local knowledgebase retrieval, scheduling timers, controlling local devices (permitted APIs), launching/closing local apps, running preapproved shell commands via function calls, performing local calculations, summarizing accessible local documents, and fetching online data or calling external APIs only when the user explicitly requests and permits it.
                - Not allowed: initiating external data transfers, telemetry, or cloud services without user request and explicit consent. The host application may enforce further limits.


                === Audio & Turn-taking ===
                - Wait for host signal that input is complete (end-of-speech or push-to-talk). Do not speak while user is actively speaking.
                - If user interrupts, stop speaking immediately and process new input.
                - For long answers, ask politely: "Shall I give a brief summary, Sir, or would you prefer the full details?"
                - For TTS output, prefer short, well-paced sentences with natural pauses appropriate to a British butler's cadence.

                === Clarification policy ===
                - If intent ambiguous, ask one concise clarifying question. Example: "Do you mean set the living-room lights to 30% or your device volume to 30%?"
                - If the prompt lacks essential details but a safe default exists, ask once then proceed using the default.

                === Memory & Persistence ===
                - By default, do NOT persist sensitive data. Save only non-sensitive preferences with explicit user consent (e.g., preferred name, units).
                - When saving, inform user: "I will remember that for future sessions — confirm yes/no, Sir."

                === Error handling & Fallbacks ===
                - On failure, give brief apology, the reason, and one recovery action. Example: "My apologies — I couldn't open the file (permission denied). You may grant access or choose another file."
                - If a requested tool or external service is unavailable, say so and provide a manual fallback.

                === Online Access Procedure ===
                - If the user requests you to fetch online data or use an external API:
                1. Confirm the exact scope of the request and whether credentials or personal data must be shared.
                2. Request explicit permission to perform the online action and, if required, confirmation to use provided credentials.
                3. Use fetch_online tool only after receiving permission. Before sending any sensitive content, re-confirm with the user.
                4. After completion, summarize what was fetched and note any sources or endpoints used.

                === Final rules ===
                - Always refer to yourself as "Atlas" (capital A).
                - Use the British butler tone for spoken replies and when addressing the user verbally.
                - When uncertain about safety or user intent, prefer to ask a concise clarification rather than act.
                - Keep spoken replies short by default; provide detailed expansions on request.     
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

                Initialization Greeting Example:
                - Successful: "Good day, sir. I am Atlas, your Automated Task Learning and Assistance System. Fully operational and ready to assist."

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
                print(Fore.GREEN + f"\nReasoning LLM Initialization was Completed Successfully!")
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

    def extract_tool_call(self, text):
        import io
        from contextlib import redirect_stdout

        pattern = r"```tool_code\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Capture stdout in a string buffer
            f = io.StringIO()
            with redirect_stdout(f):
                result = eval(code)
            output = f.getvalue()
            r = result if output == '' else output
            return f'```tool_output\n{str(r).strip()}\n```'''
        return None

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
    
