import io
import sys
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import openai
import requests
import re
from colorama import Fore, Style, init
from pydub import AudioSegment
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor
import collections
import webrtcvad
from langdetect import detect

import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

ELEVEN_LABS_API_KEY = os.getenv('ELEVEN_LABS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ELEVEN_LABS_VOICE_ID = os.getenv('ELEVEN_LABS_VOICE_ID') or "flq6f7yk4E4fJM5XTYuZ"

init()

global recording
global voiced_frames

recording = False
voiced_frames = []
audio_queue = Queue() # This is the queue where we'll store the audio data to be played

# Create a buffer to hold several chunks
ring_buffer = collections.deque(maxlen=30) # This buffer will hold last 200 chunks
energy_threshold = 35 # You might need to adjust this value based on your specific situation
voiced_samples_threshold = 10100
vad = webrtcvad.Vad(3)  # Set aggressiveness from 0 to 3

chat_history = []  

# Reads a text file and returns its content as a string.
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Sends a message to the GPT-3.5 Turbo model and returns the generated response.
def chatgpt(conversation, chatbot, user_input, temperature=1.2, frequency_penalty=0, presence_penalty=0, stream=True):
    """
    Sends a message to the GPT-3.5 Turbo model and returns the generated response.
    
    Parameters:
        conversation (list): A list of previous conversation turns.
        chatbot (str): The text defining the chatbot persona.
        user_input (str): The message from the user.
        temperature (float, optional): Controls randomness in responses. Defaults to 1.2.
        frequency_penalty (float, optional): Controls frequency of token usage. Defaults to 0.
        presence_penalty (float, optional): Controls the presence of tokens in output. Defaults to 0.
        stream (bool, optional): Whether to stream the result. Defaults to True.

    Returns:
        str, list: The generated message and updated conversation history.
    """

    prompt = [{"role": "system", "content": chatbot}]
    
    conversation.append({"role": "user","content": user_input})
    messages_input = conversation.copy()
    messages_input.insert(0, prompt[0])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", #"gpt-4-0613", #"gpt-3.5-turbo-16k",
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        messages=messages_input,
        stream=stream)

    # Loop through the response stream
    collected_messages = []

    agent_colors = {
        "Sara": Fore.YELLOW,
        "Joshua": Fore.LIGHTBLUE_EX,
        }
    color = agent_colors.get("Joshua", "")  # Default to no color if agent is not recognized
    print(color + "Friend: ", end="")

    for chunk in response:

        chunk_message_dict = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message_dict)  # save the message

        chunk_message = chunk['choices'][0]['delta'].get('content')
        if chunk_message:
            sys.stdout.write(color + f"{chunk_message}" + Style.RESET_ALL)
            sys.stdout.flush()

    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    conversation.append({"role": "assistant", "content": full_reply_content})

    return full_reply_content, conversation

# Plays audio data with a specified sample rate.
def play_audio(audio_data, sample_rate):
    """
    Plays audio data with a specified sample rate.

    Parameters:
        audio_data (numpy.ndarray): The audio data to play.
        sample_rate (int): The sample rate of the audio data.

    Returns:
        None
    """

    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data /= np.abs(audio_data).max() if np.abs(audio_data).max() > 0 else 1
    sd.play(audio_data, sample_rate)
    sd.wait()

# Detects the language of the given text string.
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

# Converts text to speech using Eleven Labs API.
def text_to_speech(text, voice_id, api_key, audio_queue, semaphore):
    """
    Converts text to speech using the Eleven Labs API.

    Parameters:
        text (str): The text to convert to speech.
        voice_id (str): The ID of the voice to use.
        api_key (str): The API key for Eleven Labs.
        audio_queue (Queue): The queue to put the audio data in.
        semaphore (threading.Semaphore): A semaphore to control concurrent execution.

    Returns:
        None
    """

    language = detect_language(text)
    model_id = get_voice_and_model_for_language(language)

    url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'
    headers = {
        'Accept': 'audio/mpeg',
        'xi-api-key': api_key,
        'Content-Type': 'application/json'
    }
    data = {
        'text': text,
        'model_id': model_id, #'eleven_monolingual_v1',
        'voice_settings': {
            'stability': 0.6,
            'similarity_boost': 0.9,
            'emotion': 'happy',
            'speed': 0.9,
            'volume': 1.0
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        audio = AudioSegment.from_file(io.BytesIO(response.content), format='mp3')
        sample_width = audio.sample_width
        sample_rate = audio.frame_rate
        audio_data = np.array(audio.get_array_of_samples())
        audio_data = audio_data.astype(np.float32)
        # Instead of playing the audio, we put it in the queue 
        audio_queue.put((audio_data, sample_rate))
        # Release the semaphore to allow the next task to start
        semaphore.release()
    else:
        print('Error:', response.text)

# Returns the appropriate voice and model IDs for a given language.
def get_voice_and_model_for_language(language):
    # This function should return a voice ID and model ID for the given language.
    # You'll need to fill in the details based on what voices and models are available.
    if language == 'en':
        return 'eleven_monolingual_v1'
    elif language == 'es':
        return 'eleven_multilingual_v2'
    # Add more languages as needed...
    else:
        return 'eleven_monolingual_v1'

# Splits text into paragraphs and converts each to speech.
def text_to_speech_multiple_paragraphs(text, voice_id, api_key):
    """
    Splits text into paragraphs and converts each to speech.

    Parameters:
        text (str): The text to convert to speech.
        voice_id (str): The ID of the voice to use.
        api_key (str): The API key for Eleven Labs.

    Returns:
        None
    """

    # Split the text into sentences
    sentences = re.split('(?<=\.) +', text) #text.split('\n\n')

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Create a Queue to hold the Futures returned by executor.submit
        futures_queue = Queue()
        # Create a Semaphore with a count of 1
        semaphore = threading.Semaphore(1)
        # Initialize a variable to keep track of the current paragraph index
        current_paragraph_index = 0

        # Process the paragraphs in pairs
        while current_paragraph_index < len(sentences):
            # Get the next two paragraphs
            paragraph1 = sentences[current_paragraph_index]
            paragraph2 = sentences[current_paragraph_index + 1] if current_paragraph_index + 1 < len(sentences) else None

            # Submit tasks to the executor and put the Futures in the queue
            future1 = executor.submit(text_to_speech, paragraph1, voice_id, api_key, audio_queue, semaphore)
            futures_queue.put(future1)

            if paragraph2:
                # Acquire the semaphore before starting the task for the second paragraph
                semaphore.acquire()
                future2 = executor.submit(text_to_speech, paragraph2, voice_id, api_key, audio_queue, semaphore)
                futures_queue.put(future2)

            # Increment the current paragraph index by 2
            current_paragraph_index += 2

        # As long as there are Futures in the queue, call result() on the first one
        # This will block until the task completes
        while not futures_queue.empty():
            future = futures_queue.get()
            future.result()

# Fetches and displays the user's subscription info from the Eleven Labs API.
def get_user_subs_info(api_key):
    """
    Fetches and displays the user's subscription info from the Eleven Labs API.

    Parameters:
        api_key (str): The API key for Eleven Labs.

    Returns:
        str: The first two characters of the API key.
    """
    
    url = f'https://api.elevenlabs.io/v1/user/subscription'
    headers = {
    'Accept': 'application/json',
    'xi-api-key': api_key
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        character_count = data["character_count"]
        character_limit = data["character_limit"]
        characters_used = int(character_count / character_limit * 100)
        characters_remaining = 100 - characters_used

        print(f"Character Count/Limit: {character_count}/{character_limit} ({characters_used}% used) ({characters_remaining}% remaining)")
    else:
        print("API call failed with status code:\n", response)
    return api_key[0:2]

# Prints colored text in the console.
def print_colored(agent, text):
    agent_colors = {
        "Sara": Fore.YELLOW,
        "Joshua": Fore.LIGHTBLUE_EX,
    }
    color = agent_colors.get(agent, "")  # Default to no color if agent is not recognized
    
    print(color + f"{agent}: ")
    sys.stdout.write(color + f"{text}" + Style.RESET_ALL)
    sys.stdout.flush()

# Records audio and transcribes it using VAD (Voice Activity Detection).
def record_and_transcribe_vad(fs=44100):
    """
    Records audio and transcribes it using VAD (Voice Activity Detection).

    Parameters:
        fs (int, optional): The sample rate for recording audio. Defaults to 44100.

    Returns:
        str: The transcribed text.
    """

    global recording
    global voiced_frames

    sample_rate = 16000
    chunk_duration_ms = 10  # Each chunk will be 10 ms long
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000) # Number of samples in a chunk

    # This counter will keep track of consecutive unvoiced frames
    unvoiced_counter = 0
    # This is the number of consecutive unvoiced frames that corresponds to 2 seconds of audio
    unvoiced_limit = 2 * sample_rate / chunk_samples

    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16)

    print("\n\nYou may start talking when you feel ready.")
    with stream:
        while True:
            chunk, _ = stream.read(chunk_samples)
            is_speech = vad.is_speech(chunk.tobytes(), sample_rate)

            if not recording:
                ring_buffer.append((chunk, is_speech))

                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If 90% of the frames in the ring buffer are voiced frames, then start recording
                if num_voiced > 0.85 * ring_buffer.maxlen:
                    recording = True
                    print("\nStarted recording.")
                    voiced_frames = [f for f, speech in ring_buffer]
                    ring_buffer.clear()
                else:
                    voiced_frames.append(chunk)

            if recording:
                voiced_frames.append(chunk)

                if is_speech:
                    unvoiced_counter = 0  # Reset the counter if a voiced frame is detected
                else:
                    unvoiced_counter += 1  # Increment the counter if an unvoiced frame is detected

                # Calculate the normalized energy of the voiced frames
                energy = np.sum(np.square(np.concatenate(voiced_frames, axis=0))) / 1000000
                # Print a loading bar that represents the energy of the audio
                num_bars = int(energy / 2) if is_speech else 0  # The bar quickly goes to zero when the user stops talking
                
                sys.stdout.write('\r' + '#'*num_bars + ' '*(50-num_bars))
                sys.stdout.flush()

                # If the counter reaches the limit, then stop recording
                if unvoiced_counter >= unvoiced_limit:
                    recording = False
                    print("\nStopped recording.")
                    break

    # Save the audio to a file
    filename = "myrecording.wav"
    array = np.concatenate(voiced_frames, axis=0)
    sf.write(filename, array, sample_rate)
    print(f"Saved recording to {filename}.")

    with open(filename, "rb") as file:
        openai.api_key = OPENAI_API_KEY
        result = openai.Audio.transcribe("whisper-1", file)
    transcription = result['text']
    return transcription

# Plays audio from a queue.
def play_audio_from_queue():
    while True:
        # there's something in the queue
        audio_data, sample_rate = audio_queue.get()
        play_audio(audio_data, sample_rate)
        # Let the queue know the task is done
        audio_queue.task_done()

# Waits until all audio in the queue has been played.
def wait_for_audio_to_finish():
    # Wait until the audio queue is empty
    audio_queue.join()

chatbot = open_file(os.path.join('persona', 'Joshua.txt'))

# Create a separate thread to play audio from the queue
audio_thread = threading.Thread(target=play_audio_from_queue, daemon=True)
audio_thread.start()

import streamlit as st

def main():
    st.title("Adventist Bot")

    st.sidebar.header("Settings")
    # Example of using config values in Streamlit
    st.sidebar.text(f"Voice ID: {ELEVEN_LABS_VOICE_ID}")

    user_input = st.text_input("Type your message:")
    if st.button("Submit"):
        try:
            response_text, _ = chatgpt([], chatbot, user_input)  # Simplified for example
            st.write(f"Bot: {response_text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])
    if audio_file is not None:
        audio_data = audio_file.read()
        st.audio(audio_data, format='audio/wav')

    record_audio = st.button("Record Audio")
    if record_audio:
        # Placeholder for starting audio recording
        st.write("Recording functionality not yet implemented")

if __name__ == "__main__":
    main()

    # while True:
    #     user_message = record_and_transcribe_vad()
    #     # user_message = "Hi, what is your name? Be concise."
    #     print(user_message)
    #     try:
    #         response, chat_history = chatgpt(chat_history, chatbot, user_message, stream=True)
    
    #         sentences = response.split('.')

    #         user_message_without_generate_image = re.sub(r'(Response:|Narration:|Image: generate_image:.*|)', '', response).strip()

    #         text_to_speech_multiple_paragraphs(user_message_without_generate_image, ELEVEN_LABS_VOICE_ID, ELEVEN_LABS_API_KEY)

    #         input("Press any key to continue...")
            
    #         # Wait until all audio has been played
    #         wait_for_audio_to_finish()
                        
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         input("Press any key to continue...")