# Adventist Bot: A Voice-Enabled SDA Chatbot

## Overview
Adventist Bot is a voice-enabled chatbot designed to engage users in meaningful conversations based on biblical teachings. Utilizing a persona named Joshua, the bot draws inspiration from biblical characters and wisdom to guide users on their spiritual journey.

## Features
- Voice-enabled interactions
- Real-time audio processing
- Uses OpenAI's GPT-3 for generating conversational responses
- Text-to-Speech functionality using Eleven Labs' custom voice

## Dependencies
To install all required packages, run:
```bash
pip install -r requirements.txt
```

## Setup

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/realgermosen/Adventist_bot.git
    ```

2. **Navigate to the Directory:**
    ```bash
    cd Adventist_bot
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Create `.env` File:**
    Create a new file in the root directory and name it `.env`. This file will store your API keys.
    ```bash
    touch .env
    ```

    - **OpenAI API Key:**
      You can find your Secret API key in your [OpenAI User Settings](https://beta.openai.com/account/api-keys).
      ```bash
      echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
      ```

    - **Eleven Labs API Key:**
      Follow the steps explained in the [Eleven Labs Authentication Docs](https://docs.elevenlabs.io/api-reference/quick-start/authentication).
      ```bash
      echo "ELEVEN_LABS_API_KEY=your_eleven_labs_api_key_here" >> .env
      ```

    - **Eleven Labs Voice ID:**
      You can either select a voice ID from [Eleven Labs Voices](https://api.elevenlabs.io/v1/voices) or use the pre-made voice called Michael.
      ```bash
      echo "ELEVEN_LABS_VOICE_ID=flq6f7yk4E4fJM5XTYuZ" >> .env  # For Michael's voice
      ```

    > Note: Make sure your `.env` file is actually named `.env` and not `.env.txt`.

5. **Run the Application:**
    ```bash
    python app.py
    ```

That's it! You should now have everything set up.


## Usage
To run the bot, execute:
```bash
python app.py
```

## Contributing
Feel free to fork this repository and contribute! Pull requests are welcome.

## License
This project is open-source, under the MIT License.

