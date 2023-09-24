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
1. Clone the repository:
    ```bash
    git clone https://github.com/realgermosen/Adventist_bot.git
    ```
2. Navigate to the `Adventist_bot` directory:
    ```bash
    cd Adventist_bot
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Setup your `.env` file with your OpenAI API key:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```
5. If you're using a custom voice from Eleven Labs, update the voice ID in the `.env` file:
    ```
    ELEVEN_LABS_API_KEY=your_eleven_labs_api_key_here
    ELEVEN_LABS_VOICE_ID=your_voice_id_here
    ```

## Usage
To run the bot, execute:
```bash
python app.py
```

## Contributing
Feel free to fork this repository and contribute! Pull requests are welcome.

## License
This project is open-source, under the MIT License.

