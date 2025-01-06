# Reddit Post Summary Telegram Bot

A Telegram bot that analyzes Reddit posts using Claude AI to provide summaries and sentiment analysis. The bot processes Reddit URLs sent by users and returns:
- A concise summary of the post content
- A summary of the top comments
- Sentiment analysis of the comments (positive vs. negative percentages)

## Features

- Post analysis using Claude AI (Anthropic's claude-3-haiku model)
- Top comments summarization
- Sentiment analysis of comments
- Automatic message chunking for long responses
- Docker support for easy deployment
- Environment variable configuration
- Token usage logging

## Prerequisites

- Python 3.11 or higher
- Reddit API credentials (client ID and client secret)
- Telegram Bot Token
- Anthropic API Key
- Docker (optional)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd reddit-analysis-telegram-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your credentials:
```env
TELEGRAM_TOKEN=your_telegram_bot_token
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

### Running Locally

1. Start the bot:
```bash
python bot.py
```

2. Open Telegram and start a chat with your bot
3. Send the bot a Reddit post URL
4. Wait for the analysis results

### Running with Docker

1. Build the Docker image:
```bash
docker build -t reddit-analysis-bot .
```

2. Run the container:
```bash
docker run -v $(pwd)/config:/app/config reddit-analysis-bot
```

Note: Make sure to place your `.env` file in a `config` directory before running the container.

## Bot Commands

- `/start` - Displays welcome message and usage instructions
- Send any Reddit post URL to get analysis

## API Usage and Limits

The bot uses several APIs:

- Reddit API (via asyncpraw)
- Anthropic's Claude API (claude-3-haiku model)
- Telegram Bot API

Be mindful of the following:
- The bot processes only the top 10 comments from each post
- Claude API has a max token limit of 1024 tokens per response
- Messages longer than Telegram's limit are automatically split into multiple messages

## Error Handling

The bot includes error handling for:
- Invalid Reddit URLs
- Missing environment variables
- API call failures
- Message size limitations

## Development

The project structure:
```
├── bot.py              # Main bot code
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
└── README.md          # Documentation
```

Key classes:
- `RedditAnalyzer`: Handles interactions with Claude AI
- `RedditBot`: Manages Reddit post processing and analysis
- Helper functions for message chunking and command handling

## Acknowledgments

- Built with [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- Uses [asyncpraw](https://asyncpraw.readthedocs.io/) for Reddit API access
- Powered by [Anthropic's Claude AI](https://www.anthropic.com/claude)