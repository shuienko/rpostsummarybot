# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Discord bot that analyzes Reddit posts and comments using Anthropic's Claude AI. The bot extracts Reddit content, performs sentiment analysis, and generates summaries with emotional insights. It's designed for private Discord servers with rate limiting and caching.

## Development Commands

### Running the Bot
```bash
python bot.py
```

### Dependencies
```bash
pip install -r requirements.txt
```

### Docker Commands
```bash
# Build the Docker image
docker build -t reddit-analysis-bot .

# Run with volume mount for config
docker run -v $(pwd)/config:/app/config reddit-analysis-bot
```

## Architecture

### Core Components

- **RedditAnalyzer** (`bot.py:179-293`): Main AI analysis engine that handles Claude API calls
  - Supports multiple Claude models (haiku3, haiku35, sonnet4)
  - Handles post summarization and comment sentiment analysis
  - Returns structured JSON responses for sentiment data

- **UsageTracker** (`bot.py:54-101`): Rate limiting and quota management
  - 30 requests per day limit per user
  - 30-second rate limiting between requests
  - Tracks daily usage and request history

- **ResultCache** (`bot.py:107-161`): In-memory caching system
  - 24-hour TTL for cached results
  - LRU eviction when max size (100 entries) is reached
  - MD5-based cache keys from Reddit URLs

### Bot Features

- **Private server access**: Works in any Discord server the bot is added to
- **Multi-model support**: Switch between Claude models via `!model` command
- **Message chunking**: Automatically splits long responses to fit Discord's 2000 character limit
- **Comprehensive logging**: Rotating file logs with both file and console output

### Environment Configuration

Required `.env` variables:
- `DISCORD_TOKEN`: Discord bot API token
- `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`: Reddit API credentials
- `ANTHROPIC_API_KEY`: Claude AI API key

### Reddit Data Flow

1. URL validation and post extraction via asyncpraw
2. Fetches post content and top 10 comments
3. Cache check using MD5 hash of URL
4. If not cached: Claude analysis of post and individual comment sentiment
5. Aggregated response with TLDR, sentiment percentages, emotions, and intensity
6. Result caching and response chunking for Discord

### Bot Commands

- `/model`: Switch Claude model (with dropdown selection)
- `/usage`: Check daily quota and remaining requests
- `/cache` and `/clearcache`: Cache management
- `/whoami`: Get Discord user ID
- `/start`: Welcome message and bot introduction
- `/help`: Show all available commands

## Key Technical Details

- **Sentiment Analysis**: Each comment gets individual analysis with emotion and intensity scoring
- **Token Tracking**: All Claude API calls log input/output token usage
- **Error Handling**: Graceful fallbacks for API failures and malformed responses
- **Discord Integration**: Uses discord.py library with modern slash commands and automatic command syncing