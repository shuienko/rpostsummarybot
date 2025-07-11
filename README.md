# 🤖 Reddit Post Summary Bot: Your AI Reddit Companion! 🚀

Ever wished you could get the TL;DR of a Reddit thread without scrolling through endless comments? Say hello to your new best friend! This Telegram bot uses Claude AI to analyze Reddit posts and serve up the juicy details in seconds.

## ✨ What Can This Magical Bot Do?

When you send a Reddit URL, the bot works its magic to deliver:

- 📌 **TLDR: POST IN A NUTSHELL** - A concise summary of what the post is about
- 💬 **WHAT THE CROWD IS SAYING** - The key points from the comments section
- 🎭 **MOOD METER: VIBES CHECK** - Sentiment analysis with percentages of positive/negative reactions
- 😄 **EMOTION DETECTOR** - Identifies the top emotions in the comments
- 🔥 **INTENSITY METER** - Measures how strongly people feel about the topic

## 🧠 AI Brainpower Options

Choose your AI processing power based on your needs:
- ⚡ **Haiku3 Mode** - Quick analysis using Claude 3 Haiku (default)
- 🚀 **Haiku35 Mode** - Latest Haiku model with improved performance
- 🔋 **Sonnet4 Mode** - Premium analysis with Claude Sonnet 4.0

## 🛠️ Cool Technical Features

- 🔄 **Smart Caching** - Results are cached for 24 hours to save API calls
- 📊 **Usage Tracking** - Limits of 30 requests per day with rate limiting
- 🔒 **Private Access** - Bot is restricted to a single authorized user
- 📱 **Message Chunking** - Long analyses are automatically split into readable chunks
- 📝 **Detailed Logging** - Comprehensive logging system with rotation

## 🤓 Prerequisites

- Python 3.11+ (because we're fancy like that)
- Reddit API credentials (for stalking Reddit posts... legally!)
- Telegram Bot Token (your bot's ID card)
- Anthropic API Key (to access the AI brains)
- Docker (optional, for those who like containers)

## 🚀 Quick Start

### 1️⃣ Clone this beauty:
```bash
git clone git@github.com:shuienko/rpostsummarybot.git
cd rpostsummarybot
```

### 2️⃣ Install the goodies:
```bash
pip install -r requirements.txt
```

### 3️⃣ Create a `.env` file with your secret stuff:
```env
TELEGRAM_TOKEN=your_telegram_bot_token
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
ANTHROPIC_API_KEY=your_anthropic_api_key
ALLOWED_USER_ID=your_telegram_user_id_here
```

### 4️⃣ Launch the bot:
```bash
python bot.py
```

## 🐳 Docker Fans

Build and run with Docker if you're too cool for local installations:

```bash
docker build -t reddit-analysis-bot .
docker run -v $(pwd)/config:/app/config reddit-analysis-bot
```

## 🎮 Bot Commands

- `/start` - Wake up the bot with a friendly greeting
- `/help` - Show all the cool commands available
- `/model [haiku3/haiku35/sonnet4]` - Switch between AI brains
- `/usage` - Check how many requests you have left today
- `/whoami` - Discover your Telegram user ID
- `/cache` - See what's in the memory bank
- `/clearcache` - Spring cleaning for the cache

## 🧙‍♂️ How It Works Behind The Curtain

1. You send a Reddit URL
2. Bot checks if it's in the cache (why work twice?)
3. If not cached, it fetches the post and top 10 comments
4. Claude AI analyzes the post content and creates a summary
5. Each comment gets a sentiment analysis (positive/negative/neutral)
6. Comments are summarized together to extract key points
7. All the data is packaged into a fun, emoji-filled response
8. Long responses are automatically split into digestible chunks

## 📊 Rate Limits & Usage

- Maximum 30 requests per day (we don't want to break the bank)
- Minimum 30 seconds between requests (patience is a virtue)
- Results are cached for 24 hours (for efficiency!)
- Cache holds up to 100 results (we're not made of memory)

## 🛡️ Error Handling

The bot gracefully handles:
- Invalid Reddit URLs (it's not a magician)
- API failures (when Reddit or Claude are having a bad day)
- Rate limiting (when you're too excited and sending too many requests)
- Message size limitations (by splitting long analyses)

## 🙏 Acknowledgments

- Built with [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- Reddit data via [asyncpraw](https://asyncpraw.readthedocs.io/)
- AI magic powered by [Anthropic's Claude](https://www.anthropic.com/claude)

## 🎭 Why This Bot Is Awesome

- It saves you time reading long Reddit threads
- It gives you the emotional temperature of discussions
- It's private and only works for you
- It has a personality (unlike some other bots)
- It uses emojis liberally because life's too short for plain text

Now go forth and analyze some Reddit posts! 🚀
