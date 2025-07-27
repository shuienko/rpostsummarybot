# Reddit Post Analyzer Bot

A Discord bot that provides AI-powered summaries and analysis of Reddit posts. Paste any Reddit post URL and get a quick, digestible summary of the post content and comments, complete with sentiment analysis and mood tracking.

## Features

- **Post Summarization**: Get a concise TL;DR of any Reddit post.
- **Comment Analysis**: Summarizes the top comments to capture the crowd's opinion.
- **Sentiment Analysis**: Breaks down comment sentiment into positive, negative, and neutral percentages.
- **Emotion Detection**: Identifies the top emotions expressed in the comments (e.g., happy, angry, curious).
- **Intensity Meter**: A visual gauge of the comment section's overall intensity.
- **AI Model Selection**: Switch between different Anthropic Claude models (`Claude 3 Haiku`, `Claude 3.5 Haiku`, `Claude Sonnet 4`) to balance speed and quality.
- **Caching**: Results are cached to provide instant responses for previously analyzed posts.
- **Usage Tracking**: Per-user rate limiting and daily quotas to manage API usage.

## How to Use

Simply paste a Reddit post URL into any channel where the bot is present. The bot will automatically fetch the data, analyze it, and post a summary.

## Architecture

The diagram below illustrates the bot's request lifecycle from the moment a user provides a Reddit URL until the analysis is returned.

```mermaid
sequenceDiagram
    participant User
    participant DiscordBot as Discord Bot
    participant Cache
    participant RedditAPI as Reddit API
    participant AnthropicAPI as Anthropic API

    User->>+DiscordBot: Pastes Reddit URL
    DiscordBot->>+Cache: Check for cached result
    alt Cache Hit
        Cache-->>-DiscordBot: Return cached summary
    else Cache Miss
        DiscordBot->>+RedditAPI: Fetch post & comments
        RedditAPI-->>-DiscordBot: Return post data
        DiscordBot->>+AnthropicAPI: Request summary & analysis
        AnthropicAPI-->>-Discord-Bot: Return analysis
        DiscordBot->>+Cache: Store new result
        Cache-->>-DiscordBot: Confirm storage
    end
    DiscordBot-->>-User: Send analysis
```

### Commands

The bot uses Discord slash commands for configuration and other actions:

-   `/start`: Displays a welcome message.
-   `/help`: Shows a list of all available commands.
-   `/model`: Allows you to switch the AI model used for analysis.
    -   `model`: `haiku3` (fastest), `haiku35` (balanced), `sonnet4` (highest quality).
-   `/usage`: Checks your current daily API usage.
-   `/whoami`: Displays your Discord user ID.
-   `/cache`: Shows statistics about the analysis cache.
-   `/clearcache`: Clears the cache manually.

## Setup and Installation

You can run the bot locally using Python or deploy it as a Docker container.

### Prerequisites

-   Python 3.11+
-   Git
-   An account with [Discord](https://discord.com/developers/applications), [Reddit](https://www.reddit.com/prefs/apps), and [Anthropic](https://console.anthropic.com/) to get API credentials.

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shuienko/rpostsummarybot.git
    cd rpostsummarybot
    ```

2.  **Set up a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create a file named `.env` in the root of the project and add the following variables:
    ```env
    DISCORD_TOKEN="your_discord_bot_token"
    REDDIT_CLIENT_ID="your_reddit_client_id"
    REDDIT_CLIENT_SECRET="your_reddit_client_secret"
    ANTHROPIC_API_KEY="your_anthropic_api_key"
    ```

5.  **Run the bot:**
    ```bash
    python bot.py
    ```

### Docker Deployment

The included `Dockerfile` allows you to run the bot in a container.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shuienko/rpostsummarybot.git
    cd rpostsummarybot
    ```

2.  **Configure environment variables:**
    Create a directory named `config` and place your `.env` file inside it.
    ```bash
    mkdir config
    # Create and edit config/.env with your variables
    nano config/.env
    ```

3.  **Build the Docker image:**
    ```bash
    docker build -t rpostsummarybot .
    ```

4.  **Run the Docker container:**
    This command mounts the `config` directory into the container.
    ```bash
    docker run -d --name rpostsummarybot -v "$(pwd)/config":/app/config rpostsummarybot
    ```

## Environment Variables

The bot requires the following environment variables to function:

-   `DISCORD_TOKEN`: The token for your Discord bot.
-   `REDDIT_CLIENT_ID`: The client ID from your Reddit script application.
-   `REDDIT_CLIENT_SECRET`: The client secret from your Reddit script application.
-   `ANTHROPIC_API_KEY`: Your API key for the Anthropic (Claude) API.

## License

This project is licensed under the terms of the license specified in the `LICENSE` file. 