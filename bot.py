import os
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import asyncpraw
import aiohttp
from typing import Tuple, List, Dict
from telegram.constants import MessageLimit
from dotenv import load_dotenv
import anthropic
import json

# Load environment variables
load_dotenv()

# Initialize Reddit API client
reddit = asyncpraw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent="python:rpostsummarybot:v1.0"
)

# Telegram bot token from environment variable
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

class RedditAnalyzer:
    def __init__(self):
        self.client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
        self.model = "claude-3-haiku-20240307"
        
    async def _call_anthropic(self, prompt: str) -> Tuple[str, int, int]:
        """Make a call to Anthropic API and return response with token counts"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract response and token counts
            response = message.content[0].text
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            
            # Log token usage
            print(json.dumps({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "prompt_type": prompt[:100] + "..."  # Log first 100 chars of prompt for context
            }, indent=2))
            
            return response.strip(), input_tokens, output_tokens
            
        except Exception as e:
            print(f"Error calling Anthropic API: {str(e)}")
            return "Error processing request", 0, 0

    async def get_post_summary(self, post_content: str) -> str:
        """Generate summary of Reddit post using Claude"""
        prompt = f"Summarize this Reddit post concisely: {post_content}"
        response, _, _ = await self._call_anthropic(prompt)
        return response

    async def analyze_comment_tone(self, comment: str) -> str:
        """Analyze if comment tone is positive or negative using Claude"""
        prompt = f"Analyze if this comment is positive or negative. Reply with only 'positive' or 'negative': {comment}"
        response, _, _ = await self._call_anthropic(prompt)
        result_text = response.lower()
        return 'positive' if 'positive' in result_text else 'negative'

    async def summarize_comments(self, comments: List[str]) -> str:
        """Generate a consolidated summary of multiple comments using Claude"""
        comments_text = "\n".join(comments[:10])  # Ensure we only take top 10
        prompt = (
            "Below are the top 10 comments from a Reddit post. "
            "Provide a 2-3 sentence summary that captures the main themes "
            "and overall sentiment of these comments. Reply with just with the summary. "
            "Don't include anything else in your response. Comments:\n\n" + comments_text
        )
        response, _, _ = await self._call_anthropic(prompt)
        return response

class RedditBot:
    def __init__(self):
        self.analyzer = RedditAnalyzer()

    async def analyze_reddit_post(self, post_url: str) -> Tuple[str, str, Dict]:
        """Analyze Reddit post and return post summary, comments summary, and sentiment stats"""
        try:
            # Extract post ID from URL
            post_id = post_url.split('/')[-3]
            submission = await reddit.submission(id=post_id)
            
            # Get post summary
            post_content = submission.selftext if submission.selftext else submission.title
            post_summary = await self.analyzer.get_post_summary(post_content)
            
            # Get top 10 comments
            await submission.comments.replace_more(limit=0)
            top_comments = sorted(submission.comments, key=lambda x: x.score, reverse=True)[:10]
            
            # Collect comment texts and analyze sentiment
            comment_texts = []
            sentiment_counts = {'positive': 0, 'negative': 0}
            
            for comment in top_comments:
                comment_texts.append(comment.body)
                tone = await self.analyzer.analyze_comment_tone(comment.body)
                sentiment_counts[tone] += 1
            
            # Get consolidated comments summary
            comments_summary = await self.analyzer.summarize_comments(comment_texts)
            
            # Calculate percentages
            total_comments = len(comment_texts)
            sentiment_stats = {
                'positive_percent': (sentiment_counts['positive'] / total_comments) * 100,
                'negative_percent': (sentiment_counts['negative'] / total_comments) * 100
            }
            
            return post_summary, comments_summary, sentiment_stats
            
        except Exception as e:
            print(f"Error analyzing post: {str(e)}")
            return None, None, None

async def chunk_message(text: str, max_length: int = MessageLimit.MAX_TEXT_LENGTH) -> List[str]:
    """Split a message into chunks that fit Telegram's message size limit"""
    chunks = []
    current_chunk = ""

    for line in text.split('\n'):
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + '\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + '\n'

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

async def start(update, context):
    """Send a message when the command /start is issued."""
    welcome_message = (
        "Welcome! Send me a Reddit post URL, and I'll analyze it for you.\n"
        "I'll provide:\n"
        "- Post summary\n"
        "- Summary of top comments\n"
        "- Sentiment analysis of comments"
    )
    await update.message.reply_text(welcome_message)

async def analyze_url(update, context):
    """Analyze the Reddit URL sent by user"""
    url = update.message.text
    if "reddit.com" not in url:
        await update.message.reply_text("Please send a valid Reddit post URL")
        return

    await update.message.reply_text("Analyzing post... Please wait.")
    
    bot = RedditBot()
    post_summary, comments_summary, sentiment_stats = await bot.analyze_reddit_post(url)
    
    if not post_summary:
        await update.message.reply_text("Error analyzing the post. Please try again.")
        return

    # Format response
    response = f"📝 Post Summary\n{post_summary}\n\n\n"
    response += f"💭 Comments Overview\n{comments_summary}\n\n\n"
    response += f"📊 Sentiment Analysis:\n"
    response += f"Positive: {sentiment_stats['positive_percent']:.1f}%\n"
    response += f"Negative: {sentiment_stats['negative_percent']:.1f}%"

    # Split response into chunks and send multiple messages if needed
    chunks = await chunk_message(response)
    for chunk in chunks:
        await update.message.reply_text(chunk)

def main():
    """Start the bot."""
    # Verify environment variables
    required_vars = ['TELEGRAM_TOKEN', 'REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'ANTHROPIC_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        return

    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_url))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()