import os
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import asyncpraw
from typing import Tuple, List, Dict
from telegram.constants import MessageLimit
from dotenv import load_dotenv
import anthropic
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up rotating file handler
log_file = os.path.join(log_dir, "bot.log")
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Set up console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure logger
logger = logging.getLogger("RedditBot")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

# Initialize Reddit API client
reddit = asyncpraw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent="python:rpostsummarybot:v1.0"
)

# Anthropic API Key
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Telegram bot token and allowed user ID from environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
ALLOWED_USER_ID = int(os.getenv('ALLOWED_USER_ID', 0))

# Usage tracking
class UsageTracker:
    def __init__(self):
        self.daily_usage = defaultdict(int)
        self.user_requests = defaultdict(list)
        self.rate_limits = {}  # user_id -> timestamp of last allowed request
        self.MAX_REQUESTS_PER_DAY = 20
        self.RATE_LIMIT_SECONDS = 60  # 1 request per minute
        
    def can_make_request(self, user_id: int) -> bool:
        """Check if user can make a request based on rate limits and daily quota"""
        # Check rate limit (requests per minute)
        current_time = time.time()
        if user_id in self.rate_limits:
            time_since_last = current_time - self.rate_limits[user_id]
            if time_since_last < self.RATE_LIMIT_SECONDS:
                return False
                
        # Check daily quota
        today = datetime.now().strftime("%Y-%m-%d")
        if self.daily_usage[f"{user_id}:{today}"] >= self.MAX_REQUESTS_PER_DAY:
            return False
            
        return True
        
    def record_request(self, user_id: int, request_type: str):
        """Record a request from a user"""
        current_time = time.time()
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Update rate limit timestamp
        self.rate_limits[user_id] = current_time
        
        # Update daily usage
        self.daily_usage[f"{user_id}:{today}"] += 1
        
        # Record request details
        self.user_requests[user_id].append({
            "timestamp": current_time,
            "type": request_type
        })
        
    def get_usage_stats(self, user_id: int) -> str:
        """Get usage statistics for a user"""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_count = self.daily_usage[f"{user_id}:{today}"]
        remaining = self.MAX_REQUESTS_PER_DAY - daily_count
        
        return f"Today's usage: {daily_count}/{self.MAX_REQUESTS_PER_DAY}\nRemaining: {remaining}"

# Initialize usage tracker
usage_tracker = UsageTracker()

# Simple cache implementation
class ResultCache:
    def __init__(self, max_size=100, ttl_hours=24):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        
    def _generate_key(self, url: str) -> str:
        """Generate a cache key from a URL"""
        return hashlib.md5(url.encode()).hexdigest()
        
    def get(self, url: str) -> Tuple[str, str, Dict]:
        """Get cached result for a URL if it exists and is not expired"""
        key = self._generate_key(url)
        
        if key in self.cache:
            # Check if entry has expired
            if datetime.now() - self.access_times[key] > self.ttl:
                # Remove expired entry
                del self.cache[key]
                del self.access_times[key]
                return None
                
            # Update access time
            self.access_times[key] = datetime.now()
            return self.cache[key]
            
        return None
        
    def set(self, url: str, result: Tuple[str, str, Dict]):
        """Cache a result for a URL"""
        key = self._generate_key(url)
        
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            
        # Store result and access time
        self.cache[key] = result
        self.access_times[key] = datetime.now()
        
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_times.clear()
        
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_hours": self.ttl.total_seconds() / 3600
        }

# Initialize cache
result_cache = ResultCache()

# Authentication decorator
def restricted(func):
    async def wrapped(update, context, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id != ALLOWED_USER_ID:
            print(f"Unauthorized access denied for user {user_id}")
            await update.message.reply_text(
                "Sorry, this bot is private. Access denied."
            )
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

class RedditAnalyzer:
    def __init__(self):
        self.client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
        # Default to haiku for cost efficiency, but allow switching to more capable models
        self.models = {
            "fast": "claude-3-haiku-20240307",
            "balanced": "claude-3-sonnet-20240229",
            "powerful": "claude-3-opus-20240229"
        }
        self.current_model = "fast"
        logger.info(f"RedditAnalyzer initialized with default model: {self.models[self.current_model]}")
        
    def set_model(self, model_key: str):
        """Set the model to use for analysis"""
        if model_key in self.models:
            self.current_model = model_key
            logger.info(f"Model changed to: {model_key} ({self.models[model_key]})")
            return True
        logger.warning(f"Invalid model key requested: {model_key}")
        return False
        
    async def _call_anthropic(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, int, int]:
        """Make a call to Anthropic API and return response with token counts"""
        try:
            logger.debug(f"Calling Anthropic API with model: {self.models[self.current_model]}")
            message = self.client.messages.create(
                model=self.models[self.current_model],
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract response and token counts
            response = message.content[0].text
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            
            # Log token usage
            usage_info = {
                "model": self.current_model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "prompt_type": prompt[:100] + "..."
            }
            logger.info(f"API call completed: {json.dumps(usage_info)}")
            
            return response.strip(), input_tokens, output_tokens
            
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}", exc_info=True)
            return "Error processing request", 0, 0

    async def get_post_summary(self, post_content: str) -> str:
        """Generate summary of Reddit post using Claude"""
        prompt = ( 
            "You are analyzing a Reddit post. Your task is to create a concise, informative summary.\n\n"
            "Guidelines:\n"
            "- Create a summary using 3-4 sentences maximum\n"
            "- Capture the main points, questions, or issues raised\n"
            "- Maintain a neutral tone\n"
            "- Don't create bullet points or numbered lists\n"
            "- If the post is very short, simply note that it's brief and summarize the key point\n\n"
            "Reddit Post:\n\n" + post_content
        )
        response, _, _ = await self._call_anthropic(prompt)
        return response

    async def analyze_comment_tone(self, comment: str) -> Dict[str, str]:
        """Analyze comment tone with more detailed sentiment analysis"""
        prompt = (
            "Analyze the sentiment and tone of this Reddit comment. Consider the overall tone, word choice, and context.\n\n"
            "Comment: \"" + comment + "\"\n\n"
            "Provide a JSON response with the following fields:\n"
            "- primary_sentiment: Either 'positive', 'negative', or 'neutral'\n"
            "- emotion: The primary emotion expressed (e.g., 'happy', 'angry', 'surprised', 'curious', etc.)\n"
            "- intensity: A rating from 1-5 where 1 is mild and 5 is extreme\n\n"
            "Format your response as valid JSON only, with no additional text."
        )
        response, _, _ = await self._call_anthropic(prompt)
        
        try:
            # Try to parse the JSON response
            result = json.loads(response)
            
            # Validate primary_sentiment field - ensure it's one of the expected values
            valid_sentiments = ['positive', 'negative', 'neutral']
            if 'primary_sentiment' not in result or result['primary_sentiment'].lower() not in valid_sentiments:
                # If invalid or missing, default to a valid sentiment based on the response
                if 'positive' in response.lower():
                    result['primary_sentiment'] = 'positive'
                elif 'negative' in response.lower():
                    result['primary_sentiment'] = 'negative'
                else:
                    result['primary_sentiment'] = 'neutral'
                logger.warning(f"Invalid sentiment value received, defaulted to: {result['primary_sentiment']}")
            else:
                # Normalize to lowercase
                result['primary_sentiment'] = result['primary_sentiment'].lower()
                
            # Ensure other required fields exist
            if 'emotion' not in result:
                result['emotion'] = 'unknown'
            if 'intensity' not in result or not isinstance(result['intensity'], (int, float)) or result['intensity'] < 1 or result['intensity'] > 5:
                result['intensity'] = 3
                
            return result
        except json.JSONDecodeError:
            # Fallback to simple sentiment if JSON parsing fails
            logger.warning(f"Failed to parse JSON response: {response[:100]}...")
            return {
                'primary_sentiment': 'positive' if 'positive' in response.lower() else 'negative',
                'emotion': 'unknown',
                'intensity': 3
            }

    async def summarize_comments(self, comments: List[str]) -> str:
        """Generate a consolidated summary of multiple comments using Claude"""
        comments_text = "\n\n---\n\n".join(comments[:10])  # Ensure we only take top 10
        prompt = (
            "You are analyzing the top comments on a Reddit post. Your task is to create a concise summary that captures "
            "the overall sentiment and main points from these comments.\n\n"
            "Guidelines:\n"
            "- Provide a 3-4 sentence summary that captures the main themes and reactions\n"
            "- Note any significant agreements or disagreements among commenters\n"
            "- Highlight any particularly insightful or helpful comments\n"
            "- Maintain a neutral tone in your summary\n\n"
            "Comments:\n\n" + comments_text
        )
        response, _, _ = await self._call_anthropic(prompt)
        return response

class RedditBot:
    def __init__(self):
        self.analyzer = RedditAnalyzer()

    async def analyze_reddit_post(self, post_url: str) -> Tuple[str, str, Dict]:
        """Analyze Reddit post and return post summary, comments summary, and sentiment stats"""
        # Check cache first
        cached_result = result_cache.get(post_url)
        if cached_result:
            logger.info(f"Cache hit for URL: {post_url}")
            return cached_result
            
        logger.info(f"Cache miss for URL: {post_url}")
        
        try:
            # Check if the URL is a short link and resolve it
            if "redd.it" in post_url or "/s/" in post_url:
                logger.debug(f"Processing short URL: {post_url}")
                submission = await reddit.submission(url=post_url)
            else:
                # Extract post ID from standard Reddit URL
                try:
                    post_id = post_url.split('/')[-3]
                    logger.debug(f"Extracted post ID: {post_id} from URL: {post_url}")
                    submission = await reddit.submission(id=post_id)
                except (IndexError, ValueError) as e:
                    logger.error(f"Failed to parse URL: {post_url}, error: {str(e)}")
                    return "Invalid Reddit URL format", "Could not parse the URL", {}
            
            logger.info(f"Retrieved submission: {submission.title[:50]}...")
            
            # Get post summary
            post_content = submission.selftext if submission.selftext else submission.title
            post_summary = await self.analyzer.get_post_summary(post_content)
            
            # Get top 10 comments
            await submission.comments.replace_more(limit=0)
            top_comments = sorted(submission.comments, key=lambda x: x.score, reverse=True)[:10]
            logger.info(f"Retrieved {len(top_comments)} top comments")
            
            # Collect comment texts and analyze sentiment
            comment_texts = []
            comment_analyses = []
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            emotion_counts = defaultdict(int)
            total_intensity = 0
            
            for comment in top_comments:
                comment_texts.append(comment.body)
                analysis = await self.analyzer.analyze_comment_tone(comment.body)
                comment_analyses.append(analysis)
                
                # Count sentiments - ensure we only use valid keys
                sentiment = analysis.get('primary_sentiment', 'neutral')
                # Double-check that sentiment is one of our expected values
                if sentiment not in sentiment_counts:
                    logger.warning(f"Unexpected sentiment value: {sentiment}, defaulting to neutral")
                    sentiment = 'neutral'
                sentiment_counts[sentiment] += 1
                
                # Count emotions
                emotion = analysis.get('emotion', 'unknown')
                emotion_counts[emotion] += 1
                
                # Sum intensities
                intensity = analysis.get('intensity', 3)
                total_intensity += intensity
            
            # Handle case with no comments
            if not comment_texts:
                logger.warning(f"No comments found for post: {submission.title[:50]}...")
                comments_summary = "No comments found on this post."
                sentiment_stats = {
                    'sentiment_breakdown': {'positive': 0, 'negative': 0, 'neutral': 0},
                    'top_emotions': [],
                    'avg_intensity': 0
                }
            else:
                # Get consolidated comments summary
                comments_summary = await self.analyzer.summarize_comments(comment_texts)
                
                # Calculate percentages and stats
                total_comments = len(comment_texts)
                
                # Get top 3 emotions
                top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                top_emotions = [{'emotion': e, 'count': c, 'percent': (c/total_comments)*100} 
                               for e, c in top_emotions]
                
                sentiment_stats = {
                    'sentiment_breakdown': {
                        'positive': (sentiment_counts['positive'] / total_comments) * 100,
                        'negative': (sentiment_counts['negative'] / total_comments) * 100,
                        'neutral': (sentiment_counts['neutral'] / total_comments) * 100
                    },
                    'top_emotions': top_emotions,
                    'avg_intensity': total_intensity / total_comments if total_comments > 0 else 0
                }
                
                logger.info(f"Analysis complete - Sentiment breakdown: {sentiment_counts}")
            
            # Cache the result
            result = (post_summary, comments_summary, sentiment_stats)
            result_cache.set(post_url, result)
            
            return result
            
        except asyncpraw.exceptions.RedditAPIException as e:
            logger.error(f"Reddit API error: {str(e)}", exc_info=True)
            return f"Reddit API error: {str(e)}", "Could not retrieve post data", {}
        except Exception as e:
            logger.error(f"Error analyzing post: {str(e)}", exc_info=True)
            return f"Error analyzing post: {str(e)}", "Analysis failed", {}

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

@restricted
async def start(update, context):
    """Send a message when the command /start is issued."""
    welcome_message = (
        "Welcome! Send me a Reddit post URL, and I'll analyze it for you.\n"
        "I'll provide:\n"
        "- Post summary\n"
        "- Summary of top comments\n"
        "- Sentiment analysis of comments\n\n"
        "Type /help to see all available commands."
    )
    await update.message.reply_text(welcome_message)

@restricted
async def help_command(update, context):
    """Send a message with all available commands."""
    help_message = (
        "📋 Available Commands:\n\n"
        "/start - Start the bot and see welcome message\n"
        "/help - Show this help message\n"
        "/usage - Check your usage statistics\n"
        "/whoami - Check your Telegram user ID\n"
        "/model - Change the AI model (fast/balanced/powerful)\n"
        "/cache - Check cache statistics\n"
        "/clearcache - Clear the cache\n\n"
        "Simply send a Reddit post URL to analyze it."
    )
    await update.message.reply_text(help_message)

@restricted
async def set_model(update, context):
    """Change the AI model used for analysis."""
    if not context.args or len(context.args) != 1:
        await update.message.reply_text(
            "Please specify a model: fast, balanced, or powerful\n"
            "Example: /model balanced\n\n"
            "- fast: Quickest, most economical (Claude 3 Haiku)\n"
            "- balanced: Good balance of speed and quality (Claude 3 Sonnet)\n"
            "- powerful: Highest quality but slower (Claude 3 Opus)"
        )
        return
        
    model_choice = context.args[0].lower()
    
    # Create a temporary analyzer to check if model is valid
    analyzer = RedditAnalyzer()
    if analyzer.set_model(model_choice):
        # Store the model choice in user data
        if not hasattr(context.bot_data, 'user_models'):
            context.bot_data['user_models'] = {}
        
        context.bot_data['user_models'][update.effective_user.id] = model_choice
        
        await update.message.reply_text(
            f"✅ Model set to '{model_choice}'\n"
            f"Model: {analyzer.models[model_choice]}"
        )
    else:
        await update.message.reply_text(
            "❌ Invalid model. Please choose from: fast, balanced, or powerful"
        )

@restricted
async def analyze_url(update, context):
    """Analyze the Reddit URL sent by user"""
    user_id = update.effective_user.id
    
    # Check rate limits
    if not usage_tracker.can_make_request(user_id):
        today = datetime.now().strftime("%Y-%m-%d")
        daily_count = usage_tracker.daily_usage[f"{user_id}:{today}"]
        
        if daily_count >= usage_tracker.MAX_REQUESTS_PER_DAY:
            await update.message.reply_text(
                "⚠️ You've reached your daily limit of requests. "
                "Please try again tomorrow."
            )
        else:
            await update.message.reply_text(
                "⚠️ Rate limit exceeded. Please wait at least 1 minute between requests."
            )
        return
    
    url = update.message.text.strip()
    
    # Improved URL validation
    valid_domains = ["reddit.com", "redd.it", "www.reddit.com", "old.reddit.com"]
    is_valid = any(domain in url for domain in valid_domains)
    
    if not is_valid:
        await update.message.reply_text(
            "Please send a valid Reddit post URL.\n"
            "Examples:\n"
            "- https://www.reddit.com/r/subreddit/comments/postid/title\n"
            "- https://redd.it/postid"
        )
        return

    # Record this request
    usage_tracker.record_request(user_id, "analyze_url")
    
    await update.message.reply_text("Analyzing post... Please wait.")
    
    bot = RedditBot()
    
    # Set the model based on user preference if available
    if hasattr(context.bot_data, 'user_models') and user_id in context.bot_data['user_models']:
        model_choice = context.bot_data['user_models'][user_id]
        bot.analyzer.set_model(model_choice)
    
    post_summary, comments_summary, sentiment_stats = await bot.analyze_reddit_post(url)
    
    if post_summary.startswith("Error") or post_summary.startswith("Invalid") or post_summary.startswith("Reddit API error"):
        await update.message.reply_text(f"❌ {post_summary}\n{comments_summary}")
        return

    # Format response
    response = f"📝 ----- Post Summary -----\n{post_summary}\n\n\n"
    response += f"💭 ----- Comments Overview -----\n{comments_summary}\n\n\n"
    response += f"📊 ----- Sentiment Analysis -----\n"
    
    # Only show percentages if we have comment data
    if sentiment_stats.get('sentiment_breakdown', {}).get('positive', 0) > 0 or \
       sentiment_stats.get('sentiment_breakdown', {}).get('negative', 0) > 0:
        
        breakdown = sentiment_stats['sentiment_breakdown']
        response += f"Sentiment Breakdown:\n"
        response += f"- Positive: {breakdown['positive']:.1f}%\n"
        response += f"- Negative: {breakdown['negative']:.1f}%\n"
        response += f"- Neutral: {breakdown['neutral']:.1f}%\n\n"
        
        if sentiment_stats.get('top_emotions'):
            response += f"Top Emotions:\n"
            for emotion in sentiment_stats['top_emotions']:
                response += f"- {emotion['emotion'].capitalize()}: {emotion['percent']:.1f}%\n"
            
        response += f"\nAverage Intensity: {sentiment_stats['avg_intensity']:.1f}/5"
    else:
        response += "No sentiment data available (post may have no comments)"

    # Split response into chunks and send multiple messages if needed
    chunks = await chunk_message(response)
    for chunk in chunks:
        await update.message.reply_text(chunk)

@restricted
async def whoami(update, context):
    """Command to check user ID"""
    await update.message.reply_text(f"Your Telegram User ID is: {update.effective_user.id}")

@restricted
async def usage(update, context):
    """Command to check usage statistics"""
    user_id = update.effective_user.id
    stats = usage_tracker.get_usage_stats(user_id)
    await update.message.reply_text(f"📊 Usage Statistics\n\n{stats}")

@restricted
async def cache_stats(update, context):
    """Command to check cache statistics"""
    stats = result_cache.get_stats()
    stats_message = (
        f"📊 Cache Statistics\n\n"
        f"Size: {stats['size']}/{stats['max_size']} entries\n"
        f"TTL: {stats['ttl_hours']} hours"
    )
    await update.message.reply_text(stats_message)

@restricted
async def clear_cache(update, context):
    """Command to clear the cache"""
    result_cache.clear()
    await update.message.reply_text("✅ Cache cleared successfully")

def main():
    """Start the bot."""
    required_vars = ['TELEGRAM_TOKEN', 'REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'ANTHROPIC_API_KEY', 'ALLOWED_USER_ID']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        return

    logger.info("Starting bot...")
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("whoami", whoami))
    application.add_handler(CommandHandler("usage", usage))
    application.add_handler(CommandHandler("model", set_model))
    application.add_handler(CommandHandler("cache", cache_stats))
    application.add_handler(CommandHandler("clearcache", clear_cache))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_url))

    # Start the Bot
    logger.info("Bot started, polling for updates...")
    application.run_polling()

if __name__ == '__main__':
    main()