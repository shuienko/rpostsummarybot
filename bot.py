import os
import discord
from discord.ext import commands
import asyncpraw
from typing import Tuple, List, Dict
from dotenv import load_dotenv
import anthropic
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import asyncio

# Load environment variables
load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

# Configuration from environment variables with defaults
MAX_REQUESTS_PER_DAY = int(os.getenv('MAX_REQUESTS_PER_DAY', '30'))
RATE_LIMIT_SECONDS = int(os.getenv('RATE_LIMIT_SECONDS', '30'))
DEFAULT_AI_MODEL = os.getenv('DEFAULT_AI_MODEL', 'haiku3')

# Cache configuration
CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', '100'))
CACHE_TTL_HOURS = int(os.getenv('CACHE_TTL_HOURS', '24'))

# Logging configuration
LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', '5242880'))  # 5MB
LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))

# AI Analysis settings
MAX_TOKENS_PER_REQUEST = int(os.getenv('MAX_TOKENS_PER_REQUEST', '1024'))
MAX_COMMENTS_TO_ANALYZE = int(os.getenv('MAX_COMMENTS_TO_ANALYZE', '10'))
MAX_EMOTIONS_TO_SHOW = int(os.getenv('MAX_EMOTIONS_TO_SHOW', '3'))
SENTIMENT_INTENSITY_MIN = int(os.getenv('SENTIMENT_INTENSITY_MIN', '1'))
SENTIMENT_INTENSITY_MAX = int(os.getenv('SENTIMENT_INTENSITY_MAX', '5'))
DEFAULT_INTENSITY_VALUE = int(os.getenv('DEFAULT_INTENSITY_VALUE', '3'))

# Discord message settings
DISCORD_MESSAGE_MAX_LENGTH = int(os.getenv('DISCORD_MESSAGE_MAX_LENGTH', '2000'))
DISCORD_TITLE_MAX_LENGTH = int(os.getenv('DISCORD_TITLE_MAX_LENGTH', '50'))

# Summary settings
POST_SUMMARY_MAX_SENTENCES = int(os.getenv('POST_SUMMARY_MAX_SENTENCES', '4'))
COMMENT_SUMMARY_MAX_SENTENCES = int(os.getenv('COMMENT_SUMMARY_MAX_SENTENCES', '4'))

# Reddit user agent
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'python:rpostsummarybot:v1.0')

# Logging truncation settings
LOG_TRUNCATE_LENGTH = int(os.getenv('LOG_TRUNCATE_LENGTH', '50'))

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up rotating file handler
log_file = os.path.join(log_dir, "bot.log")
file_handler = RotatingFileHandler(log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Set up console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure logger
logger = logging.getLogger("RedditBot")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Usage tracking
class UsageTracker:
    def __init__(self):
        self.daily_usage = defaultdict(int)
        self.user_requests = defaultdict(list)
        self.rate_limits = {}  # user_id -> timestamp of last allowed request
        self.MAX_REQUESTS_PER_DAY = MAX_REQUESTS_PER_DAY
        self.RATE_LIMIT_SECONDS = RATE_LIMIT_SECONDS
        
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
    def __init__(self, max_size=None, ttl_hours=None):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size or CACHE_MAX_SIZE
        self.ttl = timedelta(hours=ttl_hours or CACHE_TTL_HOURS)
        
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

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=None, intents=intents)

class RedditAnalyzer:
    def __init__(self):
        self.client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
        # Default to haiku for cost efficiency, but allow switching to more capable models
        self.models = {
            "haiku3":  "claude-3-haiku-20240307",
            "haiku35": "claude-3-5-haiku-latest",
            "sonnet4": "claude-sonnet-4-0"
        }
        logger.info(f"RedditAnalyzer initialized with models: {list(self.models.keys())}")

    def get_model_name(self, model_key: str) -> str:
        """Get the full model name from its key, with fallback."""
        return self.models.get(model_key, self.models["haiku3"])
        
    async def _call_anthropic(self, prompt: str, model_key: str, max_tokens: int = None) -> Tuple[str, int, int]:
        """Make a call to Anthropic API and return response with token counts"""
        if max_tokens is None:
            max_tokens = MAX_TOKENS_PER_REQUEST
        model_name = self.get_model_name(model_key)
        try:
            logger.debug(f"Calling Anthropic API with model: {model_name}")
            message = self.client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract response and token counts
            response = message.content[0].text
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            
            # Log token usage
            usage_info = {
                "model": model_key,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "prompt_type": prompt[:LOG_TRUNCATE_LENGTH] + "..."
            }
            logger.info(f"API call completed: {json.dumps(usage_info)}")
            
            return response.strip(), input_tokens, output_tokens
            
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}", exc_info=True)
            return "Error processing request", 0, 0

    async def get_post_summary(self, post_content: str, model_key: str) -> str:
        """Generate summary of Reddit post using Claude"""
        prompt = ( 
            "You are analyzing a Reddit post. Your task is to create a concise, informative summary.\n\n"
            "Guidelines:\n"
            f"- Create a summary using {POST_SUMMARY_MAX_SENTENCES} sentences maximum\n"
            "- Capture the main points, questions, or issues raised\n"
            "- Maintain a neutral tone\n"
            "- Don't create bullet points or numbered lists\n"
            "- If the post is very short, simply note that it's brief and summarize the key point\n\n"
            "Reddit Post:\n\n" + post_content
        )
        response, _, _ = await self._call_anthropic(prompt, model_key)
        return response

    async def analyze_comment_tone(self, comment: str, model_key: str) -> Dict[str, str]:
        """Analyze comment tone with more detailed sentiment analysis"""
        prompt = (
            "Analyze the sentiment and tone of this Reddit comment. Consider the overall tone, word choice, and context.\n\n"
            "Comment: \"" + comment + "\"\n\n"
            "Provide a JSON response with the following fields:\n"
            "- primary_sentiment: Either 'positive', 'negative', or 'neutral'\n"
            "- emotion: The primary emotion expressed (e.g., 'happy', 'angry', 'surprised', 'curious', etc.)\n"
            f"- intensity: A rating from {SENTIMENT_INTENSITY_MIN}-{SENTIMENT_INTENSITY_MAX} where {SENTIMENT_INTENSITY_MIN} is mild and {SENTIMENT_INTENSITY_MAX} is extreme\n\n"
            "Format your response as valid JSON only, with no additional text."
        )
        response, _, _ = await self._call_anthropic(prompt, model_key)
        
        try:
            # Clean the response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                # Remove opening ```json
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith('```'):
                # Remove opening ```
                cleaned_response = cleaned_response[3:]
            
            if cleaned_response.endswith('```'):
                # Remove closing ```
                cleaned_response = cleaned_response[:-3]
            
            # Try to parse the cleaned JSON response
            result = json.loads(cleaned_response.strip())
            
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
            if 'intensity' not in result or not isinstance(result['intensity'], (int, float)) or result['intensity'] < SENTIMENT_INTENSITY_MIN or result['intensity'] > SENTIMENT_INTENSITY_MAX:
                result['intensity'] = DEFAULT_INTENSITY_VALUE
                
            return result
        except json.JSONDecodeError:
            # Fallback to simple sentiment if JSON parsing fails
            logger.warning(f"Failed to parse JSON response: {response[:LOG_TRUNCATE_LENGTH]}...")
            return {
                'primary_sentiment': 'positive' if 'positive' in response.lower() else 'negative',
                'emotion': 'unknown',
                'intensity': DEFAULT_INTENSITY_VALUE
            }

    async def summarize_comments(self, comments: List[str], model_key: str) -> str:
        """Generate a consolidated summary of multiple comments using Claude"""
        comments_text = "\n\n---\n\n".join(comments[:MAX_COMMENTS_TO_ANALYZE])  # Ensure we only take top comments
        prompt = (
            "You are analyzing the top comments on a Reddit post. Your task is to create a concise summary that captures "
            "the overall sentiment and main points from these comments.\n\n"
            "Guidelines:\n"
            f"- Provide a {COMMENT_SUMMARY_MAX_SENTENCES} sentence summary that captures the main themes and reactions\n"
            "- Note any significant agreements or disagreements among commenters\n"
            "- Highlight any particularly insightful or helpful comments\n"
            "- Maintain a neutral tone in your summary\n\n"
            "Comments:\n\n" + comments_text
        )
        response, _, _ = await self._call_anthropic(prompt, model_key)
        return response

class RedditBot:
    def __init__(self):
        self.analyzer = RedditAnalyzer()
        self.reddit = None
    
    async def _get_reddit_client(self):
        """Get or create Reddit client in async context"""
        if self.reddit is None:
            self.reddit = asyncpraw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=REDDIT_USER_AGENT
            )
        return self.reddit

    async def cleanup(self):
        """Clean up resources"""
        if self.reddit:
            await self.reddit.close()
            self.reddit = None
            logger.info("RedditBot cleanup completed")

    async def analyze_reddit_post(self, post_url: str, model_key: str) -> Tuple[str, str, Dict]:
        """Analyze Reddit post and return post summary, comments summary, and sentiment stats"""
        # Check cache first
        cached_result = result_cache.get(post_url)
        if cached_result:
            logger.info(f"Cache hit for URL: {post_url}")
            return cached_result
            
        logger.info(f"Cache miss for URL: {post_url}")
        
        try:
            # Get Reddit client in async context
            reddit_client = await self._get_reddit_client()
            
            # Let asyncpraw handle URL parsing, which is more robust
            submission = await reddit_client.submission(url=post_url)
            
            logger.info(f"Retrieved submission: {submission.title[:DISCORD_TITLE_MAX_LENGTH]}...")
            
            # Get post summary
            post_content = submission.selftext if submission.selftext else submission.title
            post_summary = await self.analyzer.get_post_summary(post_content, model_key)
            
            # Get top comments
            await submission.comments.replace_more(limit=0)
            top_comments = sorted(submission.comments, key=lambda x: x.score, reverse=True)[:MAX_COMMENTS_TO_ANALYZE]
            logger.info(f"Retrieved {len(top_comments)} top comments")
            
            # Collect comment texts and analyze sentiment
            comment_texts = []
            comment_analyses = []
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            emotion_counts = defaultdict(int)
            total_intensity = 0
            
            for comment in top_comments:
                comment_texts.append(comment.body)
                analysis = await self.analyzer.analyze_comment_tone(comment.body, model_key)
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
                intensity = analysis.get('intensity', DEFAULT_INTENSITY_VALUE)
                total_intensity += intensity
            
            # Handle case with no comments
            if not comment_texts:
                logger.warning(f"No comments found for post: {submission.title[:DISCORD_TITLE_MAX_LENGTH]}...")
                comments_summary = "No comments found on this post."
                sentiment_stats = {
                    'sentiment_breakdown': {'positive': 0, 'negative': 0, 'neutral': 0},
                    'top_emotions': [],
                    'avg_intensity': 0
                }
            else:
                # Get consolidated comments summary
                comments_summary = await self.analyzer.summarize_comments(comment_texts, model_key)
                
                # Calculate percentages and stats
                total_comments = len(comment_texts)
                
                # Get top emotions
                top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:MAX_EMOTIONS_TO_SHOW]
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
            logger.error(f"Reddit API error: {str(e)} for URL: {post_url}", exc_info=True)
            return f"Reddit API error: {str(e)}", "Could not retrieve post data", {}
        except Exception as e:
            logger.error(f"Error analyzing post: {str(e)} for URL: {post_url}", exc_info=True)
            return f"Error analyzing post: {str(e)}", "Analysis failed", {}

async def chunk_message(text: str, max_length: int = None) -> List[str]:
    """Split a message into chunks that fit Discord's message size limit"""
    if max_length is None:
        max_length = DISCORD_MESSAGE_MAX_LENGTH
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

@bot.tree.command(name='help', description='Show all available commands')
async def help_command(interaction: discord.Interaction):
    """Send a message with all available commands."""
    help_message = (
        "✨ **REDDIT ANALYZER BOT COMMANDS** ✨\n\n"
        "❓ `/help` - Show this magical list of commands\n"
        "📊 `/usage` - Check how much you've been using the bot\n"
        "🔍 `/whoami` - Discover your Discord user ID\n"
        "🧠 `/model` - Switch between AI brains (haiku3/haiku35/sonnet4)\n"
        "💾 `/cache` - Peek at the cache statistics\n"
        "🧹 `/clearcache` - Sweep the cache clean\n\n"
        "✉️ Simply paste any Reddit post URL to get your analysis!\n\n"
        f"📋 **Current Settings:**\n"
        f"• Daily limit: {MAX_REQUESTS_PER_DAY} requests\n"
        f"• Rate limit: {RATE_LIMIT_SECONDS} seconds\n"
        f"• Default model: {DEFAULT_AI_MODEL}\n"
        f"• Max comments: {MAX_COMMENTS_TO_ANALYZE}"
    )
    await interaction.response.send_message(help_message)

@bot.tree.command(name='model', description='Switch between AI models')
@discord.app_commands.describe(model='Choose AI model: haiku3, haiku35, or sonnet4')
@discord.app_commands.choices(model=[
    discord.app_commands.Choice(name='Claude 3 Haiku (fastest, cheapest)', value='haiku3'),
    discord.app_commands.Choice(name='Claude 3.5 Haiku (balanced)', value='haiku35'),
    discord.app_commands.Choice(name='Claude Sonnet 4 (highest quality)', value='sonnet4')
])
async def set_model(interaction: discord.Interaction, model: discord.app_commands.Choice[str]):
    """Change the AI model used for analysis."""
    model_choice = model.value
    
    # The analyzer is now part of the global reddit_bot
    if model_choice in reddit_bot.analyzer.models:
        if not hasattr(bot, 'user_models'):
            bot.user_models = {}
        
        bot.user_models[interaction.user.id] = model_choice
        
        await interaction.response.send_message(
            f"✅ Model set to '{model.name}' for your future requests.\n"
            f"Full model name: {reddit_bot.analyzer.get_model_name(model_choice)}"
        )
    else:
        await interaction.response.send_message(
            "❌ Invalid model. Please choose from the available options."
        )

@bot.event
async def on_message(message):
    """Handle incoming messages"""
    # Ignore messages from all bots
    if message.author == bot.user or message.author.bot:
        return
        
    # Check if message contains a Reddit URL
    url = message.content.strip()
    valid_domains = ["reddit.com", "redd.it", "www.reddit.com", "old.reddit.com"]
    is_valid = any(domain in url for domain in valid_domains)
    
    if not is_valid:
        return
    
    user_id = message.author.id
    
    # Check rate limits
    if not usage_tracker.can_make_request(user_id):
        today = datetime.now().strftime("%Y-%m-%d")
        daily_count = usage_tracker.daily_usage[f"{user_id}:{today}"]
        
        if daily_count >= usage_tracker.MAX_REQUESTS_PER_DAY:
            await message.channel.send(
                "⚠️ You've reached your daily limit of requests. "
                "Please try again tomorrow."
            )
        else:
            await message.channel.send(
                f"⚠️ Rate limit exceeded. Please wait at least {RATE_LIMIT_SECONDS} seconds between requests."
            )
        return

    # Record this request
    usage_tracker.record_request(user_id, "analyze_url")
    
    await message.channel.send("Analyzing post... Please wait.")
    
    # Determine which model to use for this request
    model_choice = DEFAULT_AI_MODEL  # Default model
    if hasattr(bot, 'user_models') and user_id in bot.user_models:
        model_choice = bot.user_models[user_id]
        logger.info(f"Using preferred model '{model_choice}' for user {user_id}")
    
    post_summary, comments_summary, sentiment_stats = await reddit_bot.analyze_reddit_post(url, model_key=model_choice)
    
    if post_summary.startswith("Error") or post_summary.startswith("Invalid") or post_summary.startswith("Reddit API error"):
        await message.channel.send(f"❌ {post_summary}\n{comments_summary}")
        return

    # Format response with fun, engaging headings
    response = f"🔍 **THE REDDIT RUNDOWN** 🔍\n\n"
    response += f"📌 **TLDR: POST IN A NUTSHELL** 📌\n{post_summary}\n\n\n"
    response += f"💬 **WHAT THE CROWD IS SAYING** 💬\n{comments_summary}\n\n\n"
    response += f"🎭 **MOOD METER: VIBES CHECK** 🎭\n"
    
    # Only show percentages if we have comment data
    if sentiment_stats.get('sentiment_breakdown', {}).get('positive', 0) > 0 or \
       sentiment_stats.get('sentiment_breakdown', {}).get('negative', 0) > 0:
        
        breakdown = sentiment_stats['sentiment_breakdown']
        response += f"👥 Comment Mood Breakdown:\n"
        response += f"😊 Positive: {breakdown['positive']:.1f}%\n"
        response += f"😔 Negative: {breakdown['negative']:.1f}%\n"
        response += f"😐 Neutral: {breakdown['neutral']:.1f}%\n\n"
        
        if sentiment_stats.get('top_emotions'):
            response += f"✨ Top Emotions in the Thread:\n"
            for emotion in sentiment_stats['top_emotions']:
                # Add emoji based on emotion
                emoji = "🤔"  # default
                emotion_name = emotion['emotion'].lower()
                
                if any(word in emotion_name for word in ["happy", "joy", "excite"]):
                    emoji = "😄"
                elif any(word in emotion_name for word in ["sad", "disappoint"]):
                    emoji = "😢"
                elif any(word in emotion_name for word in ["angry", "frustrat", "annoy"]):
                    emoji = "😠"
                elif any(word in emotion_name for word in ["surprise"]):
                    emoji = "😲"
                elif any(word in emotion_name for word in ["fear", "worry", "anxious"]):
                    emoji = "😨"
                elif any(word in emotion_name for word in ["curious"]):
                    emoji = "🧐"
                
                response += f"{emoji} {emotion['emotion'].capitalize()}: {emotion['percent']:.1f}%\n"
            
        # Create a visual intensity meter
        intensity = sentiment_stats['avg_intensity']
        intensity_meter = "▓" * int(intensity) + "░" * (SENTIMENT_INTENSITY_MAX - int(intensity))
        response += f"\n🔥 Intensity Meter: {intensity_meter} ({intensity:.1f}/{SENTIMENT_INTENSITY_MAX})"
    else:
        response += "👻 No comments found - it's quiet in here!"

    # Add a fun footer
    response += "\n\n✨ Analysis powered by AI magic ✨"

    # Split response into chunks and send multiple messages if needed
    chunks = await chunk_message(response)
    for chunk in chunks:
        await message.channel.send(chunk)

@bot.tree.command(name='whoami', description='Get your Discord user ID')
async def whoami(interaction: discord.Interaction):
    """Command to check user ID"""
    await interaction.response.send_message(f"Your Discord User ID is: {interaction.user.id}")

@bot.tree.command(name='usage', description='Check your daily usage statistics')
async def usage(interaction: discord.Interaction):
    """Command to check usage statistics"""
    user_id = interaction.user.id
    stats = usage_tracker.get_usage_stats(user_id)
    await interaction.response.send_message(
        f"📊 Usage Statistics\n\n{stats}\n\n"
        f"📋 **Limits:**\n"
        f"• Daily limit: {MAX_REQUESTS_PER_DAY} requests\n"
        f"• Rate limit: {RATE_LIMIT_SECONDS} seconds between requests"
    )

@bot.tree.command(name='cache', description='View cache statistics')
async def cache_stats(interaction: discord.Interaction):
    """Command to check cache statistics"""
    stats = result_cache.get_stats()
    stats_message = (
        f"📊 Cache Statistics\n\n"
        f"Size: {stats['size']}/{stats['max_size']} entries\n"
        f"TTL: {stats['ttl_hours']} hours\n"
        f"Max size: {CACHE_MAX_SIZE} entries\n"
        f"TTL setting: {CACHE_TTL_HOURS} hours"
    )
    await interaction.response.send_message(stats_message)

@bot.tree.command(name='clearcache', description='Clear the analysis cache')
async def clear_cache(interaction: discord.Interaction):
    """Command to clear the cache"""
    result_cache.clear()
    await interaction.response.send_message("✅ Cache cleared successfully")

@bot.event
async def on_ready():
    """Called when the bot is ready"""
    logger.info(f"Bot logged in as {bot.user} (ID: {bot.user.id})")
    try:
        # Sync slash commands
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} slash commands")
        print(f"Bot is ready! Synced {len(synced)} slash commands.")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")
        print(f"Failed to sync commands: {e}")

# Initialize the main bot logic handler as a singleton
reddit_bot = RedditBot()

async def main():
    """Start the bot."""
    required_vars = ['DISCORD_TOKEN', 'REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'ANTHROPIC_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        return
    
    # Log configuration on startup
    logger.info("Bot configuration:")
    logger.info(f"  Max requests per day: {MAX_REQUESTS_PER_DAY}")
    logger.info(f"  Rate limit seconds: {RATE_LIMIT_SECONDS}")
    logger.info(f"  Default AI model: {DEFAULT_AI_MODEL}")
    logger.info(f"  Cache max size: {CACHE_MAX_SIZE}")
    logger.info(f"  Cache TTL hours: {CACHE_TTL_HOURS}")
    logger.info(f"  Max comments to analyze: {MAX_COMMENTS_TO_ANALYZE}")
    logger.info(f"  Max tokens per request: {MAX_TOKENS_PER_REQUEST}")

    logger.info("Starting Discord bot...")
    
    try:
        # Start the Bot
        logger.info("Bot starting, connecting to Discord...")
        await bot.start(DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up resources
        logger.info("Cleaning up...")
        await reddit_bot.cleanup()
        if not bot.is_closed():
            await bot.close()
        logger.info("Shutdown complete")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully without displaying a traceback
        logger.info("KeyboardInterrupt received in outer loop - exiting gracefully.")