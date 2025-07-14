import os
import io
from io import BytesIO
import base64
import requests
import json
from PIL import Image
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import asyncio
from datetime import datetime
from flask import Flask, request, jsonify
import threading

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GOOGLE_VISION_API_KEY = os.getenv('GOOGLE_VISION_API_KEY')
SERP_API_KEY = os.getenv('SERP_API_KEY', None)  # Optional for enhanced reverse search

class GoogleVisionAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.vision_api_url = 'https://vision.googleapis.com/v1/images:annotate'

    def analyze_image(self, image_data):
        """
        Comprehensive image analysis using Google Cloud Vision API
        """
        try:
            # Convert image to base64
            if isinstance(image_data, bytes):
                image_content = base64.b64encode(image_data).decode('utf-8')
            elif isinstance(image_data, Image.Image):
                buffered = BytesIO()
                image_data.save(buffered, format="JPEG")
                image_content = base64.b64encode(buffered.getvalue()).decode('utf-8')
            else:
                raise ValueError("Unsupported image data type")

            # Comprehensive analysis request
            request_data = {
                "requests": [
                    {
                        "image": {
                            "content": image_content
                        },
                        "features": [
                            {
                                "type": "FACE_DETECTION",
                                "maxResults": 10
                            },
                            {
                                "type": "LABEL_DETECTION",
                                "maxResults": 15
                            },
                            {
                                "type": "SAFE_SEARCH_DETECTION"
                            },
                            {
                                "type": "WEB_DETECTION"
                            },
                            {
                                "type": "OBJECT_LOCALIZATION",
                                "maxResults": 10
                            },
                            {
                                "type": "IMAGE_PROPERTIES"
                            },
                            {
                                "type": "TEXT_DETECTION"
                            }
                        ]
                    }
                ]
            }

            # Make the API request
            response = requests.post(
                f"{self.vision_api_url}?key={self.api_key}",
                json=request_data
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return None

    def get_comprehensive_analysis(self, image_data):
        """
        Get comprehensive image analysis including authenticity indicators
        """
        try:
            result = self.analyze_image(image_data)
            if not result or 'responses' not in result or not result['responses']:
                return None
                
            response = result['responses'][0]
            
            # Face analysis
            face_annotations = response.get('faceAnnotations', [])
            face_analysis = {
                'faces_detected': len(face_annotations),
                'face_details': []
            }
            
            # Analyze each face for authenticity indicators
            for face in face_annotations:
                face_detail = {
                    'detection_confidence': face.get('detectionConfidence', 0),
                    'joy_likelihood': face.get('joyLikelihood', 'UNKNOWN'),
                    'sorrow_likelihood': face.get('sorrowLikelihood', 'UNKNOWN'),
                    'anger_likelihood': face.get('angerLikelihood', 'UNKNOWN'),
                    'surprise_likelihood': face.get('surpriseLikelihood', 'UNKNOWN'),
                    'under_exposed_likelihood': face.get('underExposedLikelihood', 'UNKNOWN'),
                    'blurred_likelihood': face.get('blurredLikelihood', 'UNKNOWN'),
                    'headwear_likelihood': face.get('headwearLikelihood', 'UNKNOWN'),
                    'landmarks': len(face.get('landmarks', []))
                }
                face_analysis['face_details'].append(face_detail)
            
            # Labels and objects
            labels = response.get('labelAnnotations', [])
            objects = response.get('localizedObjectAnnotations', [])
            
            # Safe search
            safe_search = response.get('safeSearchAnnotation', {})
            
            # Web detection for reverse image search
            web_detection = response.get('webDetection', {})
            
            # Image properties
            image_properties = response.get('imagePropertiesAnnotation', {})
            
            # Text detection
            text_annotations = response.get('textAnnotations', [])
            
            return {
                'face_analysis': face_analysis,
                'labels': [{'description': label['description'], 'confidence': label['score']} 
                          for label in labels],
                'objects': [{'name': obj['name'], 'confidence': obj['score']} 
                           for obj in objects],
                'safe_search': safe_search,
                'web_detection': web_detection,
                'image_properties': image_properties,
                'text_detected': len(text_annotations) > 0,
                'text_content': text_annotations[0]['description'] if text_annotations else None
            }
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {str(e)}")
            return None

class AdvancedAnalyzer:
    def __init__(self):
        pass
    
    def analyze_image_authenticity(self, vision_analysis):
        """
        Advanced authenticity analysis using only Google Vision API data
        """
        try:
            indicators = []
            red_flags = []
            authenticity_score = 50  # Start neutral
            
            # Face analysis
            face_analysis = vision_analysis['face_analysis']
            if face_analysis['faces_detected'] > 0:
                for face in face_analysis['face_details']:
                    # Detection confidence analysis
                    conf = face['detection_confidence']
                    if conf > 0.9:
                        authenticity_score += 15
                        indicators.append(f"High face detection confidence ({conf*100:.1f}%)")
                    elif conf < 0.6:
                        authenticity_score -= 20
                        red_flags.append(f"Low face detection confidence ({conf*100:.1f}%)")
                    
                    # Blur analysis
                    if face['blurred_likelihood'] in ['VERY_LIKELY', 'LIKELY']:
                        authenticity_score -= 15
                        red_flags.append("Blurred face detected (potential manipulation)")
                    
                    # Landmark analysis
                    if face['landmarks'] > 15:
                        authenticity_score += 10
                        indicators.append("Good facial landmark detection")
                    elif face['landmarks'] < 8:
                        authenticity_score -= 10
                        red_flags.append("Poor facial landmark detection")
                    
                    # Emotion consistency check
                    emotions = [
                        face['joy_likelihood'],
                        face['sorrow_likelihood'],
                        face['anger_likelihood'],
                        face['surprise_likelihood']
                    ]
                    strong_emotions = [e for e in emotions if e in ['VERY_LIKELY', 'LIKELY']]
                    if len(strong_emotions) > 2:
                        authenticity_score -= 10
                        red_flags.append("Multiple strong emotions detected (unnatural)")
            
            # Safe search analysis
            safe_search = vision_analysis['safe_search']
            if safe_search.get('spoof', 'UNKNOWN') in ['VERY_LIKELY', 'LIKELY']:
                authenticity_score -= 35
                red_flags.append("Spoofing detected by safe search")
            
            # Web detection analysis
            web_detection = vision_analysis['web_detection']
            if 'webEntities' in web_detection and web_detection['webEntities']:
                authenticity_score += 10
                indicators.append("Image found in web databases")
                
                # Check for celebrity or public figure detection
                for entity in web_detection['webEntities']:
                    if entity.get('score', 0) > 0.8:
                        description = entity.get('description', '').lower()
                        if any(keyword in description for keyword in ['person', 'celebrity', 'actor', 'politician']):
                            red_flags.append(f"High-confidence match for public figure: {entity.get('description')}")
            
            # Label analysis for digital artifacts
            suspicious_labels = ['screenshot', 'digital', 'computer', 'monitor', 'screen']
            for label in vision_analysis['labels']:
                if label['description'].lower() in suspicious_labels and label['confidence'] > 0.7:
                    authenticity_score -= 10
                    red_flags.append(f"Digital artifact detected: {label['description']}")
            
            # Image properties analysis
            image_props = vision_analysis.get('image_properties', {})
            if 'dominantColors' in image_props:
                colors = image_props['dominantColors'].get('colors', [])
                if len(colors) < 3:
                    authenticity_score -= 5
                    red_flags.append("Limited color palette (potential processing)")
            
            # Text detection analysis
            if vision_analysis['text_detected']:
                text_content = vision_analysis.get('text_content', '').lower()
                suspicious_text = ['deepfake', 'ai generated', 'synthetic', 'fake']
                if any(word in text_content for word in suspicious_text):
                    authenticity_score -= 25
                    red_flags.append("Suspicious text detected in image")
            
            # Normalize score
            authenticity_score = max(0, min(100, authenticity_score))
            
            # Determine prediction
            if authenticity_score >= 75:
                prediction = "Likely Authentic"
            elif authenticity_score >= 45:
                prediction = "Inconclusive"
            else:
                prediction = "Suspicious"
            
            return {
                'prediction': prediction,
                'confidence': authenticity_score,
                'indicators': indicators,
                'red_flags': red_flags,
                'technical_quality': self._assess_technical_quality(vision_analysis)
            }
            
        except Exception as e:
            print(f"Error in advanced analysis: {str(e)}")
            return {
                'prediction': 'Error',
                'confidence': 0,
                'indicators': [],
                'red_flags': [f"Analysis error: {str(e)}"],
                'technical_quality': 'Unknown'
            }
    
    def _assess_technical_quality(self, vision_analysis):
        """
        Assess technical quality of the image
        """
        quality_score = 0
        quality_factors = []
        
        # Face quality assessment
        if vision_analysis['face_analysis']['faces_detected'] > 0:
            avg_confidence = sum(face['detection_confidence'] 
                               for face in vision_analysis['face_analysis']['face_details']) / len(vision_analysis['face_analysis']['face_details'])
            if avg_confidence > 0.8:
                quality_score += 2
                quality_factors.append("High face detection quality")
            
            # Check for technical issues
            for face in vision_analysis['face_analysis']['face_details']:
                if face['under_exposed_likelihood'] in ['VERY_LIKELY', 'LIKELY']:
                    quality_score -= 1
                    quality_factors.append("Under-exposed faces")
                if face['blurred_likelihood'] in ['VERY_LIKELY', 'LIKELY']:
                    quality_score -= 1
                    quality_factors.append("Blurred faces")
        
        # Label confidence assessment
        high_conf_labels = [l for l in vision_analysis['labels'] if l['confidence'] > 0.8]
        if len(high_conf_labels) > 5:
            quality_score += 1
            quality_factors.append("Clear content recognition")
        
        # Determine overall quality
        if quality_score >= 2:
            return "High Quality"
        elif quality_score >= 0:
            return "Medium Quality"
        else:
            return "Low Quality"

class ImageAnalysisBot:
    def __init__(self):
        self.vision_api = GoogleVisionAPI(GOOGLE_VISION_API_KEY) if GOOGLE_VISION_API_KEY else None
        self.advanced_analyzer = AdvancedAnalyzer()
        
    def calculate_authenticity_score(self, vision_analysis):
        """
        Calculate authenticity score based on various indicators
        """
        score = 50  # Start with neutral score
        indicators = []
        
        # Face analysis indicators
        if vision_analysis['face_analysis']['faces_detected'] > 0:
            for face in vision_analysis['face_analysis']['face_details']:
                # High detection confidence is good
                if face['detection_confidence'] > 0.8:
                    score += 10
                    indicators.append("High face detection confidence")
                elif face['detection_confidence'] < 0.5:
                    score -= 15
                    indicators.append("Low face detection confidence")
                
                # Blur detection
                if face['blurred_likelihood'] in ['VERY_LIKELY', 'LIKELY']:
                    score -= 20
                    indicators.append("Face appears blurred")
                
                # Landmark detection
                if face['landmarks'] > 15:
                    score += 5
                    indicators.append("Good facial landmark detection")
                elif face['landmarks'] < 5:
                    score -= 10
                    indicators.append("Poor facial landmark detection")
        
        # Web detection indicators
        web_detection = vision_analysis['web_detection']
        if 'webEntities' in web_detection and web_detection['webEntities']:
            score += 5
            indicators.append("Image found in web databases")
        
        if 'fullMatchingImages' in web_detection and web_detection['fullMatchingImages']:
            score += 10
            indicators.append("Exact matches found online")
        
        # Safe search indicators
        safe_search = vision_analysis['safe_search']
        if safe_search.get('spoof', 'UNKNOWN') in ['VERY_LIKELY', 'LIKELY']:
            score -= 30
            indicators.append("Potential spoofing detected")
        
        # Normalize score
        score = max(0, min(100, score))
        
        return score, indicators

    async def get_image_sources(self, vision_analysis, image=None):
        """
        Extract image sources from web detection. If none found, try SerpAPI if available. If still none, show visually similar images as fallback.
        """
        web_detection = vision_analysis['web_detection']
        sources = []
        used = 'Google Vision API'
        fallback_type = None
        # Debug: print web_detection for troubleshooting
        print("[DEBUG] web_detection:", json.dumps(web_detection, indent=2))
        # Extract page URLs
        if 'pagesWithMatchingImages' in web_detection:
            for page in web_detection['pagesWithMatchingImages']:
                url = page.get('url')
                if url:
                    sources.append({
                        'url': url,
                        'type': 'page',
                        'title': page.get('pageTitle', 'No title')
                    })
        # Extract full matching image URLs
        if 'fullMatchingImages' in web_detection:
            for match in web_detection['fullMatchingImages']:
                url = match.get('url')
                if url:
                    sources.append({
                        'url': url,
                        'type': 'image',
                        'title': 'Matching Image'
                    })
        # Add web entities for context
        web_entities = []
        if 'webEntities' in web_detection:
            for entity in web_detection['webEntities']:
                if 'description' in entity and entity.get('score', 0) > 0.5:
                    web_entities.append({
                        'name': entity['description'],
                        'score': entity.get('score', 0)
                    })
        # If no sources found, try SerpAPI if available and image is provided
        if not sources and SERP_API_KEY and image is not None:
            serp_sources = await self.get_serpapi_sources(image)
            if serp_sources:
                sources = serp_sources
                used = 'SerpAPI'
        # If still no sources, use visually similar images as fallback
        if not sources and 'visuallySimilarImages' in web_detection:
            for match in web_detection['visuallySimilarImages']:
                url = match.get('url')
                if url:
                    sources.append({
                        'url': url,
                        'type': 'visually_similar',
                        'title': 'Visually Similar Image'
                    })
            if sources:
                used = 'Visually Similar Images (Fallback)'
                fallback_type = 'visually_similar'
        return {
            'sources': sources[:10],
            'web_entities': web_entities[:5],
            'source_type': used,
            'fallback_type': fallback_type
        }

    async def get_serpapi_sources(self, image):
        """
        Use SerpAPI to find image sources if Google Vision API finds none.
        """
        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            url = f"https://serpapi.com/search.json?engine=google_lens&api_key={SERP_API_KEY}"
            payload = {"image": img_str}
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                sources = []
                # Extract from visual matches
                if "visual_matches" in data:
                    for match in data["visual_matches"]:
                        if "link" in match:
                            sources.append({
                                'url': match["link"],
                                'type': 'serpapi',
                                'title': match.get('title', 'SerpAPI Match')
                            })
                # Extract from knowledge graph if available
                if "knowledge_graph" in data and "source" in data["knowledge_graph"]:
                    sources.append({
                        'url': data["knowledge_graph"]["source"],
                        'type': 'serpapi',
                        'title': data["knowledge_graph"].get('title', 'SerpAPI Knowledge')
                    })
                return sources[:10]
            else:
                print(f"[DEBUG] SerpAPI error: {response.status_code} {response.text}")
                return []
        except Exception as e:
            print(f"[DEBUG] SerpAPI exception: {str(e)}")
            return []

# Initialize the bot
image_bot = ImageAnalysisBot()

# Flask app for webhook
app = Flask(__name__)

# Telegram bot and application
bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
application = None
loop = None  # Global event loop for async tasks

def run_asyncio_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

@app.route('/set_webhook', methods=['GET'])
def set_webhook():
    if not TELEGRAM_TOKEN:
        return 'TELEGRAM_TOKEN not set', 500
    if not os.environ.get('RENDER_EXTERNAL_URL') and not os.environ.get('WEBHOOK_URL'):
        return 'Set RENDER_EXTERNAL_URL or WEBHOOK_URL env variable', 500
    webhook_url = os.environ.get('WEBHOOK_URL') or os.environ.get('RENDER_EXTERNAL_URL')
    webhook_url = webhook_url.rstrip('/') + f"/webhook/{TELEGRAM_TOKEN}"
    try:
        set_hook = asyncio.run(bot.set_webhook(url=webhook_url))
        if set_hook:
            return f'Webhook set to {webhook_url}', 200
        else:
            return 'Failed to set webhook', 500
    except Exception as e:
        return f'Error: {e}', 500

@app.route(f'/webhook/{TELEGRAM_TOKEN}', methods=['POST'])
def webhook():
    if request.method == "POST":
        update = Update.de_json(request.get_json(force=True), bot)
        global loop
        asyncio.run_coroutine_threadsafe(application.process_update(update), loop)
        return 'ok', 200
    return 'not allowed', 405

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to Advanced Image Analysis Bot!\n\n"
        "Send me any image and I'll analyze it for:\n"
        "• Authenticity assessment using advanced algorithms\n"
        "• Comprehensive image analysis\n"
        "• Reverse image search\n"
        "• Face and content detection\n"
        "• Technical quality assessment\n\n"
        "✅ FREE - Uses only Google Vision API!\n"
        "Use /help to see available commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Available Commands:\n\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        "Simply send any image to get:\n"
        "• AI-powered authenticity analysis (FREE)\n"
        "• Detailed technical assessment\n"
        "• Source verification\n"
        "• Content analysis\n\n"
        "Powered by Google Vision API (FREE tier)!"
    )

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle images sent by users"""
    try:
        # Check if APIs are available
        if not image_bot.vision_api:
            await update.message.reply_text("❌ Google Vision API is not configured. Please check your API key.")
            return
        
        # Send processing message
        processing_message = await update.message.reply_text("🔄 Analyzing image... Please wait.")
        
        # Get the photo file
        photo = await update.message.photo[-1].get_file()
        
        # Download the photo
        photo_bytes = await photo.download_as_bytearray()
        
        # Create a PIL Image from bytes
        image = Image.open(BytesIO(photo_bytes))
        
        # Update processing message
        await processing_message.edit_text("🔍 Analyzing image...\n🔄 Running comprehensive analysis...")
        
        # Get comprehensive Google Vision analysis
        vision_analysis = image_bot.vision_api.get_comprehensive_analysis(image)
        
        if not vision_analysis:
            await processing_message.edit_text("❌ Failed to analyze image. Please try again.")
            return
        
        # Update processing message
        await processing_message.edit_text("🔍 Analyzing image...\n✅ Vision analysis complete\n🔄 Running advanced authenticity assessment...")
        
        # Get advanced analysis using only Vision API data
        advanced_result = image_bot.advanced_analyzer.analyze_image_authenticity(vision_analysis)
        
        # Calculate authenticity score from advanced analysis
        auth_score = advanced_result['confidence']
        
        # Update processing message
        await processing_message.edit_text("🔍 Analyzing image...\n✅ Vision analysis complete\n✅ Advanced assessment complete\n🔄 Searching for sources...")
        
        # Get image sources (pass image for fallback)
        sources_info = await image_bot.get_image_sources(vision_analysis, image=image)
        
        # Format the response
        response = f"🔍 Advanced Image Analysis Results\n\n"
        
        # Authenticity assessment
        prediction = advanced_result['prediction']
        if prediction == "Likely Authentic":
            emoji = "✅"
            verdict = "Real"
        elif prediction == "Inconclusive":
            emoji = "⚠️"
            verdict = "Uncertain"
        else:
            emoji = "🚨"
            verdict = "Fake"
        
        response += f"{emoji} Authenticity Assessment:\n"
        response += f"• Status: <b>{verdict}</b>\n"
        response += f"• Confidence Score: {auth_score:.1f}%\n"
        response += f"• Technical Quality: {advanced_result['technical_quality']}\n"
        
        # Face analysis
        face_analysis = vision_analysis['face_analysis']
        if face_analysis['faces_detected'] > 0:
            response += f"\n👤 Face Analysis:\n"
            response += f"• Faces Detected: {face_analysis['faces_detected']}\n"
            
            # Face quality indicators
            avg_confidence = sum(face['detection_confidence'] for face in face_analysis['face_details']) / len(face_analysis['face_details'])
            response += f"• Average Detection Confidence: {avg_confidence*100:.1f}%\n"
            
            # Check for quality issues
            quality_issues = []
            for face in face_analysis['face_details']:
                if face['blurred_likelihood'] in ['VERY_LIKELY', 'LIKELY']:
                    quality_issues.append("Blurred face detected")
                if face['under_exposed_likelihood'] in ['VERY_LIKELY', 'LIKELY']:
                    quality_issues.append("Under-exposed face")
            
            if quality_issues:
                response += f"• Quality Issues: {', '.join(quality_issues)}\n"
        
        # Advanced analysis indicators
        if advanced_result['indicators']:
            response += f"\n✅ Positive Indicators:\n"
            for indicator in advanced_result['indicators'][:3]:
                response += f"• {indicator}\n"
        
        # Red flags
        if advanced_result['red_flags']:
            response += f"\n🚩 Red Flags:\n"
            for flag in advanced_result['red_flags'][:3]:
                response += f"• {flag}\n"
        
        # Content analysis
        if vision_analysis['labels']:
            response += f"\n🏷️ Content Analysis:\n"
            for label in vision_analysis['labels'][:3]:
                response += f"• {label['description']} ({label['confidence']*100:.1f}%)\n"
        
        # Objects detected
        if vision_analysis['objects']:
            response += f"\n📦 Objects Detected:\n"
            for obj in vision_analysis['objects'][:3]:
                response += f"• {obj['name']} ({obj['confidence']*100:.1f}%)\n"
        
        # Text detection
        if vision_analysis['text_detected']:
            response += f"\n📝 Text Detected: Yes\n"
        
        # Source verification
        if sources_info['sources']:
            if sources_info.get('fallback_type') == 'visually_similar':
                response += f"\n🖼️ No direct sources found. Showing visually similar images as fallback:\n"
            else:
                response += f"\n🔗 Backtracking Links ({sources_info['source_type']}):\n"
            for source in sources_info['sources'][:3]:
                # Extract domain name for display
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(source['url']).netloc.replace('www.', '')
                except Exception:
                    domain = source['url']
                response += f"• <a href=\"{source['url']}\">{domain}</a>\n"
        elif sources_info['web_entities']:
            response += f"\n🌐 Related Web Entities:\n"
            for entity in sources_info['web_entities'][:3]:
                response += f"• {entity['name']} ({entity['score']*100:.1f}%)\n"
        else:
            response += f"\n🔎 No matching sources found online\n"
        
        # Add timestamp
        response += f"\n🕐 Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send the results
        await processing_message.edit_text(response, parse_mode='HTML')
        
    except Exception as e:
        error_message = f"❌ Error analyzing image: {str(e)}"
        if 'processing_message' in locals():
            await processing_message.edit_text(error_message)
        else:
            await update.message.reply_text(error_message)

def main():
    global application, loop
    # Check if Telegram token is available
    if not TELEGRAM_TOKEN:
        print("❌ Error: TELEGRAM_TOKEN not found in environment variables or .env file.")
        return
    # Check if at least Google Vision API is available
    if not GOOGLE_VISION_API_KEY:
        print("❌ Error: GOOGLE_VISION_API_KEY not found in environment variables or .env file.")
        return
    # Create application
    application = Application.builder().token(TELEGRAM_TOKEN).concurrent_updates(True).build()
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    # Initialize the application (required for webhook mode)
    asyncio.run(application.initialize())
    # Start a background event loop for async tasks
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    # Start Flask app
    port = int(os.environ.get("PORT", 10000))
    print(f"🤖 Starting Advanced Image Analysis Telegram Bot (webhook mode) on port {port}...")
    print(f"Google Vision API: {'✅ Enabled' if image_bot.vision_api else '❌ Disabled'}")
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()