from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import librosa
import numpy as np
import joblib
import os
import soundfile as sf
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from datetime import datetime
from dotenv import load_dotenv
import requests
import json
from werkzeug.utils import secure_filename
import logging
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import shutil
from pymongo import MongoClient
from bson import json_util
from urllib.parse import urlencode

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate required environment variables
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
if not GOOGLE_MAPS_API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in .env file")

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
RECIPIENT_PHONE_NUMBERS = [num.strip() for num in os.getenv('RECIPIENT_PHONE_NUMBERS', '').split(',') if num.strip()]
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', 465))
EMAIL_USERNAME = os.getenv('EMAIL_USERNAME')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_RECIPIENTS = [email.strip() for email in os.getenv('EMAIL_RECIPIENTS', '').split(',') if email.strip()]
EMERGENCY_NUMBER = os.getenv('EMERGENCY_NUMBER', '100')
MIN_SCREAMS_FOR_ALERT = int(os.getenv('MIN_SCREAMS_FOR_ALERT', 2))
SIMULATE_CALLS = os.getenv('SIMULATE_CALLS', 'True').lower() == 'true'
IPINFO_TOKEN = os.getenv('IPINFO_TOKEN', '198ea98632a590')
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')

# Load ML model
try:
    model = joblib.load('scream_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    logger.error(f"Failed to load model or scaler: {str(e)}")
    raise

# State
scream_detections = []
UPLOADS_DIR = 'Uploads'

# Initialize MongoDB client
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['scream_detection_db']
scream_collection = db['scream_detections']

# Initialize uploads directory
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Clean up uploads directory on startup
shutil.rmtree(UPLOADS_DIR, ignore_errors=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

def extract_features(audio_path):
    try:
        # Load audio file using librosa
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if len(y) == 0:
            raise ValueError("Audio file is empty or corrupted")
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=27)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        features = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(zcr.T, axis=0),
            np.mean(rms.T, axis=0)
        ])
        return features
    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        raise

def get_browser_location_data(lat, lng, accuracy=None):
    try:
        # Validate latitude and longitude
        if lat is None or lng is None:
            raise ValueError("Latitude or longitude is None")
        lat = float(lat)
        lng = float(lng)
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            raise ValueError("Invalid latitude or longitude values")

        # Reverse geocoding with Google Maps API
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={GOOGLE_MAPS_API_KEY}"
        response = requests.get(geocode_url, timeout=5)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        
        address = data['results'][0]['formatted_address'] if data.get('results') else "Location detected"
        
        # Fallback to Nominatim if Google Maps fails
        if not address or address == "Location detected":
            try:
                geolocator = Nominatim(user_agent="scream_detection_system")
                location = geolocator.reverse(f"{lat}, {lng}", exactly_one=True, timeout=5)
                address = location.address if location else "Location detected"
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                logger.warning(f"Nominatim geocoding failed: {str(e)}")
                address = "Location detected"
        
        timestamp = int(time.time())
        
        # Static Map URL
        static_map_params = {
            "center": f"{lat},{lng}",
            "zoom": 17,
            "size": "600x300",
            "maptype": "roadmap",
            "markers": f"color:red|{lat},{lng}",
            "key": GOOGLE_MAPS_API_KEY
        }
        static_map_url = f"https://maps.googleapis.com/maps/api/staticmap?{urlencode(static_map_params)}"

        # Interactive Map URL (Embed API)
        embed_params = {
            "key": GOOGLE_MAPS_API_KEY,
            "center": f"{lat},{lng}",
            "zoom": 18
        }
        embed_url = f"https://www.google.com/maps/embed/v1/view?{urlencode(embed_params)}"

        return {
            'coordinates': f"{lat:.8f},{lng:.8f}",
            'address': address,
            'accuracy': accuracy or 50,
            'maps_url': f"https://www.google.com/maps?q={lat},{lng}",
            'static_map_url': static_map_url,
            'embed_url': embed_url,
            'source': 'browser_geolocation',
            'latitude': lat,
            'longitude': lng,
            'timestamp': timestamp
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Google Maps API error: {str(e)}")
        # Fallback with default values
        return {
            'coordinates': "0,0",
            'address': "Location unavailable due to geocoding error",
            'accuracy': accuracy or 50,
            'maps_url': "",
            'static_map_url': "",
            'embed_url': "",
            'source': 'error',
            'latitude': 0,
            'longitude': 0,
            'timestamp': int(time.time())
        }
    except Exception as e:
        logger.error(f"Geocoding error: {str(e)}")
        return {
            'coordinates': "0,0",
            'address': "Location unavailable due to geocoding error",
            'accuracy': accuracy or 50,
            'maps_url': "",
            'static_map_url': "",
            'embed_url': "",
            'source': 'error',
            'latitude': 0,
            'longitude': 0,
            'timestamp': int(time.time())
        }

@app.route('/get-browser-location', methods=['POST'])
def handle_browser_location():
    try:
        data = request.json
        if not data or 'lat' not in data or 'lng' not in data:
            return jsonify({'success': False, 'error': 'Invalid location data'}), 400

        location = get_browser_location_data(
            float(data['lat']),
            float(data['lng']),
            float(data.get('accuracy', 50))
        )
        return jsonify({'success': True, 'location': location})
    except Exception as e:
        logger.error(f"Browser location error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/update-location', methods=['POST'])
def update_location():
    try:
        data = request.json
        if not data or 'lat' not in data or 'lng' not in data:
            return jsonify({'success': False, 'error': 'Invalid location data'}), 400

        location = get_browser_location_data(
            float(data['lat']),
            float(data['lng']),
            float(data.get('accuracy', 50))
        )
        return jsonify({'success': True, 'location': location})
    except Exception as e:
        logger.error(f"Location update error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def get_ip_based_location():
    try:
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        if client_ip == '127.0.0.1':
            client_ip = requests.get('https://api.ipify.org', timeout=5).text
        
        response = requests.get(f'https://ipinfo.io/{client_ip}?token={IPINFO_TOKEN}', timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if 'loc' in data:
            lat, lng = map(float, data['loc'].split(','))
            address = f"{data.get('city', 'Unknown')}, {data.get('region', 'Unknown')}, {data.get('country', 'Unknown')}"
            
            return {
                'coordinates': f"{lat:.8f},{lng:.8f}",
                'address': address,
                'maps_url': f'https://www.google.com/maps?q={lat},{lng}',
                'static_map_url': f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom=15&size=600x300&maptype=roadmap&markers=color:red%7C{lat},{lng}&key={GOOGLE_MAPS_API_KEY}",
                'source': 'ipinfo.io',
                'latitude': lat,
                'longitude': lng,
                'accuracy': 1000  # Lower accuracy for IP-based
            }
    except Exception as e:
        logger.warning(f"IP location error: {str(e)}")
    return None

def get_current_location():
    try:
        if request.method == 'POST' and request.form and 'location' in request.form:
            loc_data = json.loads(request.form['location'])
            return get_browser_location_data(
                loc_data['lat'],
                loc_data['lng'],
                loc_data.get('accuracy')
            )
        
        ip_location = get_ip_based_location()
        if ip_location:
            return ip_location

        return {
            'coordinates': '0,0',
            'address': "Enable browser geolocation for accurate results",
            'maps_url': '',
            'static_map_url': '',
            'embed_url': '',
            'source': 'none',
            'latitude': 0,
            'longitude': 0,
            'accuracy': 0,
            'timestamp': int(time.time())
        }
    except Exception as e:
        logger.error(f"Location error: {str(e)}")
        return {
            'coordinates': '0,0',
            'address': "Location service error",
            'maps_url': '',
            'static_map_url': '',
            'embed_url': '',
            'source': 'error',
            'latitude': 0,
            'longitude': 0,
            'accuracy': 0,
            'timestamp': int(time.time())
        }

def send_sms_alert(location):
    try:
        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
            raise ValueError("Twilio credentials not configured properly")
        if not RECIPIENT_PHONE_NUMBERS:
            raise ValueError("No recipient phone numbers configured")

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        message_body = (
            f"Scream detected\n"
            f"Coordinates: {location.get('coordinates', 'N/A')}\n"
            f"Location: {location.get('address', 'Unknown')}\n"
            f"Map: {location.get('maps_url', 'N/A')}"
        )

        success_count = 0
        for recipient in RECIPIENT_PHONE_NUMBERS:
            try:
                message = client.messages.create(
                    body=message_body,
                    from_=TWILIO_PHONE_NUMBER,
                    to=recipient
                )
                logger.info(f"SMS sent to {recipient}: {message.sid}")
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to send SMS to {recipient}: {str(e)}")
                continue

        return success_count > 0
    except Exception as e:
        logger.error(f"SMS Error: {str(e)}")
        return False

def send_email_alert(location):
    try:
        if not all([EMAIL_HOST, EMAIL_PORT, EMAIL_USERNAME, EMAIL_PASSWORD]):
            raise ValueError("Email not configured properly")
        if not EMAIL_RECIPIENTS:
            raise ValueError("No email recipients configured")

        msg = MIMEMultipart()
        msg['From'] = EMAIL_USERNAME
        msg['To'] = ', '.join(EMAIL_RECIPIENTS)
        msg['Subject'] = "🚨 EMERGENCY: Scream Detected"
        
        body = f"""
        <html>
            <body>
                <h2 style="color: red;">SCREAM DETECTION ALERT</h2>
                
                <p><strong>Location:</strong> {location.get('address', 'Unknown location')}</p>
                <p><strong>Coordinates:</strong> {location.get('coordinates', 'N/A')}</p>
                <p><strong>Accuracy:</strong> {location.get('accuracy', 'Unknown')} meters</p>
                
                <h3>Quick Links:</h3>
                <ul>
                    <li><a href="https://maps.google.com/?q={location.get('coordinates', '0,0')}">Open in Google Maps</a></li>
                    <li><a href="https://www.google.com/maps/dir/?api=1&destination={location.get('coordinates', '0,0')}">Get Directions</a></li>
                </ul>
                
                <p><strong>Detection Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <p>This is an automated alert from the Scream Detection System.</p>
                
                <p style="font-size: small;">
                    <em>Source: {location.get('source', 'Unknown')}</em>
                </p>
            </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT) as server:
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(
                EMAIL_USERNAME,
                EMAIL_RECIPIENTS,
                msg.as_string()
            )
        logger.info("Email alert sent successfully")
        return True
    except Exception as e:
        logger.error(f"Email Error: {str(e)}")
        return False

def initiate_emergency_call(location):
    try:
        if SIMULATE_CALLS:
            call_details = (
                f"SIMULATED CALL TO {EMERGENCY_NUMBER}\n"
                f"Location: {location.get('address', 'Unknown')}\n"
                f"Coordinates: {location.get('coordinates', 'N/A')}\n"
                f"Accuracy: {location.get('accuracy', 'Unknown')}m\n"
                f"Time: {datetime.now().isoformat()}"
            )
            logger.info(call_details)
            return True
        return False
    except Exception as e:
        logger.error(f"Call Error: {str(e)}")
        return False

def check_emergency_required():
    recent_screams = [s for s in scream_detections if time.time() - s < 30]
    if len(recent_screams) >= MIN_SCREAMS_FOR_ALERT:
        return {
            'requires_approval': True,
            'emergency_number': EMERGENCY_NUMBER,
            'location': get_current_location()
        }
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        audio_path = os.path.join(UPLOADS_DIR, f"upload_{timestamp}_{filename}")
        file.save(audio_path)

        features = extract_features(audio_path)
        features_scaled = scaler.transform([features])

        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)

        location = get_current_location()
        if 'location' in request.form:
            loc_data = json.loads(request.form['location'])
            location = get_browser_location_data(
                loc_data['lat'],
                loc_data['lng'],
                loc_data.get('accuracy', 50)
            )

        # Store in MongoDB
        detection_document = {
            'timestamp': datetime.utcnow(),
            'prediction': "Screaming Detected" if prediction[0] == 1 else "No Screaming Detected",
            'probability': float(probabilities[0][1]),
            'location': {
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'accuracy': location['accuracy'],
                'address': location['address']
            },
            'audio_path': audio_path
        }
        scream_collection.insert_one(detection_document)

        result = {
            'prediction': "Screaming Detected" if prediction[0] == 1 else "No Screaming Detected",
            'probability': float(probabilities[0][1]),
            'probabilities': probabilities.tolist(),
            'location': location
        }

        if prediction[0] == 1:
            scream_detections.append(time.time())
            result['action_required'] = check_emergency_required()

        return jsonify(result)
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the uploaded file
        if 'audio_path' in locals() and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {audio_path}: {str(e)}")

@app.route('/send-sms-alert', methods=['POST'])
def handle_sms_alert():
    try:
        location = request.json.get('location')
        if not location:
            raise ValueError("Location data missing")
        
        if send_sms_alert(location):
            return jsonify({'status': 'success', 'message': 'SMS alert sent'})
        return jsonify({'status': 'error', 'message': 'Failed to send SMS'}), 500
    except Exception as e:
        logger.error(f"SMS handler error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/send-email-alert', methods=['POST'])
def handle_email_alert():
    try:
        location = request.json.get('location')
        if not location:
            raise ValueError("Location data missing")
            
        if send_email_alert(location):
            return jsonify({'status': 'success', 'message': 'Email alert sent'})
        return jsonify({'status': 'error', 'message': 'Failed to send email'}), 500
    except Exception as e:
        logger.error(f"Email handler error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/initiate-emergency-call', methods=['POST'])
def handle_emergency_call():
    try:
        location = request.json.get('location')
        if not location:
            raise ValueError("Location data missing")
            
        if initiate_emergency_call(location):
            return jsonify({'status': 'success', 'message': f'Emergency call to {EMERGENCY_NUMBER} initiated'})
        return jsonify({'status': 'error', 'message': 'Failed to initiate call'}), 500
    except Exception as e:
        logger.error(f"Call handler error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/realtime', methods=['POST'])
def realtime_detection():
    if 'audio' not in request.files:
        logger.error("No audio file received in request")
        return jsonify({'error': 'No audio file received'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        logger.error("No selected file in request")
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Create unique filename with timestamp
        timestamp = int(time.time())
        filename = f"realtime_{timestamp}.wav"
        audio_path = os.path.join(UPLOADS_DIR, filename)
        
        # Save the file
        audio_file.save(audio_path)
        
        # Verify file exists and has content
        if not os.path.exists(audio_path):
            logger.error(f"File not found after saving: {audio_path}")
            raise FileNotFoundError(f"File not found after saving: {audio_path}")
        
        file_size = os.path.getsize(audio_path)
        logger.info(f"Saved audio file size: {file_size} bytes")
        if file_size == 0:
            raise ValueError("Uploaded audio file is empty")
        
        # Process audio
        features = extract_features(audio_path)
        features_scaled = scaler.transform([features])
        
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        # Get location
        location = get_current_location()
        if 'location' in request.form:
            loc_data = json.loads(request.form['location'])
            location = get_browser_location_data(
                loc_data['lat'],
                loc_data['lng'],
                loc_data.get('accuracy', 50)
            )
        
        # Store in MongoDB
        detection_document = {
            'timestamp': datetime.utcnow(),
            'prediction': "Screaming Detected" if prediction[0] == 1 else "No Screaming Detected",
            'probability': float(probabilities[0][1]),
            'location': {
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'accuracy': location['accuracy'],
                'address': location['address']
            },
            'audio_path': audio_path
        }
        scream_collection.insert_one(detection_document)
        
        result = {
            'prediction': "Screaming Detected" if prediction[0] == 1 else "No Screaming Detected",
            'probability': float(probabilities[0][1]),
            'probabilities': probabilities.tolist(),
            'location': location
        }
        
        if prediction[0] == 1:
            scream_detections.append(time.time())
            result['action_required'] = check_emergency_required()
        
        logger.info("Audio processed successfully")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Realtime detection error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the uploaded file
        if 'audio_path' in locals() and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {audio_path}: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)