🎯 Scream Detection System


📖 Overview


The Scream Detection System is an AI-powered application that identifies screams in real-time audio or uploaded files and can trigger emergency alerts via SMS or email, including geolocation data to designated contacts. This platform is designed for enhanced safety, enabling rapid and automated notifications upon scream detection.

<br>

✨ Features

1.Real-Time Audio Monitoring: Live scream detection using the microphone.

<br>

2.Audio File Analysis: Analyze uploaded WAV or MP3 files for screams.

<br>

3.Geolocation Integration: Retrieves location details using browser geolocation APIs (with IP address fallbacks). Google Maps API is used for mapping detected incident locations.

<br>

4.Emergency Alerts (SMS & Email):

SMS: Alerts are sent using the Twilio API. A Twilio account and API credentials are required for setup.

Email: Notifications sent via Gmail. Uses Google App Passwords for secure SMTP access to Gmail accounts.

<br>


📦 Required Libraries

Below are the key Python and frontend libraries used in the project:

1.Flask (backend REST API)

<br>

2.librosa (audio feature extraction: MFCC, chroma, ZCR, RMS, etc.)

<br>

3.scikit-learn (model training, RandomForestClassifier)

<br>

4.imbalanced-learn (SMOTE for class balancing)

<br>

5.joblib (model/scaler serialization)

<br>

6.pymongo (MongoDB data storage)

<br>

7.twilio (SMS sending)

<br>

8.smtplib and email (SMTP email sending)

<br>

9.dotenv (environment variable management)

<br>

10.requests (HTTP API calls, e.g., geolocation)

<br>

11.Frontend: Standard HTML, CSS, JavaScript, and Google Maps JavaScript API

<br>


⚙️ How It Works

🖥️ Frontend (index.html):

1.Users can monitor their mic for live scream detection or upload audio files for analysis.

2.On scream detection, your location is retrieved via browser geolocation APIs. IP-based geolocation is used as a fallback if users deny browser location requests.

3.Location and detection results are displayed using the Google Maps JavaScript API.

4.Alert options for SMS and email are enabled immediately after a scream is detected.

<br>

🔙 Backend (app.py):

1.Analyzes audio features with librosa and predicts scream probability using a pre-trained RandomForestClassifier.

2.Handles geolocation lookup using browser geolocation API data or via IP geolocation services as backup.

<br>

3.Alert Trigger Logic:

1.For each audio segment, the backend computes the probability that a scream is present.

2.If this scream probability score exceeds a configurable threshold (e.g., 0.80), the event is classified as a scream.

3.If the threshold is breached, the system:

4.Triggers SMS and/or email alerts to the configured contacts.

5.Logs event details in MongoDB, including the probability, timestamp, and other metadata.

6.If the score does not exceed the threshold, no alert is generated.

7.You may adjust the threshold value to suit your application's sensitivity or reduce false positives as needed.

<br>

4.Sends SMS using the Twilio API (Twilio credentials are required in your .env file).

5.Sends emails via Gmail’s SMTP server, using a Google App Password for secure authentication (not your main Gmail password).

6.Records detection events and metadata in a MongoDB database. MongoDB not only stores each event but retains all data for future analysis, enabling you to perform comprehensive reviews, reporting, and improvements to the system over time.

<br>

🗂️ MongoDB Data Structure and Fields

Each detection event in MongoDB typically includes the following fields and objects for comprehensive analysis and future reference:


1.timestamp: The exact ISO format date and time when the scream was detected or when an event was logged.

<br>

2.audio_file_name / audio_source: Name or identifier of the audio file analyzed, or a flag for live audio.

<br>

3.scream_probability: The probability/confidence score (float, e.g., 0.92) that the given audio segment contains a scream, as predicted by the model.

<br>

4.is_scream: Boolean value (True/False) indicating if the detection crosses the predefined threshold and is officially classified as a scream, which in turn controls whether alerts are generated.

<br>

5.threshold_value: The threshold value currently used for classification (e.g., 0.80). (Added for traceability and reproducibility.)

<br>

6.location:

latitude and longitude: GPS coordinates (floats) of the detected event if browser geolocation is allowed, or IP-based location otherwise.

accuracy: Reported accuracy in meters (if available from browser).

address / place details: (Optional) Reverse-geocoded address, city, or country derived from coordinates.

<br>

7.alert_type: Specifies whether an SMS, email, or both were triggered for this detection.

<br>

8.user_id: (If authentication is used) Associates the event with a user.

<br>

9.device_info: (Optional) Captures browser or device info related to the detection session.

<br>

10,response_status: Indicates whether outbound alerts (SMS, email) were successfully sent.

<br>

11.features: (Optional) The feature vector or summary statistics (MFCC, chroma, ZCR, RMS) for advanced analysis or model audit.

<br>

12.notes / error: Additional notes or error messages if any issues arose during processing.

<br>

🧑‍💻 Model Training (model.ipynb):

1.Extracts audio features and handles class imbalance using SMOTE.

2.Trains and exports the classifier and scaler for use in backend predictions.

<br>

🧪 Model Testing (testmodel.ipynb):

1.Evaluates the model’s accuracy on a test set and outputs performance metrics.

<br>

🧠 Model Algorithm & Accuracy

1.Algorithm Used: The system uses a RandomForestClassifier algorithm from the scikit-learn library. Random Forest is an ensemble machine learning method that constructs multiple decision trees and outputs the mode of their predictions for robust classification.

<br>

2.Audio Features Used:
The model extracts several key audio features from the input sound using the librosa library, including:

1.MFCC (Mel-frequency Cepstral Coefficients):
MFCCs capture the timbral and spectral characteristics of an audio signal by representing how energy is distributed across different frequency bands, modeled on human hearing. They are highly effective for distinguishing different types of sounds, including the sharp, high-pitched qualities often present in screams.

<br>

2.Chroma Features:
Chroma features represent the intensity of each of the 12 distinct semitones (chromatic scale) of the musical octave, summarizing the harmonic content of the audio. In scream detection, chroma can help differentiate between harmonic (voiced) and noisy (unvoiced or chaotic) sounds.

<br>

3.Zero Crossing Rate (ZCR):
ZCR measures how often the signal changes sign in a given frame (i.e., crosses the zero amplitude axis). Screams tend to have a much higher zero-crossing rate due to their noisy, unstructured nature compared to speech or music, making ZCR a useful indicator.

<br>

4.RMS Energy (Root Mean Square Energy):
RMS energy quantifies the loudness or intensity of an audio segment. Screams are generally loud events with high energy content, so RMS is effective for helping the model identify potential alert events.

<br>

5.These features are extracted and combined into a feature vector that represents the spectral, temporal, and energy-related aspects of audio signals, enabling the model to robustly distinguish screams from non-scream sounds.

<br>

3.Handling Imbalance: The SMOTE (Synthetic Minority Over-sampling Technique) method from the imbalanced-learn package is used during model training to address class imbalance in the dataset.

<br>

4.Accuracy Achieved:
The trained model achieves up to 88.4% accuracy on the validation/test dataset, as reported in the testmodel.ipynb notebook. The model's performance was evaluated using other metrics such as precision, recall, and F1-score, confirming its robustness for real-world scream detection scenarios.

<br>

🛠️ Setup

1.Clone the repository and navigate to the project folder.

<br>

2.Set up environment variables in a .env file:

1.Twilio credentials (for SMS)

2.MongoDB URI

3.Gmail and Google App Password (for SMTP)

4.Any API keys (e.g., Google Maps JavaScript API key)

<br>

3.Install dependencies:

pip install -r requirements.txt

<br>

4.Train and export the model:
   
Run and complete model.ipynb.

<br>
5.Start the backend:

python app.py

<br>

6.Open index.html in your browser.

<br>

ℹ️ Additional Information

1.SMS Alerts: Implemented with Twilio Python SDK. Get a free trial  at Twilio, set up credentials.

2.Email Alerts: Implemented using Python’s smtplib and email libraries, configured for Gmail. For Gmail, generate an “App Password” from your Google Account security settings, and use this password instead of your main email password for the SMTP configuration.

3.Location Services: Uses the browser’s Geolocation API if available. If denied, attempts location estimation via IP geolocation (various APIs may be supported in app.py). Display and mapping use the Google Maps JavaScript API (API key required).

<br>

Dataset:

https://www.kaggle.com/datasets/whats2000/human-screaming-detection-dataset

📸 Screenshots



<img width="646" height="361" alt="1" src="https://github.com/user-attachments/assets/6414eff4-8363-4ff5-9d32-e1887facb438" />




<img width="648" height="361" alt="2" src="https://github.com/user-attachments/assets/9ed8fe68-a025-419e-a9b8-a36a50a8451e" />




<img width="648" height="361" alt="3" src="https://github.com/user-attachments/assets/fa4c50be-9351-43d8-bf6a-2e056911b4b5" />





<img width="648" height="257" alt="4" src="https://github.com/user-attachments/assets/38881437-2c9c-45cb-99f8-16554d65900b" />




<img width="629" height="264" alt="5" src="https://github.com/user-attachments/assets/80a56566-5225-4852-a78b-74b66718a711" />



<img width="654" height="346" alt="6" src="https://github.com/user-attachments/assets/0b27fafd-b999-45a8-a1d8-5814a101cb5a" />




<img width="641" height="361" alt="7" src="https://github.com/user-attachments/assets/f1053f89-046a-4107-a123-a830f7c6951e" />



<img width="324" height="356" alt="8" src="https://github.com/user-attachments/assets/0fca4a1c-8903-4aac-8272-d8b2223adcce" />





<img width="394" height="350" alt="9" src="https://github.com/user-attachments/assets/6e907b2e-f15a-4cea-96b7-8ed9b1ff53cc" />






<img width="706" height="410" alt="10" src="https://github.com/user-attachments/assets/22684c31-44a4-40f7-8b2f-e626df658f34" />































