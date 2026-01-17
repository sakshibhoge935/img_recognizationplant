# ultimate_plant_app_v22.py

import os
import cv2
import numpy as np
import threading
import time
import queue
from flask import Flask, render_template_string, Response, request, jsonify, redirect, url_for

# --- 1. CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pyttsx3

# --- 2. ROBUST PLANT DATABASE (Detailed Info) ---
PLANT_DB = {
    "Tulsi": {
        "images": [
            "https://cdn.britannica.com/87/207887-050-F48CB55D/basil.jpg",
            "https://plantsguru.com/cdn/shop/files/ram-tulsi.webp?v=1737032350",
            "https://www.shutterstock.com/image-photo/closeup-fresh-holy-basil-ocimum-260nw-2669449147.jpg",
            "https://www.trustbasket.com/cdn/shop/articles/Tulsi.jpg?v=1687508866",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRdJKXhf-8_QsH55SxIUHnR6nZc9b6HbpF3lg&s"
        ],
        "info": """• Scientific Name: Ocimum tenuiflorum (Holy Basil).
• Overview: Known as the 'Queen of Herbs' in Ayurveda, revered for its divine and medicinal properties.
• Key Compounds: Ursolic acid, eugenol, bioflavonoids, and Vitamin A/C.
• Respiratory Health: Powerful expectorant; helps relieve coughs, colds, bronchitis, and asthma by clearing congestion.
• Stress Relief: Acts as a powerful adaptogen, lowering cortisol levels and promoting mental clarity.
• Immunity: Boosts the body's natural defense mechanism against viral and bacterial infections.
• Skin Care: Its antibacterial properties treat acne, skin infections, and ringworm.
• Dental Health: Destroys bacteria that cause dental cavities, plaque, and bad breath.
• Heart Health: Reduces cholesterol levels and regulates blood pressure.
• Usage: Can be consumed raw, brewed as tea, or applied as a paste."""
    },
    "Rose": {
        "images": [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Rosa_rubiginosa_1.jpg/300px-Rosa_rubiginosa_1.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Rosa_Precious_platinum.jpg/300px-Rosa_Precious_platinum.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Rosa_canina_flower.jpg/300px-Rosa_canina_flower.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Rose_bush.jpg/300px-Rose_bush.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Rose_flower_2.jpg/300px-Rose_flower_2.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Rosa_Red_Chateau01.jpg/300px-Rosa_Red_Chateau01.jpg"
        ],
        "info": """• Scientific Name: Rosa rubiginosa / Rosa damascena.
• Overview: A symbol of love and beauty, but also a powerhouse of medicinal benefits used for centuries.
• Nutritional Value: Rose hips are one of the richest natural sources of Vitamin C.
• Skincare: Rose water is a natural astringent that tightens pores, reduces redness, and hydrates dry skin.
• Anti-Inflammatory: Helps treat eczema, dermatitis, and acne due to its cooling properties.
• Digestion: Rose petal tea improves digestion, soothes the stomach, and eliminates toxins from the body.
• Mental Health: The aroma of rose is proven to reduce anxiety, stress, and depression.
• Pain Relief: Known to alleviate menstrual cramps and reduce uterine congestion.
• Sore Throat: A decoction of rose petals is effective in soothing sore throats and enlarged tonsils.
• Usage: Use as rose water, essential oil, herbal tea, or dried petal powder."""
    },
    "Lemon": {
        "images": [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Lemon.jpg/300px-Lemon.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Lemon_Fruit.jpg/300px-Lemon_Fruit.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Lemon_Close_Up.jpg/300px-Lemon_Close_Up.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Lemon_fruit.jpg/300px-Lemon_fruit.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Whole_Lemon.jpg/300px-Whole_Lemon.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Lemon_Tree_in_Santa_Clara_California.jpg/300px-Lemon_Tree_in_Santa_Clara_California.jpg"
        ],
        "info": """• Scientific Name: Citrus limon.
• Overview: A versatile citrus fruit essential for culinary, cleaning, and medicinal purposes worldwide.
• Vitamin Powerhouse: Extremely high in Vitamin C, essential for immune system function.
• Digestive Aid: Warm lemon water jumpstarts the digestive system and helps prevent constipation.
• Kidney Health: Contains citrate, which helps prevent calcium kidney stones.
• Weight Loss: Soluble pectin fiber helps you feel full longer, aiding in weight management.
• Skin Health: The Vitamin C plays a vital role in the formation of collagen, the support system of the skin.
• Heart Health: Hesperidin and diosmin in lemons have been found to lower cholesterol.
• Alkalizing: Despite being acidic, it produces an alkalizing by-product in the body.
• Usage: Juice, zest, essential oil, or infused water."""
    },
    "Mint": {
        "images": [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Mint-leaves-2009.jpg/300px-Mint-leaves-2009.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Mentha_spicata_leaves.jpg/300px-Mentha_spicata_leaves.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Mentha_aquatica_2.jpg/300px-Mentha_aquatica_2.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Mint_plants.jpg/300px-Mint_plants.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Mentha_piperita_2009.jpg/300px-Mentha_piperita_2009.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Mentha_longifolia_HabitusR.jpg/300px-Mentha_longifolia_HabitusR.jpg"
        ],
        "info": """• Scientific Name: Mentha (Peppermint/Spearmint).
• Overview: A fast-growing aromatic herb known for its cooling sensation and digestive benefits.
• Active Compound: Menthol, which acts as a natural decongestant and pain reliever.
• Digestive Health: Relieves indigestion, bloating, gas, and symptoms of IBS (Irritable Bowel Syndrome).
• Respiratory Relief: The aroma opens up nasal passages and clears the lungs of congestion.
• Oral Hygiene: Kills bacteria in the mouth, prevents cavities, and freshens breath instantly.
• Headache Relief: Applying mint oil to the forehead relaxes muscles and relieves tension headaches.
• Nausea: Smelling or consuming mint helps reduce nausea, especially travel sickness.
• Skin: Mint juice soothes mosquito bites and acts as a skin toner.
• Usage: Tea, chutneys, garnishes, essential oils, and medicinal balms."""
    },
    "Aloe_Vera": {
        "images": [
           "https://rukminim2.flixcart.com/image/480/640/xif0q/plant-sapling/4/z/a/yes-annual-yes-aloe-vera-plant-1-pot-live-hub-original-imaghjs5wret9teg.jpeg?q=90",
           "https://m.media-amazon.com/images/I/81XWpVvk5AL._AC_UF1000,1000_QL80_.jpg",
           "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQpYW-XrWPkGER_H36hjurtGZvFwErJ6RviDQ&s",
           "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSpsNOKQaApMDxYeIOq1h4b8r4JhEF13o0cYA&s",
           "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzyeHtGtxJSvipUgeu2IAa9ryeh5ivzg1RuQ&s"
        ],
        "info": """• Scientific Name: Aloe barbadensis miller.
• Overview: A succulent plant species often cited as the 'plant of immortality' by Ancient Egyptians.
• Burn Treatment: The gel provides immediate cooling relief for sunburns and minor kitchen burns.
• Wound Healing: Accelerates skin repair by increasing collagen synthesis and cross-linking.
• Moisturizer: A non-greasy moisturizer that hydrates skin without clogging pores.
• Digestive Health: Aloe latex (yellow part) helps cure constipation, while the juice soothes acid reflux.
• Dental Health: Aloe vera mouthwash is as effective as chlorhexidine in reducing dental plaque.
• Anti-Aging: Rich in beta-carotene and Vitamin C/E, helping to improve skin elasticity.
• Blood Sugar: Some studies suggest it enhances insulin sensitivity in type 2 diabetes management.
• Usage: Gel applied topically, juice consumed orally (processed to remove aloin)."""
    }
}

PLANT_NAMES = list(PLANT_DB.keys())
X_data, y_data = [], [] 

# --- 3. STARTUP LOGIC ---
def create_virtual_plant(name):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    if name == "Tulsi": cv2.ellipse(img, (32, 32), (10, 25), 0, 0, 360, (0, 180, 0), -1)
    elif name == "Rose": cv2.circle(img, (32, 32), 15, (0, 0, 200), -1)
    elif name == "Lemon": cv2.circle(img, (32, 32), 18, (0, 255, 255), -1)
    elif name == "Mint": 
        for i in range(5): cv2.circle(img, (32+np.random.randint(-10,10), 32+np.random.randint(-10,10)), 8, (50, 200, 50), -1)
    elif name == "Aloe_Vera": cv2.line(img, (32, 60), (32, 10), (34, 139, 34), 5)
    noise = np.random.randint(-20, 20, (64, 64, 3), dtype=np.int16)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return image.img_to_array(img) / 255.0

def load_initial_knowledge():
    global X_data, y_data
    print("🚀 Initializing AI Knowledge...")
    for idx, name in enumerate(PLANT_NAMES):
        for _ in range(50):
            X_data.append(create_virtual_plant(name))
            y_data.append(idx)

# --- 4. MODEL TRAINING ---
model = None
is_training = False

def train_model():
    global model, is_training
    if is_training: return
    is_training = True
    print("🧠 Training AI Brain...")
    
    X = np.array(X_data)
    y = to_categorical(np.array(y_data), num_classes=len(PLANT_NAMES))
    
    new_model = Sequential([
        Input(shape=(64, 64, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(PLANT_NAMES), activation='softmax')
    ])
    new_model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    new_model.fit(X, y, epochs=10, verbose=0)
    
    model = new_model
    is_training = False
    print("✅ AI Ready!")

load_initial_knowledge()
threading.Thread(target=train_model).start()

# --- 5. FLASK & CAMERA ---
app = Flask(__name__)
cap = None
camera_lock = threading.Lock() # 🛑 SAFETY LOCK
speech_queue = queue.Queue()
current_prediction = "Scanning..." 

def speech_worker():
    try:
        engine = pyttsx3.init()
        while True:
            text = speech_queue.get()
            if text is None: break
            engine.say(text)
            engine.runAndWait()
            speech_queue.task_done()
    except: pass

threading.Thread(target=speech_worker, daemon=True).start()

def gen_frames():
    global cap, current_prediction
    frame_count = 0
    
    while True:
        with camera_lock: # 🛑 SAFELY ACCESS CAMERA
            if cap is None or not cap.isOpened():
                # Release lock quickly before yielding to prevent blocking
                pass 
            else:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        pass # Handle read failure
                    else:
                        # Only process if we got a frame
                        if model is not None and not is_training and frame_count % 5 == 0:
                            try:
                                img_proc = cv2.resize(frame, (64, 64))
                                img_proc = np.expand_dims(img_proc / 255.0, axis=0)
                                preds = model.predict(img_proc, verbose=0)[0]
                                top_idx = np.argmax(preds)
                                conf = preds[top_idx]
                                if conf > 0.6: current_prediction = PLANT_NAMES[top_idx]
                                else: current_prediction = "Scanning..."
                            except: pass

                        # Draw UI
                        cv2.rectangle(frame, (0,0), (640, 50), (0,0,0), -1)
                        cv2.putText(frame, f"AI Sees: {current_prediction}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        
                        if is_training:
                             cv2.rectangle(frame, (0, 440), (640, 480), (0,0,255), -1)
                             cv2.putText(frame, "LEARNING...", (250, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                        ret, buffer = cv2.imencode('.jpg', frame)
                        if ret:
                            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        frame_count += 1
                        
                except Exception as e:
                    # Catch OpenCV errors gracefully
                    print(f"Camera Error: {e}")
        
        # If we didn't yield a frame above (camera closed), yield a blank one here
        # This prevents the "Streamed response" error
        if cap is None or not cap.isOpened():
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "CAMERA OFF", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            ret, buf = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.5)

# --- 6. ROUTES & UI ---
@app.route('/')
def index():
    options = "".join([f"<option value='{p}'>{p}</option>" for p in PLANT_NAMES])
    return render_template_string(f"""
<!DOCTYPE html>
<html>
<head>
    <title>AyuVeda</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400&display=swap" rel="stylesheet">
    <style>
        :root {{ --primary: #00ff88; --secondary: #00d4ff; --bg: #0a0a0a; --panel: rgba(30, 30, 30, 0.6); }}
        body {{ background: var(--bg); color: #eee; font-family: 'Roboto', sans-serif; margin: 0; padding: 20px; }}
        h1 {{ font-family: 'Orbitron', sans-serif; background: linear-gradient(to right, var(--primary), var(--secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }}
        .main-grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; max-width: 1400px; margin: auto; }}
        @media (max-width: 900px) {{ .main-grid {{ grid-template-columns: 1fr; }} }}
        .glass-panel {{ background: var(--panel); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.37); }}
        .cam-container {{ border-radius: 10px; overflow: hidden; border: 2px solid var(--primary); }}
        .cam-container img {{ width: 100%; display: block; }}
        .info-scroll {{ max-height: 250px; overflow-y: auto; white-space: pre-wrap; line-height: 1.6; color: #ccc; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; }}
        .img-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; margin-top: 15px; }}
        .grid-img {{ width: 100%; height: 100px; object-fit: cover; border-radius: 8px; transition: transform 0.3s; cursor: pointer; }}
        .grid-img:hover {{ transform: scale(1.05); }}
        button {{ padding: 12px 25px; border: none; border-radius: 8px; font-weight: bold; font-family: 'Orbitron', sans-serif; cursor: pointer; margin: 5px; }}
        .btn-start {{ background: linear-gradient(45deg, #00b09b, #96c93d); color: #000; }}
        .btn-stop {{ background: linear-gradient(45deg, #ff416c, #ff4b2b); color: white; }}
        .btn-teach {{ background: linear-gradient(45deg, #2193b0, #6dd5ed); color: #000; width: 100%; margin-top: 10px; }}
        .chat-container {{ height: 300px; overflow-y: auto; background: rgba(0,0,0,0.3); padding: 15px; margin-bottom: 10px; border-radius: 10px; }}
        .msg {{ margin: 8px 0; padding: 10px 15px; border-radius: 15px; max-width: 80%; }}
        .bot {{ background: rgba(0, 255, 136, 0.1); border-left: 3px solid var(--primary); color: #fff; }}
        .user {{ background: rgba(0, 212, 255, 0.1); border-right: 3px solid var(--secondary); color: #fff; margin-left: auto; text-align: right; }}
        input, select {{ width: 100%; padding: 10px; background: #222; color: white; border: 1px solid #444; border-radius: 5px; margin-bottom: 10px; }}
    </style>
    <script>
        const PLANT_DB = {PLANT_DB};
        setInterval(function(){{
            fetch('/get_prediction').then(r => r.json()).then(data => {{
                let plant = data.plant;
                if(plant !== "Scanning..." && PLANT_DB[plant]){{
                    document.getElementById('plant_name').innerText = plant;
                    document.getElementById('plant_info').innerText = PLANT_DB[plant].info;
                    let images = PLANT_DB[plant].images;
                    let gridHTML = "";
                    for(let i=0; i<Math.min(6, images.length); i++){{
                        gridHTML += `<img src="${{images[i]}}" class="grid-img" onclick="window.open(this.src)">`;
                    }}
                    document.getElementById('img_grid').innerHTML = gridHTML;
                }}
            }});
        }}, 1000);
        function toggleCam(active){{
            fetch(active ? '/start_camera' : '/stop_camera').then(()=>{{ if(active) location.reload(); }});
        }}
        function sendChat(){{
            let input = document.getElementById('chatInput');
            let txt = input.value;
            if(!txt) return;
            let box = document.getElementById('chatHistory');
            box.innerHTML += `<div class='msg user'>${{txt}}</div>`;
            fetch('/chat', {{ method: 'POST', body: 'msg='+encodeURIComponent(txt), headers: {{'Content-Type': 'application/x-www-form-urlencoded'}} }})
            .then(r => r.text()).then(reply => {{
                box.innerHTML += `<div class='msg bot'>${{reply}}</div>`;
                box.scrollTop = box.scrollHeight;
                input.value = '';
            }});
        }}
    </script>
</head>
<body>
    <h1>🌿 AYUVEDA</h1>
    <div class="main-grid">
        <div>
            <div class="glass-panel" style="text-align:center; margin-bottom: 20px;">
                <button class="btn-start" onclick="toggleCam(true)">▶ Initialize</button>
                <button class="btn-stop" onclick="toggleCam(false)">⏹ Terminate</button>
            </div>
            <div class="glass-panel cam-container">
                <img src="/video_feed" width="640" height="480">
            </div>
            <div class="glass-panel" style="margin-top: 20px;">
                <h3 style="color:var(--accent); margin-top:0;">⚡ Override / Teach AI</h3>
                <form action="/correct" method="post">
                    <select name="correct_plant">{options}</select>
                    <button type="submit" class="btn-teach">Confirm & Learn</button>
                </form>
            </div>
        </div>
        <div>
            <div class="glass-panel" style="margin-bottom: 20px;">
                <h2 id="plant_name" style="color:var(--secondary); margin-top:0;">Scanning...</h2>
                <div class="info-scroll">
                    <p id="plant_info" style="margin:0;">Waiting for visual confirmation...</p>
                </div>
                <div id="img_grid" class="img-grid"></div>
            </div>
            <div class="glass-panel">
                <h3 style="margin-top:0;">💬 Assistant</h3>
                <div id="chatHistory" class="chat-container">
                    <div class="msg bot">System Online. Ask about Tulsi, Rose, Lemon, Mint, or Aloe Vera.</div>
                </div>
                <div style="display:flex; gap:10px;">
                    <input type="text" id="chatInput" placeholder="Query database...">
                    <button onclick="sendChat()" style="background:var(--primary); color:black; margin:0;">➤</button>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
""")

@app.route('/get_prediction')
def get_prediction():
    return jsonify({"plant": current_prediction})

@app.route('/correct', methods=['POST'])
def correct():
    global cap, X_data, y_data
    plant = request.form.get("correct_plant")
    idx = PLANT_NAMES.index(plant)
    
    with camera_lock: # 🛑 SAFELY READ FOR CORRECTION
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                img = cv2.resize(frame, (64, 64))
                for _ in range(30): 
                    X_data.append(image.img_to_array(img) / 255.0)
                    y_data.append(idx)
                threading.Thread(target=train_model).start()
    return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.form.get("msg", "").lower()
    found = next((p for p in PLANT_NAMES if p.lower() in msg), None)
    if found: resp = PLANT_DB[found]['info']
    else: resp = "Data unavailable. I specialize in Tulsi, Rose, Lemon, Mint, and Aloe Vera."
    speech_queue.put(resp)
    return resp

@app.route('/start_camera')
def start_camera():
    global cap
    with camera_lock:
        if cap is None: cap = cv2.VideoCapture(0)
    return "OK"

@app.route('/stop_camera')
def stop_camera():
    global cap
    with camera_lock:
        if cap: cap.release(); cap = None
    return "OK"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)