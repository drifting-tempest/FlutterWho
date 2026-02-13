import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse


## START LIKE THISSS -
## uvicorn web:app --reload



app = FastAPI()


class_names = ["apollo", "buckeye", "comma", "glasswing", "hackberry_emperor", "monarch", "morpho", "red_admiral"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# loading MobileNetV2 
model = models.mobilenet_v2(weights=None)


model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load("butterfly_classifier.pth", map_location=device))
model.eval()  



transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  #Mean values for RGB
                         [0.229, 0.224, 0.225])   #Std dev values for RGB
])


##web application code


@app.get("/", response_class=HTMLResponse)
def main_page():
    html_content = """
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlutterWho</title>
    <link href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;700&display=swap" rel="stylesheet">
    <style>
        /* cute pinkkk */
        :root {
            --dark-pink: #8B1A4B;
            --deep-pink: #C2185B;
            --hot-pink: #FF4081;
            --light-pink: #F8BBD9;
            --pale-pink: #FCE4EC;
        }

        /* LOADING SCREEN - CUTESY BUTTERFLY */
        #loader {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: linear-gradient(135deg, var(--dark-pink), var(--deep-pink));
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            transition: opacity 3s ease-out, transform 3s ease-out;
        }

        #loader.hidden {
            opacity: 0;
            transform: scale(0.9);
            pointer-events: none;
        }

        .butterfly-loader {
            width: 80px;
            height: 80px;
            position: relative;
            animation: flap 3.5s ease-in-out infinite alternate;
        }

        .wing {
            position: absolute;
            background: linear-gradient(135deg, var(--pale-pink), var(--hot-pink));
            border-radius: 50% 30% 50% 20%;
            box-shadow: 0 0 20px rgba(255, 64, 129, 0.6);
        }

        .wing.top-left {
            width: 40px;
            height: 35px;
            top: 10px;
            left: 10px;
            transform-origin: bottom right;
        }

        .wing.top-right {
            width: 40px;
            height: 35px;
            top: 10px;
            right: 10px;
            transform-origin: bottom left;
        }

        .wing.bottom-left {
            width: 30px;
            height: 25px;
            bottom: 10px;
            left: 15px;
            transform-origin: top right;
        }

        .wing.bottom-right {
            width: 30px;
            height: 25px;
            bottom: 10px;
            right: 15px;
            transform-origin: top left;
        }

        .body {
            position: absolute;
            width: 12px;
            height: 30px;
            background: linear-gradient(var(--light-pink), var(--hot-pink));
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            border-radius: 6px;
            box-shadow: 0 0 10px rgba(255, 64, 129, 0.8);
        }

        @keyframes flap {
            0% {
                transform: rotate(-5deg);
            }
            100% {
                transform: rotate(5deg);
            }
        }

        .loading-dots {
            display: flex;
            gap: 8px;
            margin-top: 30px;
        }

        .dot {
            width: 12px;
            height: 12px;
            background: var(--pale-pink);
            border-radius: 50%;
            animation: bounce 1.2s infinite ease-in-out;
        }

        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1.2); opacity: 1; }
        }

        body {
            font-family: 'Spectral', serif;
            text-align: center;
            margin: 0;
            background-color: var(--dark-pink);
            color: white;
            display: flex;
            flex-direction: column;
            align-items: center;
        }


        /* top nav bar; no need for any use rn */
        nav {
            background-color: var(--deep-pink);
            padding: 16px 0;
            display: flex;
            justify-content: center;
            gap: 80px;
            font-size: 19px;
            letter-spacing: 1px;
            width: 100%;
        }


        nav a {
            color: white;
            text-decoration: none;
            font-weight: 600;
        }


        nav a:hover {
            text-decoration: underline;
        }


        /* MAIN MAIN title, glow effect */
        h1 {
            margin-top: 20px;
            margin-bottom: 20px;
            font-size: 3.2rem;
            font-weight: 100;
            letter-spacing: 6px;
            background: linear-gradient(135deg, var(--pale-pink), var(--hot-pink));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 
                0 0 8px var(--hot-pink),
                0 0 16px var(--light-pink);
            animation: glow 2.5s ease-in-out infinite alternate;
            filter: drop-shadow(0 0 15px rgba(255, 64, 129, 0.3));
        }


        @keyframes glow {
            from {
                text-shadow: 
                    0 0 15px var(--hot-pink),
                    0 0 30px var(--light-pink),
                    0 0 50px var(--deep-pink);
                filter: drop-shadow(0 0 10px rgba(255, 64, 129, 0.3));
            }
            to {
                text-shadow: 
                    0 0 30px var(--hot-pink),
                    0 0 50px var(--light-pink),
                    0 0 80px var(--deep-pink);
                filter: drop-shadow(0 0 25px rgba(255, 64, 129, 0.4));
            }
        }


        /* container for prediction stuff */
        .container {
            display: flex;
            gap: 40px;
            margin-top: 0px;
            justify-content: center;
            flex-wrap: wrap;  /* Makes it responsive on smaller screens */
        }


        .dropzone, .preview-box, .info-box {
            background-color: var(--hot-pink);
            width: 230px;
            min-height: 250px;
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            cursor: pointer;
            transition: all 0.3s ease, opacity 0.5s ease;
            background-opacity: 70%;
        }


        .dropzone:hover {
            background-color: var(--light-pink);
            color: var(--dark-pink);
        }


        .arrow-circle {
            background-color: var(--light-pink);
            width: 120px;
            height: 120px;
            border-radius: 36%;  /* Not quite 50% to make it slightly less circular */
            display: flex;
            align-items: center;
            justify-content: center;
        }


        .arrow {
            font-size: 50px;
            color: var(--dark-pink);
            font-weight: bold;
            transform: translateY(-3px);  /* Slight adjustment to center it better */
        }


        .dragtext {
            font-size: 18px;
            margin-top: 14px;
            letter-spacing: 1.5px;
        }


        #d {
            margin-top: 20px;
            font-size: 20px;
            color: var(--pale-pink);
        }


        #result {
            margin-top: 20px;
            font-size: 16px;
            color: var(--pale-pink);
        }


        .preview-box img {
            max-width: 90%;
            max-height: 90%;
            border-radius: 16px;
        }


        .info-box {
            background-color: var(--light-pink);
            opacity: 0;  /* Hidden initially, fades in after prediction */
            padding: 16px;
            color: var(--dark-pink);
            text-align: center;
        }
        
        .confetti {
            position: fixed;
            width: 10px;
            height: 10px;
            top: -10px;
            z-index: 999;
            animation: fall linear forwards;
        }


        @keyframes fall {
            to {
                transform: translateY(110vh) rotate(360deg);
                opacity: 0;
            }
        }
    </style>
</head>


<body>
    <!-- CUTESY LOADING SCREEN -->
    <div id="loader">
        <div class="butterfly-loader">
            <div class="wing top-left"></div>
            <div class="wing top-right"></div>
            <div class="wing bottom-left"></div>
            <div class="wing bottom-right"></div>
            <div class="body"></div>
        </div>
        <div class="loading-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
        <p style="margin-top: 20px; font-size: 18px; letter-spacing: 2px;">Fluttering awake...</p>
    </div>

    <nav>
        <a href="#">FlutterWho</a>
    </nav>


    <h1>FlutterWho</h1>


    <div class="container">
        <!-- Upload zone -->
        <div class="dropzone" id="dropzone">
            <div class="arrow-circle">
                <div class="arrow">↑</div>
            </div>
            <p id="d"> DRAG & DROP or CLICK </p>
        </div>


        <!-- Image preview appears here -->
        <div class="preview-box" id="preview-box" style="opacity:0;">
        </div>


        <!-- Butterfly info appears here after prediction -->
        <div class="info-box" id="info-box">
        </div>
    </div>


    <p id="result"></p>


    <script>
        // HIDE LOADING SCREEN when page fully loads
        window.addEventListener('load', function() {
            document.getElementById('loader').classList.add('hidden');
        });

        // main eles
        const dropzone = document.getElementById("dropzone");
        const result = document.getElementById("result");
        const previewBox = document.getElementById("preview-box");
        const infoBox = document.getElementById("info-box");


        // all butterflies info
        const butterflyInfo = {
            "monarch": {
                temperature: "20-35°C",
                vegetation: "Milkweed fields, meadows, gardens",
                funFact: "Migrates up to 4,800 km from Canada to Mexico!"
            },
            "morpho": {
                temperature: "25-35°C",
                vegetation: "Tropical rainforests",
                funFact: "Iridescent blue wings visible only at certain angles."
            },
            "comma": {
                temperature: "10-25°C",
                vegetation: "Woodlands, hedgerows, gardens",
                funFact: "Ragged wing edges camouflage as dead leaves."
            },
            "red_admiral": {
                temperature: "15-30°C",
                vegetation: "Woodlands, gardens, parks",
                funFact: "Fast flier, territorial males patrol patches."
            },
            "buckeye": {
                temperature: "20-35°C",
                vegetation: "Open fields, roadsides",
                funFact: "Large eyespots deter predators."
            },
            "glasswing": {
                temperature: "25-35°C",
                vegetation: "Tropical forests",
                funFact: "Transparent wings from modified scales."
            },
            "apollo": {
                temperature: "5-25°C",
                vegetation: "Alpine meadows, rocky slopes",
                funFact: "High-altitude specialist, endangered in Europe."
            },
            "hackberry_emperor": {
                temperature: "20-35°C",
                vegetation: "Hackberry woodlands, river valleys",
                funFact: "Aggressive toward other butterflies!"
            }
        };


        // previewing uploaded img
        function showPreview(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewBox.innerHTML = "<img src='" + e.target.result + "' />";
                previewBox.style.opacity = "1";
            };
            reader.readAsDataURL(file);
        }


        // butterfly info
        function showInfo(className) {
            if (butterflyInfo[className]) {
                const info = butterflyInfo[className];
                infoBox.innerHTML = `
                    <b>Temperature:</b> ${info.temperature}<br>
                    <b>Vegetation:</b> ${info.vegetation}<br>
                    <b>Fun Fact:</b> ${info.funFact}
                `;
                infoBox.style.opacity = "1";
            } else {
                infoBox.innerHTML = "No info available";
                infoBox.style.opacity = "1";
            }
        }


        function launchConfetti() {
            for (let i = 0; i < 40; i++) {
                const confetti = document.createElement("div");
                confetti.className = "confetti";
                confetti.style.left = Math.random() * 100 + "vw";
                
                // pick b/w 2 colors
                confetti.style.backgroundColor = i % 2 === 0 ? "#F8BBD9" : "#FF4081";
                confetti.style.animationDuration = (2 + Math.random() * 1.5) + "s";


                document.body.appendChild(confetti);


                // removes in 3 secs
                setTimeout(() => confetti.remove(), 3000);
            }
        }


        // main function
        function handleFile(file) {
            showPreview(file);
            
            // prepping file
            const formData = new FormData();
            formData.append("file", file);


            // prediction getting
            fetch("/predict/", { 
                method: "POST", 
                body: formData 
            })
            .then(response => response.json())
            .then(data => {
                result.innerHTML = "Top 3:<br>" + 
                    data.predictions.map(p => 
                        `<b>${p.class.replace(/_/g, ' ')}</b>: ${p.confidence.toFixed(1)}%`
                    ).join("<br>");
                
                // Show info for the top prediction
                showInfo(data.predictions[0].class);  
                
                // Trigger confetti celebration
                launchConfetti();
            });
        }


        dropzone.addEventListener("dragover", function(e) {
            e.preventDefault();
            dropzone.style.background = "#F8BBD9";
        });


        dropzone.addEventListener("dragleave", function() {
            dropzone.style.background = "#FF4081";
        });


        dropzone.addEventListener("drop", function(e) {
            e.preventDefault();
            dropzone.style.background = "#FF4081";
            const file = e.dataTransfer.files[0];
            handleFile(file);
        });


        dropzone.addEventListener("click", function() {
            // Create hidden file input element
            const input = document.createElement("input");
            input.type = "file";
            input.accept = "image/*";
            input.onchange = function(e) {
                const file = e.target.files[0];
                handleFile(file);
            };
            input.click();  // Trigger the file picker
        });


    </script>
</body>
</html> """


    return HTMLResponse(content=html_content)



## actual prediction


@app.post("/predict/")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")


    # transformations and add batch dimension
    img_tensor = transform(img).unsqueeze(0).to(device)
  
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, prediction = torch.max(probabilities, dim=0)


    # top 3 predictions
    top3_probabilities, top3_indices = torch.topk(probabilities, 3)
  
    # responses
    predictions_list = []
    for idx, prob in zip(top3_indices, top3_probabilities):
        predictions_list.append({
            "class": class_names[idx], 
            "confidence": prob.item() * 100
        })
  
    return JSONResponse({
        "predictions": predictions_list
    })
