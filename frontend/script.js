const WORDS = [
  "cat","cup","car","sun","eye","house",
  "pants","tree","strawberry","wine glass"
];

const STORAGE_KEY = "montagsmaler_correct_drawings";

// DOM
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const targetWordSpan = document.getElementById("target-word");
const predictionSpan = document.getElementById("prediction");
const confidenceSpan = document.getElementById("confidence");
const topList = document.getElementById("top-list");

const captionText = document.getElementById("caption-text");
const captionConf = document.getElementById("caption-conf");
const captionTop = document.getElementById("caption-top");

const saveBtn = document.getElementById("save-btn");
const galleryDiv = document.getElementById("gallery");

const penBtn = document.getElementById("pen-btn");
const eraserBtn = document.getElementById("eraser-btn");
const clearBtn = document.getElementById("clear");
const predictBtn = document.getElementById("predict");

let targetWord = null;
let drawing = false;
let mode = "pen";
let lastImage = null;
let lastConfidence = null;
let lastCaption = null;
let lastCaptionConf = null;
let lastPredictTime = 0;
const LIVE_PREDICT_INTERVAL = 500; // ms


// ===== Canvas =====
ctx.lineCap = "round";
setPen();

function setPen() {
  mode = "pen";
  ctx.globalCompositeOperation = "source-over";
  ctx.strokeStyle = "black";
  ctx.lineWidth = 12;
}

function setEraser() {
  mode = "eraser";
  ctx.globalCompositeOperation = "destination-out";
  ctx.lineWidth = 24;
}

penBtn.onclick = setPen;
eraserBtn.onclick = setEraser;

// ===== Wort =====
function chooseNewWord() {
  targetWord = WORDS[Math.floor(Math.random() * WORDS.length)];
  targetWordSpan.textContent = targetWord;
}
chooseNewWord();

// ===== Zeichnen =====
canvas.addEventListener("mousedown", () => {
  drawing = true;
  ctx.beginPath();
});

canvas.addEventListener("mouseup", () => {
  drawing = false;
  ctx.beginPath();
});

canvas.addEventListener("mousemove", e => {
  if (!drawing) return;

  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left;
  const y = e.clientY - r.top;

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);

  //LIVE-PREDICT
  const now = Date.now();
  if (
    mode === "pen" &&
    now - lastPredictTime > LIVE_PREDICT_INTERVAL
  ) {
    lastPredictTime = now;
    maybePredict();
  }
});

// ===== Clear =====
clearBtn.onclick = () => {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  chooseNewWord();
};

// ===== Snapshot =====
function snapshot() {
  const tmp = document.createElement("canvas");
  tmp.width = canvas.width;
  tmp.height = canvas.height;
  const t = tmp.getContext("2d");
  t.fillStyle = "white";
  t.fillRect(0,0,tmp.width,tmp.height);
  t.drawImage(canvas,0,0);
  return tmp.toDataURL("image/png");
}

// ===== Predict =====
predictBtn.onclick = maybePredict;

async function maybePredict() {
  const img = snapshot();
  lastImage = img;

  const fd = new FormData();
  fd.append("image_base64", img);

  // CNN
  const r1 = await fetch("http://127.0.0.1:8003/predict",{method:"POST",body:fd});
  const cnn = await r1.json();

  predictionSpan.textContent = cnn.prediction;
  confidenceSpan.textContent = Math.round(cnn.confidence*100)+"%";

  topList.innerHTML="";
  cnn.top.forEach(x=>{
    const li=document.createElement("li");
    li.textContent=`${x.label} (${Math.round(x.confidence*100)}%)`;
    topList.appendChild(li);
  });

  // CLIP
  const r2 = await fetch("http://127.0.0.1:8003/caption",{method:"POST",body:fd});
  const clip = await r2.json();

  captionText.textContent = clip.caption;
  captionConf.textContent = Math.round(clip.confidence*100)+"%";

  lastCaption = clip.caption;
  lastCaptionConf = clip.confidence;
  lastConfidence = cnn.confidence;

  saveBtn.style.display = "inline-block";
}

// ===== Speichern =====
saveBtn.onclick = () => {
  const data = JSON.parse(localStorage.getItem(STORAGE_KEY)||"[]");
  const entry = {
    image:lastImage,
    label:targetWord,
    confidence:lastConfidence,
    caption:lastCaption,
    captionConfidence:lastCaptionConf,
    time:Date.now()
  };
  data.push(entry);
  localStorage.setItem(STORAGE_KEY,JSON.stringify(data));
  addToGallery(entry);
  saveBtn.style.display="none";
};

// ===== Galerie =====
function addToGallery(e) {
  const w=document.createElement("div");
  w.className="gallery-item";

  const img=document.createElement("img");
  img.src=e.image;

  const c1=document.createElement("div");
  c1.className="gallery-caption";
  c1.textContent=`CNN: ${e.label}`;

  const c2=document.createElement("div");
  c2.className="gallery-caption";
  c2.textContent=`CLIP: ${e.caption}`;

  const c3=document.createElement("div");
  c3.className="gallery-confidence";
  c3.textContent=`${Math.round(e.captionConfidence*100)}%`;

  const overlay=document.createElement("div");
  overlay.className="gallery-overlay";

  const del=document.createElement("button");
  del.textContent="×";
  del.onclick=()=>{
    w.remove();
    const d=JSON.parse(localStorage.getItem(STORAGE_KEY)||"[]")
      .filter(x=>x.time!==e.time);
    localStorage.setItem(STORAGE_KEY,JSON.stringify(d));
  };

  overlay.appendChild(del);
  w.append(img,c1,c2,c3,overlay);
  galleryDiv.appendChild(w);
}

// ===== Laden =====
(JSON.parse(localStorage.getItem(STORAGE_KEY)||"[]")).forEach(addToGallery);
