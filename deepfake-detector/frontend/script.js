/* =====================================================
   FINAL FRONTEND ⇆ BACKEND + HISTORY SYSTEM
===================================================== */

const API_BASE = "http://127.0.0.1:8000";

/* =====================================================
   AUTO LOAD ON DASHBOARD
===================================================== */
document.addEventListener("DOMContentLoaded", () => {
  if (location.pathname.includes("dashboard.html")) {
    loadDashboard();
    loadHistory();
  }
});

/* =====================================================
   FILE DETECTION
===================================================== */
async function uploadFile() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) return alert("Please select a file.");

  const fd = new FormData();
  fd.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: fd,
    });

    const data = await res.json();

    const result = {
      label: data.label,
      confidence: data.confidence,
      filetype: file.type,
      timestamp: new Date().toLocaleString(),
    };

    // Save current result
    localStorage.setItem("deepfakeResult", JSON.stringify(result));

    // Add to history
    saveToHistory(result);

    window.location.href = "dashboard.html";
  } catch (err) {
    alert("Backend not reachable.");
  }
}

/* =====================================================
   LIVE CAMERA DETECTION
===================================================== */
function startLiveDetection() {
  navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
    const video = document.createElement("video");
    video.autoplay = true;

    document.getElementById("livePreview").innerHTML = "";
    document.getElementById("livePreview").appendChild(video);

    video.srcObject = stream;

    setTimeout(() => {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      canvas.getContext("2d").drawImage(video, 0, 0);

      canvas.toBlob(async (blob) => {
        const fd = new FormData();
        fd.append("file", blob, "live.jpg");

        try {
          const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            body: fd,
          });

          const data = await res.json();

          const result = {
            label: data.label,
            confidence: data.confidence,
            filetype: "Live Camera Frame",
            timestamp: new Date().toLocaleString(),
          };

          // Save current result
          localStorage.setItem("deepfakeResult", JSON.stringify(result));

          // Add to history
          saveToHistory(result);

          stream.getTracks().forEach((t) => t.stop());
          window.location.href = "dashboard.html";
        } catch (err) {
          alert("Live detection failed.");
        }
      });
    }, 3000);
  });
}

/* =====================================================
   LOAD CURRENT DASHBOARD RESULT
===================================================== */
function loadDashboard() {
  const data = JSON.parse(localStorage.getItem("deepfakeResult"));
  if (!data) return;

  document.getElementById("status").innerText = data.label;
  document.getElementById("confidence").innerText = data.confidence + "%";
  document.getElementById("filetype").innerText = data.filetype;
  document.getElementById("timestamp").innerText = data.timestamp;

  if (data.label === "FAKE") {
    document.getElementById("status").classList.add("status-fake");
  } else {
    document.getElementById("status").classList.add("status-real");
  }
}

/* =====================================================
   HISTORY — SAVE
===================================================== */
function saveToHistory(result) {
  let history = JSON.parse(localStorage.getItem("history")) || [];
  history.unshift(result); // add newest first
  localStorage.setItem("history", JSON.stringify(history));
}

/* =====================================================
   HISTORY — LOAD
===================================================== */
function loadHistory() {
  const history = JSON.parse(localStorage.getItem("history")) || [];
  const container = document.getElementById("historyList");

  if (!container) return;

  if (history.length === 0) {
    container.innerHTML = "<p>No history yet.</p>";
    return;
  }

  let html = "";

  history.forEach((item) => {
    html += `
      <div class="row" style="padding:12px 0;">
        <div>
          <strong>${item.label}</strong>
          <span>(${item.confidence}%)</span><br>
          <small>${item.filetype}</small>
        </div>
        <div><small>${item.timestamp}</small></div>
      </div>
    `;
  });

  container.innerHTML = html;
}

/* =====================================================
   HISTORY — CLEAR
===================================================== */
function clearHistory() {
  localStorage.removeItem("history");
  loadHistory();
}

/* =====================================================
   DOWNLOAD PDF
===================================================== */
function downloadPDF() {
  fetch(`${API_BASE}/report`)
    .then((res) => res.blob())
    .then((blob) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "deepfake_report.pdf";
      a.click();
    })
    .catch(() => {
      alert("Could not download PDF. Backend offline?");
    });
}

/* =====================================================
   NAVIGATION
===================================================== */
function goToDetect() {
  window.location.href = "detect.html";
}
