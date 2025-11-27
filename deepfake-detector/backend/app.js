// -------------------------------------------
// IMPORTS (ONLY ONCE, CLEANED)
// -------------------------------------------
const express = require("express");
const cors = require("cors");
const fileUpload = require("express-fileupload");
const path = require("path");
const { exec } = require("child_process");

// -------------------------------------------
// APP SETUP
// -------------------------------------------
const app = express();
app.use(cors());
app.use(express.json());
app.use(fileUpload());

// Serve uploads folder
app.use("/uploads", express.static(path.join(__dirname, "uploads")));

// -------------------------------------------
// HOME ROUTE
// -------------------------------------------
app.get("/", (req, res) => {
  res.send(`
        <h1>Deepfake Detector Backend</h1>
        <p>Server is running ✔</p>
        <p>Use /video to upload and detect</p>
    `);
});

// -------------------------------------------
// FILE UPLOAD + DEEPFAKE DETECTION (MAIN ROUTE)
// -------------------------------------------
app.post("/video", async (req, res) => {
  try {
    if (!req.files || !req.files.video) {
      return res.status(400).json({ error: "No video uploaded" });
    }

    const videoFile = req.files.video;
    const uploadPath = path.join(__dirname, "uploads", videoFile.name);

    await videoFile.mv(uploadPath);

    // Run Python model
    exec(`python model.py "${uploadPath}"`, (err, stdout, stderr) => {
      if (err) {
        console.log(stderr);
        return res.json({ error: "Python model error" });
      }

      let result;
      try {
        result = JSON.parse(stdout.trim());
      } catch (e) {
        return res.json({ error: "Invalid model output" });
      }

      // Send file path also for heatmap & AV-check
      result.file_path = uploadPath;

      res.json(result);
    });
  } catch (error) {
    res.status(500).json({ error: "Server error" });
  }
});

// -------------------------------------------
// REALTIME DETECTION ROUTE
// -------------------------------------------
app.post("/realtime", async (req, res) => {
  try {
    if (!req.files || !req.files.frame) {
      return res.status(400).json({ error: "No frame uploaded" });
    }

    const frame = req.files.frame;
    const framePath = path.join(
      __dirname,
      "uploads",
      `frame_${Date.now()}.jpg`
    );
    await frame.mv(framePath);

    exec(`python inference_frame.py "${framePath}"`, (err, stdout) => {
      if (err) return res.json({ error: "Realtime model error" });

      let result = {};
      try {
        result = JSON.parse(stdout);
      } catch {
        result = { label: "UNKNOWN", confidence: 0 };
      }

      res.json(result);
    });
  } catch (err) {
    res.status(500).json({ error: "Realtime error" });
  }
});

// -------------------------------------------
// DOWNLOAD PDF REPORT
// -------------------------------------------
app.get("/report", (req, res) => {
  exec(`python report_gen.py`, (err) => {
    if (err) return res.json({ error: "Report generation failed" });

    res.download(path.join(__dirname, "deepfake_report.pdf"));
  });
});

// -------------------------------------------
// HEATMAP (GRADCAM)
// -------------------------------------------
app.post("/heatmap", (req, res) => {
  const file = req.body.filepath;
  if (!file) return res.json({ error: "No file path provided" });

  const out = `heatmap_${Date.now()}.jpg`;

  exec(`python gradcam.py "${file}" "${out}"`, (err) => {
    if (err) return res.json({ error: "Heatmap generation failed" });

    res.json({ image_path: out });
  });
});

// -------------------------------------------
// AV-CHECK ROUTE
// -------------------------------------------
app.post("/avcheck", (req, res) => {
  const file = req.body.filepath;
  if (!file) return res.json({ error: "No file path provided" });

  exec(`python av_check.py "${file}"`, (err, stdout) => {
    if (err) return res.json({ error: "AV-check failed" });

    res.json(JSON.parse(stdout));
  });
});

// -------------------------------------------
// START SERVER (ONLY ONCE!)
// -------------------------------------------
const PORT = 5000;
app.listen(PORT, () => {
  console.log(`🔥 Backend running at http://127.0.0.1:${PORT}`);
});
