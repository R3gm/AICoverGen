---
title: AICoverGen
emoji: 🚀
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 5.44.0
app_file: app.py
pinned: false
license: mit
---

# AICoverGen  

## Overview  
**AICoverGen** is a WebUI designed to create **AI-powered song covers** with customizable voice models, pitch adjustments, and audio mixing options.  
Built with ❤️ for music enthusiasts and creators, it allows you to transform vocals, generate new covers, and manage models easily.  

| Description | Link |  
| ----------- | ---- |  
| 📙 Colab Notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/R3gm/AICoverGen/blob/main/colab_notebook.ipynb) |  
| 🎉 Repository | [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?style=flat-square&logo=github)](https://github.com/R3gm/AICoverGen) |  
| 🚀 Online DEMO | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/r3gm/AICoverGen) |  

## Key Functions  
1. **Voice Models**  
   - Store models inside the folder: `AICoverGen/rvc_models`.  
   - Use the **Refresh Models 🔁** button after adding new ones.  

2. **Audio Input**  
   - Upload audio files (songs, recordings).  
   - Paste a **YouTube URL** for direct processing.  

3. **Pitch Control**  
   - **Pitch Change (Vocals ONLY):** Adjust pitch for male ↔ female conversions (`+1` / `-1`).  
   - **Overall Pitch Change:** Shift pitch/key of both vocals & instrumentals (semitones).  

4. **Voice Conversion Options**  
   - Apply trained voice models to replace original vocals with AI-generated ones.  

5. **Audio Mixing Options**  
   - Balance vocals and background for best quality results.  

6. **Cover Generation**  
   - Generate AI covers and download the final track.  

## Summary Table  

| Function                  | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| **Voice Models**          | Manage voice models in the `rvc_models` folder.                             |
| **Audio Upload**          | Import audio via file upload or YouTube URL.                                |
| **Pitch Change (Vocals)** | Adjust pitch for gender conversion of vocals only.                          |
| **Overall Pitch Change**  | Modify the pitch/key of both vocals and instrumentals together.              |
| **Voice Conversion**      | Replace original vocals with selected AI voice models.                      |
| **Audio Mixing**          | Fine-tune balance between vocals and instrumentals.                         |
| **Generate & Download**   | Create your AI cover and download the result as an audio file.               |

👉 This is just a fork of [SociallyIneptWeeb/AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen), playing around with some changes and improvements.
