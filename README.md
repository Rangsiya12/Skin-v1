# 🏥 Skin Cancer Detection LINE Bot v8

**AI-Powered Skin Cancer Detection via LINE Bot using YOLOv8**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![LINE Bot SDK](https://img.shields.io/badge/LINE-Bot%20SDK-00C300.svg)](https://github.com/line/line-bot-sdk-python)
[![Railway](https://img.shields.io/badge/Deploy-Railway-purple.svg)](https://railway.app/)

LINE Bot ที่ใช้ AI สำหรับการตรวจจับโรคผิวหนังเบื้องต้น โดยใช้โมเดล YOLOv8 ที่ได้รับการฝึกฝนเฉพาะสำหรับการวิเคราะห์รอยโรคผิวหนัง 3 ประเภท พร้อมแสดงผลด้วย Bounding Box สีสันที่แตกต่างตามระดับความเสี่ยง

## 🎯 Features

### 🔍 **AI Detection Capabilities**
- ตรวจจับโรคผิวหนัง 3 ประเภท:
  - **Melanoma** (เมลาโนมา) - ความเสี่ยงสูง 🔴
  - **Nevus** (เนวัส) - ความเสี่ยงต่ำ 🟢  
  - **Seborrheic Keratosis** (เซบอร์รีอิก เคราโทซิส) - ความเสี่ยงปานกลาง 🟠

### 📱 **LINE Bot Integration**
- รับรูปภาพผ่าน LINE Chat
- ส่งผลลัพธ์พร้อมรูปภาพที่มี Bounding Box
- แสดงความแม่นยำและคำแนะนำเบื้องต้น
- รองรับภาษาไทยและอังกฤษ

### 🎨 **Advanced Visualization**
- **Smart Bounding Box**: กรอบสีแสดงตำแหน่งรอยโรค
- **Adaptive Font Size**: ขนาดตัวอักษรปรับตามขนาดรูปภาพ
- **Color-Coded Results**: สีที่แตกต่างตามระดับความเสี่ยง
- **Text Shadow**: เงาตัวอักษรเพื่อความชัดเจน

### ☁️ **Cloud Deployment**
- Deploy บน Railway Platform
- Auto-scaling และ High Availability
- Webhook สำหรับ LINE Bot API

## 🚀 Quick Start

### 1. **Clone Repository**
bash
git clone https://github.com/Konrawut11/skin-cancer-linebot-v8.git
cd skin-cancer-linebot-v8

### 2. **Install Dependencies**
bash
