# FYP Dashboard - Automatic Number Plate Recognition System

A comprehensive dashboard for monitoring vehicle parking bays with automatic number plate recognition (ANPR) capabilities.

## Features

- Real-time camera feed monitoring
- Automatic vehicle number plate detection and recognition
- Multi-camera support with customizable bay layouts
- Bay occupancy tracking
- Web-based dashboard interface

## Prerequisites

- Python 3.8+
- XAMPP (Apache and MySQL)
- Google Chrome (for vehicle information scraping)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up XAMPP and MySQL

- Download and install [XAMPP](https://www.apachefriends.org/)
- Start Apache and MySQL modules from the XAMPP Control Panel

### 3. Set Up Database

If the `fyp` database doesn't already exist:

1. Open phpMyAdmin (usually at `http://localhost/phpmyadmin`)
2. Create a new database called `fyp`
3. Import the `fyp.sql` file into the `fyp` database:
   - Go to the `fyp` database
   - Click on the **Import** tab
   - Select the `fyp.sql` file from this repository
   - Click **Go**

### 4. Run the Application

Start the backend server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## How to Use the Dashboard

### Accessing the Dashboard

1. Open `index.html` in your web browser

### Initial Setup (Cameras and Bays)

1. Click **Set up bays** in the top right corner
2. Click **Add a camera**
3. **Name the camera** (e.g., "Camera 1")
4. **Provide camera source** (choose one):
   - **Camera Index**: If the camera is directly connected to your computer (0, 1, 2, etc.)
   - **RTSP/URL**: If using a network/IP camera
   - **Test Videos**: For testing purposes, use `test_videos/cam1.mp4` through `test_videos/cam6.mp4` for pre-recorded looping videos

### Adding Bays

1. Select the camera from the dropdown menu
2. Draw the bay bounding box on the camera feed
3. Name the bay on the right side
4. Press **Save**

### Managing Cameras and Bays

- **Delete a bay**: Press the red cross button on the middle right side
- **Delete a camera**: Use the camera management panel

### Monitoring Live Feed

1. Go back to the main dashboard page
2. Select which camera's live feed you want to monitor
3. View all bays and their occupancy statuses on the left side panel

## Project Structure

```
├── main.py                 # FastAPI backend server
├── vision_pipeline.py      # ANPR detection and OCR pipeline
├── debug.py               # Utility scripts for development
├── index.html             # Dashboard web interface
├── fyp.sql                # Database schema and initial data
├── Weights/
│   └── best.pt           # YOLO model weights
└── test_videos/          # Sample videos for testing
    ├── cam1.mp4
    ├── cam2.mp4
    ├── cam3.mp4
    ├── cam4.mp4
    ├── cam5.mp4
    └── cam6.mp4
```

## Technologies Used

- **Backend**: FastAPI (Python web framework)
- **Database**: MySQL
- **Computer Vision**: OpenCV, YOLOv8, EasyOCR
- **Frontend**: HTML/JavaScript
- **Vehicle Info**: SeleniumBase (for web scraping)

## Troubleshooting

### Database Connection Error
- Ensure MySQL is running in XAMPP
- Verify the database name is `fyp`
- Check that the `fyp.sql` file has been imported

### Camera Not Found
- For local cameras, ensure the camera index is correct (usually 0 or 1)
- For IP cameras, verify the RTSP URL is accessible
- Check that your camera/webcam is not being used by another application

### YOLO Model Not Loading
- Ensure `Weights/best.pt` exists in the project directory
- The model will be downloaded automatically on first run if missing

## Support

For issues or questions, please refer to the project documentation or check the browser console for error messages.
