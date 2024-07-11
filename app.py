import streamlit as st
import cv2
import numpy as np
from vidstab import VidStab
import tempfile
import os
import ffmpeg

@st.cache_data
def stabilize_video(input_path):
    stabilizer = VidStab()
    vid = cv2.VideoCapture(input_path)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    
    stabilized_frames = []
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, border_type='black')
        stabilized_frames.append(stabilized_frame)
    
    max_left = max_right = max_top = max_bottom = 0
    for frame in stabilized_frames:
        non_black = np.where(frame != 0)
        if len(non_black[0]) > 0 and len(non_black[1]) > 0:
            top, bottom = non_black[0].min(), non_black[0].max()
            left, right = non_black[1].min(), non_black[1].max()
            max_top = max(max_top, top)
            max_bottom = min(max_bottom, bottom) if max_bottom != 0 else bottom
            max_left = max(max_left, left)
            max_right = min(max_right, right) if max_right != 0 else right
    
    new_height = max_bottom - max_top
    new_width = max_right - max_left
    
    if new_height <= 0 or new_width <= 0:
        st.warning("Stabilization resulted in invalid dimensions. Using original dimensions.")
        new_height, new_width = height, width
        max_top, max_bottom, max_left, max_right = 0, height, 0, width
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height))
    
    for frame in stabilized_frames:
        cropped_frame = frame[max_top:max_bottom, max_left:max_right]
        if cropped_frame.shape[:2] != (new_height, new_width):
            cropped_frame = cv2.resize(cropped_frame, (new_width, new_height))
        out.write(cropped_frame)
    
    vid.release()
    out.release()
    # cv2.destroyAllWindows()
    
    # Convert to web-compatible format and include audio
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    stream = ffmpeg.input(temp_output)
    audio = ffmpeg.input(input_path).audio
    stream = ffmpeg.output(stream, audio, output_path, vcodec='libx264', acodec='aac')
    ffmpeg.run(stream, overwrite_output=True)
    
    os.unlink(temp_output)
    
    return output_path, new_width, new_height

st.title("Video Stabilization App")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# Button for editing
if st.button('Edit Video'):
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    input_path = tfile.name
    
    st.text("Processing video... This may take a while.")
    output_path, new_width, new_height = stabilize_video(input_path)
    st.text(f"Video stabilized! New dimensions: {new_width}x{new_height}")
    
    # Display videos side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Video")
        st.video(input_path)
    
    with col2:
        st.subheader("Stabilized Video")
        st.video(output_path)
    
    # Provide download button
    with open(output_path, "rb") as file:
        btn = st.download_button(
            label="Download stabilized video",
            data=file,
            file_name="stabilized_video.mp4",
            mime="video/mp4"
        )
    
    # Clean up temporary files
    os.unlink(input_path)
    os.unlink(output_path)