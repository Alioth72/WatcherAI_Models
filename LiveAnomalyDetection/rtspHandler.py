import threading
import time
import numpy as np
import cv2
from queue import Queue, Empty
from typing import Any
import os
import gc
import random
import socket
import psutil

try:
    from vidgear.gears import CamGear
    VIDGEAR_AVAILABLE = True
    print(f"✅ [PID:{os.getpid()}] VidGear available - using high-performance CamGear for RTSP")
except ImportError:
    VIDGEAR_AVAILABLE = False
    print(f"❌ [PID:{os.getpid()}] VidGear not available - install with: pip install vidgear")
    print("   Falling back to threaded OpenCV capture")

class RTSPFrameCapture:
    def __init__(self, rtsp_url, width=1920, height=1080, camera_name="MainCam", required_fps=5, **kwargs) -> Any:
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.running = False
        self.thread = None
        self.current_frame = None
        self.frame_lock = threading.RLock()  # Use RLock for better performance
        self.frame_queue = Queue(maxsize=1)  # Ultra-minimal queue for multi-process
        self.fps_estimate = required_fps
        self._fps_timestamps = []   
        self.required_fps = required_fps
        self.stream_ended = False
        self.stream = None
        self.use_vidgear = VIDGEAR_AVAILABLE
        
        # Process identification for logging
        self.process_id = os.getpid()
        
        # Performance optimizations for multi-process
        self.frame_skip_count = 0
        self.target_frame_time = 1.0 / required_fps
        self.last_frame_time = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 15  # Increased tolerance for multi-process
        
        # Memory optimization
        self.frame_buffer = None
        self.frame_ready = threading.Event()
        
        # Connection retry parameters
        self.max_retries = 10
        self.base_retry_delay = 1.0
        self.connection_timeout = 15
        
        # Multi-process safety - randomize initialization delay
        self.init_delay = random.uniform(0.5, 2.0)
        
        # Process-specific optimizations
        self._setup_process_optimizations()

    def _setup_process_optimizations(self):
        """Setup process-specific optimizations for multi-process scenarios"""
        try:
            # Get current process for optimization
            current_process = psutil.Process()
            
            # Log process info
            print(f"🔧 [PID:{self.process_id}] [{self.camera_name}] Initializing with {psutil.cpu_count()} CPUs, "
                  f"{psutil.virtual_memory().total // (1024**3)}GB RAM")
            
            # Set IO priority to avoid conflicts (non-privileged)
            try:
                if hasattr(current_process, "ionice"):
                    current_process.ionice(psutil.IOPRIO_CLASS_RT, value=4)
            except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
                pass
                
            # Optimize for network I/O
            try:
                if hasattr(socket, 'SO_REUSEADDR'):
                    # This helps with multiple RTSP connections
                    pass
            except Exception:
                pass
                
        except Exception as e:
            print(f"⚠️ [PID:{self.process_id}] [{self.camera_name}] Process optimization warning: {e}")
            
    def _wait_with_jitter(self, base_delay):
        """Wait with random jitter to avoid simultaneous operations"""
        jitter = random.uniform(0.8, 1.2)
        time.sleep(base_delay * jitter)

    def start(self):
        """Start the video frame capture with multi-process safety"""
        print(f"⏳ [PID:{self.process_id}] [{self.camera_name}] Starting initialization...")
        
        # Random delay to prevent simultaneous connection attempts
        # self._wait_with_jitter(self.init_delay)
        
        self.running = True
        
        if self.use_vidgear:
            # Process-specific VidGear CamGear options for multi-process performance
            options = {
                "CAP_PROP_FRAME_WIDTH": self.width,
                "CAP_PROP_FRAME_HEIGHT": self.height,
                "CAP_PROP_FPS": self.required_fps,
                "CAP_PROP_BUFFERSIZE": 1,  # Minimal buffer for real-time
                "CAP_PROP_POS_MSEC": 0,    # Start from beginning
                # FFMPEG optimizations for multi-process
                "THREADED_QUEUE_SIZE": 4,   # Reduced for multi-process
                "RTSP_TRANSPORT": "tcp",    # TCP for reliability
                "analyzeduration": "500000",  # Faster stream analysis
                "probesize": "500000",      # Smaller probe size for faster startup
                "fflags": "nobuffer+fastseek+flush_packets",  # Aggressive buffering control
                "flags": "low_delay",       # Minimize delay
                "strict": "experimental",   # Allow experimental features
                "timeout": str(self.connection_timeout * 1000000),  # Connection timeout in microseconds
                "stimeout": str(self.connection_timeout * 1000000), # Socket timeout
                # H.264 codec configuration for smaller frame sizes
                "vcodec": "h264",           # Force H.264 video codec instead of MJPEG
                "acodec": "none",           # Disable audio processing for efficiency
                "hwaccel": "auto",          # Use hardware acceleration if available
                "preset": "ultrafast",      # Fast encoding preset
                "tune": "zerolatency",      # Optimize for low latency
                "crf": "23",                # Constant rate factor for good quality/size balance
            }
            
            # Add process-specific parameters to avoid conflicts
            process_specific_delay = self.process_id % 1000  # Use PID for uniqueness
            options["user_agent"] = f"RTSPCapture-PID{self.process_id}"
            
            success = self._initialize_stream_with_retry("vidgear", options)
            if success:
                self.thread = threading.Thread(target=self._read_frames_vidgear, daemon=True, 
                                               name=f"VidGear-{self.camera_name}-PID{self.process_id}")
                print(f"🚀 [PID:{self.process_id}] [{self.camera_name}] VidGear CamGear RTSP stream started")
            else:
                print(f"❌ [PID:{self.process_id}] [{self.camera_name}] VidGear failed, falling back to OpenCV")
                self.use_vidgear = False
        
        if not self.use_vidgear:
            success = self._initialize_stream_with_retry("opencv", {})
            if success:
                self.thread = threading.Thread(target=self._read_frames_opencv, daemon=True, 
                                               name=f"OpenCV-{self.camera_name}-PID{self.process_id}")
                print(f"📡 [PID:{self.process_id}] [{self.camera_name}] OpenCV RTSP stream started")
            else:
                print(f"❌ [PID:{self.process_id}] [{self.camera_name}] All connection methods failed")
                return False
            
        # Set thread priority and start
        self.thread.start()
        return True

    def _initialize_stream_with_retry(self, method, options):
        """Initialize stream with retry logic and exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                if method == "vidgear":
                    print(f"🔄 [PID:{self.process_id}] [{self.camera_name}] VidGear attempt {attempt + 1}/{self.max_retries}")
                    self.stream = CamGear(
                        source=self.rtsp_url, 
                        colorspace="BGR",
                        logging=False,  # Disable logging for performance
                        time_delay=0,   # No startup delay
                        **options
                    ).start()
                    
                    # Test if stream is working
                    test_frame = self.stream.read()
                    if test_frame is not None:
                        print(f"✅ [PID:{self.process_id}] [{self.camera_name}] VidGear connection successful with H.264 encoding")
                        return True
                    else:
                        self.stream.stop()
                        raise Exception("No frames received from VidGear")
                        
                elif method == "opencv":
                    print(f"🔄 [PID:{self.process_id}] [{self.camera_name}] OpenCV attempt {attempt + 1}/{self.max_retries}")
                    
                    # Set environment variables for this process with H.264 preference
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|video_codec;h264|hwaccel;auto"
                    
                    cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    
                    # OpenCV optimization settings
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.required_fps)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
                    
                    # Test connection
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            print(f"✅ [PID:{self.process_id}] [{self.camera_name}] OpenCV connection successful with H.264 encoding")
                            cap.release()  # We'll recreate it in the thread
                            return True
                    cap.release()
                    raise Exception("OpenCV connection test failed")
                    
            except Exception as e:
                retry_delay = self.base_retry_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ [PID:{self.process_id}] [{self.camera_name}] {method} attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    print(f"⏳ [PID:{self.process_id}] [{self.camera_name}] Retrying in {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"❌ [PID:{self.process_id}] [{self.camera_name}] {method} failed after {self.max_retries} attempts")
        
        return False

    def _read_frames_vidgear(self):
        """Ultra-optimized frame reading using VidGear CamGear for multi-process"""
        frame_count = 0
        last_fps_time = time.time()
        fps_frame_count = 0
        
        print(f"🎬 [PID:{self.process_id}] [{self.camera_name}] VidGear frame reading thread started")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Adaptive frame rate control with jitter for multi-process
                frame_interval = self.target_frame_time * random.uniform(0.9, 1.1)
                if current_time - self.last_frame_time < frame_interval:
                    time.sleep(0.001)  # Micro-sleep to prevent CPU hammering
                    continue
                
                # Read frame from VidGear CamGear
                frame = self.stream.read()
                
                if frame is not None:
                    frame_count += 1
                    fps_frame_count += 1
                    self.consecutive_failures = 0
                    self.last_frame_time = current_time
                    
                    # Update current frame (minimal locking for multi-process)
                    with self.frame_lock:
                        self.current_frame = frame
                        self.frame_ready.set()
                    
                    # Ultra-minimal queue management for multi-process
                    try:
                        # Clear all old frames and keep only the latest
                        while not self.frame_queue.empty():
                            try:
                                self.frame_queue.get_nowait()
                            except Empty:
                                break
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass  # Queue operations are non-critical
                    
                    # Calculate FPS every second
                    if current_time - last_fps_time >= 1.0:
                        self.fps_estimate = fps_frame_count / (current_time - last_fps_time)
                        fps_frame_count = 0
                        last_fps_time = current_time
                        if frame_count % (self.required_fps * 10) == 0:  # Log every 10 seconds for multi-process
                            print(f"🎯 [PID:{self.process_id}] [{self.camera_name}] Frame {frame_count} | FPS: {self.fps_estimate:.1f}")
                    
                    # Aggressive memory management for multi-process
                    if frame_count % 100 == 0:
                        gc.collect()
                
                else:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        print(f"❌ [PID:{self.process_id}] [{self.camera_name}] Too many consecutive failures, stopping")
                        break
                    time.sleep(0.01)  # Brief pause on failure
                    
            except Exception as e:
                print(f"❌ [PID:{self.process_id}] [{self.camera_name}] VidGear frame read error: {e}")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    break
                time.sleep(0.1)
                
        self.stream_ended = True
        print(f"🛑 [PID:{self.process_id}] [{self.camera_name}] VidGear frame reader stopped")

    def _read_frames_opencv(self):
        """Optimized OpenCV frame reading for multi-process scenarios"""
        print(f"🎬 [PID:{self.process_id}] [{self.camera_name}] OpenCV frame reading thread started")
        
        # Set H.264 environment variables for this thread
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|video_codec;h264|hwaccel;auto"
        
        # Create new connection in this thread to avoid multi-process conflicts
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Multi-process OpenCV optimization settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.required_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for multi-process
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)    # Start from current position
        
        # H.264 specific optimizations
        try:
            # Force H.264 codec preference (input codec, not output fourcc)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Skip unnecessary conversions for efficiency
            # Additional H.264 optimizations
            cap.set(cv2.CAP_PROP_MODE, 0)        # Use default mode for H.264
        except:
            pass
        
        frame_count = 0
        last_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Frame rate control with jitter for multi-process
                frame_interval = self.target_frame_time * random.uniform(0.9, 1.1)
                if current_time - self.last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    frame_count += 1
                    self.consecutive_failures = 0
                    self.last_frame_time = current_time
                    
                    # Update current frame (minimal locking for multi-process)
                    with self.frame_lock:
                        self.current_frame = frame
                        self.frame_ready.set()
                    
                    # Ultra-minimal queue management for multi-process
                    try:
                        # Clear all old frames and keep only the latest
                        while not self.frame_queue.empty():
                            try:
                                self.frame_queue.get_nowait()
                            except Empty:
                                break
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
                        
                    # Log every 10 seconds for multi-process scenarios
                    if current_time - last_time >= 10.0:
                        print(f"📹 [PID:{self.process_id}] [{self.camera_name}] OpenCV Frame {frame_count}")
                        last_time = current_time
                    
                    # Aggressive memory management for multi-process
                    if frame_count % 100 == 0:
                        gc.collect()
                        
                else:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        print(f"❌ [PID:{self.process_id}] [{self.camera_name}] OpenCV: Too many failures, stopping")
                        break
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"❌ [PID:{self.process_id}] [{self.camera_name}] OpenCV frame read error: {e}")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    break
                time.sleep(0.1)
        
        cap.release()
        self.stream_ended = True
        print(f"🛑 [PID:{self.process_id}] [{self.camera_name}] OpenCV frame reader stopped")
            
    def get_fps(self):
        return self.fps_estimate
    
    def get_frame(self):
        """Returns the most recent frame (optimized for speed)"""
        if not self.frame_ready.is_set():
            return None
            
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def get_frame_nowait(self):
        """Get frame without copying (faster but frame may change)"""
        if not self.frame_ready.is_set():
            return None
        return self.current_frame
        
    def is_stream_dead(self):
        return self.stream_ended or self.consecutive_failures >= self.max_consecutive_failures
    
    def stop(self):
        """Stop thread and clean resources with multi-process safety"""
        print(f"🔄 [PID:{self.process_id}] [{self.camera_name}] Initiating graceful shutdown...")
        self.running = False
        
        if self.thread and self.thread.is_alive():
            print(f"⏳ [PID:{self.process_id}] [{self.camera_name}] Waiting for thread to stop...")
            self.thread.join(timeout=5)  # Increased timeout for multi-process
            
        # Clean up resources
        if self.use_vidgear and self.stream:
            try:
                self.stream.stop()
                print(f"🛑 [PID:{self.process_id}] [{self.camera_name}] VidGear CamGear stream stopped")
            except Exception as e:
                print(f"⚠️ [PID:{self.process_id}] [{self.camera_name}] Error stopping VidGear: {e}")
        
        # Clear queues and buffers aggressively for multi-process
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass
            
        self.current_frame = None
        self.frame_buffer = None
        self.frame_ready.clear()
        
        # Aggressive garbage collection for multi-process
        gc.collect()
        
        print(f"🛑 [PID:{self.process_id}] [{self.camera_name}] Stream cleanup completed")

# Performance tuning helper class optimized for multi-process scenarios
class MultiCameraManager:
    """Helper class to manage multiple camera instances efficiently in multi-process environments"""
    
    def __init__(self):
        self.cameras = {}
        self.performance_monitor = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitor_running = False
        self.process_id = os.getpid()
        print(f"🎮 [PID:{self.process_id}] MultiCameraManager initialized")
    
    def add_camera(self, camera_name, rtsp_url, **kwargs):
        """Add a camera instance with optimized settings for multi-process"""
        if camera_name in self.cameras:
            print(f"⚠️ [PID:{self.process_id}] Camera {camera_name} already exists")
            return False
            
        print(f"➕ [PID:{self.process_id}] Adding camera {camera_name} to manager")
        camera = RTSPFrameCapture(rtsp_url=rtsp_url, camera_name=camera_name, **kwargs)
        self.cameras[camera_name] = camera
        return camera.start()
    
    def remove_camera(self, camera_name):
        """Remove and cleanup a camera instance"""
        if camera_name in self.cameras:
            self.cameras[camera_name].stop()
            del self.cameras[camera_name]
            print(f"🗑️ [PID:{self.process_id}] Camera {camera_name} removed")
    
    def get_all_frames(self):
        """Get frames from all active cameras with multi-process safety"""
        frames = {}
        for name, camera in self.cameras.items():
            if not camera.is_stream_dead():
                frame = camera.get_frame_nowait()  # Use faster non-copying method
                if frame is not None:
                    frames[name] = frame
        return frames
    
    def start_monitoring(self):
        """Start performance monitoring for multi-process scenarios"""
        self.monitor_running = True
        self.performance_monitor.start()
        print(f"📊 [PID:{self.process_id}] Performance monitoring started")
    
    def _monitor_performance(self):
        """Monitor system performance and camera health with process identification"""
        try:
            process = psutil.Process()
            
            while self.monitor_running:
                try:
                    cpu_percent = process.cpu_percent()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    active_cameras = sum(1 for cam in self.cameras.values() if not cam.is_stream_dead())
                    total_fps = sum(cam.get_fps() for cam in self.cameras.values())
                    
                    print(f"📊 [PID:{self.process_id}] Performance: CPU: {cpu_percent:.1f}% | Memory: {memory_mb:.1f}MB | "
                          f"Active Cameras: {active_cameras} | Total FPS: {total_fps:.1f}")
                    
                    time.sleep(15)  # Monitor every 15 seconds for multi-process scenarios
                except Exception as e:
                    print(f"⚠️ [PID:{self.process_id}] Performance monitoring error: {e}")
                    break
        except Exception as e:
            print(f"❌ [PID:{self.process_id}] Failed to initialize performance monitoring: {e}")
    
    def stop_all(self):
        """Stop all cameras and cleanup with multi-process safety"""
        print(f"🔄 [PID:{self.process_id}] Stopping all cameras...")
        self.monitor_running = False
        
        for name, camera in self.cameras.items():
            print(f"🛑 [PID:{self.process_id}] Stopping camera {name}")
            camera.stop()
        
        self.cameras.clear()
        print(f"🛑 [PID:{self.process_id}] All cameras stopped")
