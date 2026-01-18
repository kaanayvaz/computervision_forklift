"""
Unit tests for VideoReader.

Tests cover:
- Video opening and metadata extraction
- Frame iteration with skip
- Error handling
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestVideoReader:
    """Tests for VideoReader class."""
    
    @pytest.fixture
    def sample_video(self, tmp_path):
        """Create a sample test video file."""
        video_path = tmp_path / "test_video.mp4"
        
        # Create a small test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        
        # Write 30 frames (1 second)
        for i in range(30):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 8  # Varying blue channel
            writer.write(frame)
        
        writer.release()
        return video_path
    
    def test_video_reader_opens_file(self, sample_video):
        """Test VideoReader opens video file successfully."""
        from video.reader import VideoReader
        
        reader = VideoReader(sample_video)
        
        assert reader.total_frames == 30
        assert reader.fps == pytest.approx(30.0, rel=0.1)
        assert reader.resolution == (640, 480)
        
        reader.release()
    
    def test_video_reader_file_not_found(self, tmp_path):
        """Test VideoReader raises error for missing file."""
        from video.reader import VideoReader
        
        with pytest.raises(FileNotFoundError):
            VideoReader(tmp_path / "nonexistent.mp4")
    
    def test_video_reader_iteration(self, sample_video):
        """Test VideoReader yields frames correctly."""
        from video.reader import VideoReader
        
        reader = VideoReader(sample_video, frame_skip=1)
        frames_read = 0
        
        for frame_id, frame in reader:
            assert frame.shape == (480, 640, 3)
            frames_read += 1
        
        assert frames_read == 30
        reader.release()
    
    def test_video_reader_frame_skip(self, sample_video):
        """Test VideoReader respects frame_skip parameter."""
        from video.reader import VideoReader
        
        reader = VideoReader(sample_video, frame_skip=3)
        frame_ids = [fid for fid, _ in reader]
        
        # Should read frames 0, 3, 6, 9, ..., 27
        assert frame_ids == list(range(0, 30, 3))
        reader.release()
    
    def test_video_reader_max_resolution(self, sample_video):
        """Test VideoReader resizes frames to max_resolution."""
        from video.reader import VideoReader
        
        reader = VideoReader(sample_video, max_resolution=(320, 240))
        
        for frame_id, frame in reader:
            h, w = frame.shape[:2]
            assert w <= 320
            assert h <= 240
            break  # Just check first frame
        
        reader.release()
    
    def test_video_reader_context_manager(self, sample_video):
        """Test VideoReader works as context manager."""
        from video.reader import VideoReader
        
        with VideoReader(sample_video) as reader:
            metadata = reader.get_metadata()
            assert "fps" in metadata
            assert "total_frames" in metadata
    
    def test_video_reader_metadata(self, sample_video):
        """Test VideoReader returns correct metadata."""
        from video.reader import VideoReader
        
        with VideoReader(sample_video) as reader:
            metadata = reader.get_metadata()
            
            assert metadata["fps"] == pytest.approx(30.0, rel=0.1)
            assert metadata["width"] == 640
            assert metadata["height"] == 480
            assert metadata["total_frames"] == 30
            assert metadata["duration_seconds"] == pytest.approx(1.0, rel=0.1)
    
    def test_video_reader_read_specific_frame(self, sample_video):
        """Test reading a specific frame by ID."""
        from video.reader import VideoReader
        
        with VideoReader(sample_video) as reader:
            frame = reader.read_frame(15)
            
            assert frame is not None
            assert frame.shape == (480, 640, 3)


class TestVideoWriter:
    """Tests for VideoWriter class."""
    
    def test_video_writer_creates_file(self, tmp_path):
        """Test VideoWriter creates output file."""
        from video.reader import VideoWriter
        
        output_path = tmp_path / "output.mp4"
        
        with VideoWriter(output_path, fps=30.0, resolution=(640, 480)) as writer:
            # Write some frames
            for _ in range(10):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                writer.write(frame)
        
        assert output_path.exists()
        assert writer.frame_count == 10
