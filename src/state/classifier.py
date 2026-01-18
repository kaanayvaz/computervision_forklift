"""
Rule-based state classification for forklifts.

Classifies forklift operational state based on:
- Velocity (moving vs. idle)
- Pallet carrying status
- Temporal smoothing for stability
"""

from collections import deque
from typing import Optional

from core.entities import TrackedObject, ForkliftState
from core.utils import get_logger

logger = get_logger(__name__)


class StateClassifier:
    """
    Rule-based forklift state classifier.
    
    Determines operational state based on velocity and pallet status.
    Uses temporal smoothing to prevent rapid state flickering.
    
    State Classification Rules:
    - IDLE: velocity < threshold AND not carrying pallet
    - MOVING_EMPTY: velocity >= threshold AND not carrying pallet
    - MOVING_LOADED: velocity >= threshold AND carrying pallet
    - LOADING: velocity < threshold AND pallet transitioning to carried
    - UNLOADING: velocity < threshold AND pallet transitioning from carried
    
    Args:
        idle_threshold: Velocity threshold (pixels/frame) for idle detection.
        confirmation_frames: Consecutive frames needed to confirm state change.
        hysteresis_factor: Multiplier for state transition thresholds.
    """
    
    def __init__(
        self,
        idle_threshold: float = 2.0,
        confirmation_frames: int = 5,
        hysteresis_factor: float = 1.5
    ):
        self.idle_threshold = idle_threshold
        self.confirmation_frames = confirmation_frames
        self.hysteresis_factor = hysteresis_factor
        
        # Per-track state history for smoothing
        self._state_history: dict[int, deque] = {}
        self._confirmed_state: dict[int, ForkliftState] = {}
        self._pallet_history: dict[int, deque] = {}  # For loading/unloading detection
        
        logger.info(
            f"StateClassifier initialized: "
            f"idle_threshold={idle_threshold}, "
            f"confirmation={confirmation_frames}"
        )
    
    def classify(
        self,
        track: TrackedObject,
        velocity: float,
        is_carrying_pallet: bool
    ) -> ForkliftState:
        """
        Classify current operational state of forklift.
        
        Args:
            track: TrackedObject being classified.
            velocity: Current velocity in pixels/frame.
            is_carrying_pallet: Whether forklift is carrying a pallet.
            
        Returns:
            ForkliftState enum value.
        """
        track_id = track.track_id
        
        # Get raw state based on current values
        raw_state = self._get_raw_state(velocity, is_carrying_pallet)
        
        # Check for loading/unloading transitions
        transition_state = self._check_pallet_transition(
            track_id, is_carrying_pallet, velocity
        )
        if transition_state is not None:
            raw_state = transition_state
        
        # Apply temporal smoothing
        smoothed_state = self._apply_temporal_smoothing(track_id, raw_state)
        
        return smoothed_state
    
    def _get_raw_state(
        self,
        velocity: float,
        is_carrying_pallet: bool
    ) -> ForkliftState:
        """Determine raw state from velocity and pallet status."""
        is_idle = velocity < self.idle_threshold
        
        if is_idle:
            if is_carrying_pallet:
                # Could be waiting with load or about to unload
                return ForkliftState.MOVING_LOADED  # Conservative estimate
            else:
                return ForkliftState.IDLE
        else:
            if is_carrying_pallet:
                return ForkliftState.MOVING_LOADED
            else:
                return ForkliftState.MOVING_EMPTY
    
    def _check_pallet_transition(
        self,
        track_id: int,
        is_carrying_pallet: bool,
        velocity: float
    ) -> Optional[ForkliftState]:
        """Detect loading/unloading transitions."""
        # Initialize pallet history if needed
        if track_id not in self._pallet_history:
            self._pallet_history[track_id] = deque(maxlen=10)
        
        history = self._pallet_history[track_id]
        history.append(is_carrying_pallet)
        
        if len(history) < 3:
            return None
        
        # Check for recent transition while stationary
        is_idle = velocity < self.idle_threshold
        if not is_idle:
            return None
        
        # Look for False -> True transition (loading)
        if not history[-3] and not history[-2] and history[-1]:
            logger.debug(f"Track {track_id}: LOADING detected")
            return ForkliftState.LOADING
        
        # Look for True -> False transition (unloading)
        if history[-3] and history[-2] and not history[-1]:
            logger.debug(f"Track {track_id}: UNLOADING detected")
            return ForkliftState.UNLOADING
        
        return None
    
    def _apply_temporal_smoothing(
        self,
        track_id: int,
        raw_state: ForkliftState
    ) -> ForkliftState:
        """Apply temporal smoothing to prevent state flickering."""
        # Initialize if needed
        if track_id not in self._state_history:
            self._state_history[track_id] = deque(maxlen=self.confirmation_frames * 2)
            self._confirmed_state[track_id] = ForkliftState.UNKNOWN
        
        history = self._state_history[track_id]
        history.append(raw_state)
        
        # Count occurrences of raw_state in recent history
        count = sum(1 for s in history if s == raw_state)
        
        # If we have enough confirmation, update confirmed state
        if count >= self.confirmation_frames:
            self._confirmed_state[track_id] = raw_state
        
        return self._confirmed_state[track_id]
    
    def classify_batch(
        self,
        tracks: list[TrackedObject],
        velocities: dict[int, float],
        pallet_status: dict[int, bool]
    ) -> dict[int, ForkliftState]:
        """
        Classify states for multiple tracks.
        
        Args:
            tracks: List of TrackedObjects.
            velocities: Dictionary mapping track_id to velocity.
            pallet_status: Dictionary mapping track_id to carrying status.
            
        Returns:
            Dictionary mapping track_id to ForkliftState.
        """
        results = {}
        
        for track in tracks:
            tid = track.track_id
            velocity = velocities.get(tid, 0.0)
            is_carrying = pallet_status.get(tid, False)
            
            state = self.classify(track, velocity, is_carrying)
            results[tid] = state
        
        return results
    
    def get_state_history(
        self,
        track_id: int
    ) -> list[ForkliftState]:
        """Get state history for a track."""
        if track_id in self._state_history:
            return list(self._state_history[track_id])
        return []
    
    def is_idle(
        self,
        track: TrackedObject,
        velocity: float,
        is_carrying_pallet: bool
    ) -> bool:
        """Check if forklift is in idle state."""
        state = self.classify(track, velocity, is_carrying_pallet)
        return state == ForkliftState.IDLE
    
    def is_active(
        self,
        track: TrackedObject,
        velocity: float,
        is_carrying_pallet: bool
    ) -> bool:
        """Check if forklift is in active (moving) state."""
        state = self.classify(track, velocity, is_carrying_pallet)
        return state in [
            ForkliftState.MOVING_EMPTY,
            ForkliftState.MOVING_LOADED,
            ForkliftState.LOADING,
            ForkliftState.UNLOADING
        ]
    
    def clear_track(self, track_id: int) -> None:
        """Clear state history for a track."""
        if track_id in self._state_history:
            del self._state_history[track_id]
        if track_id in self._confirmed_state:
            del self._confirmed_state[track_id]
        if track_id in self._pallet_history:
            del self._pallet_history[track_id]
    
    def reset(self) -> None:
        """Reset all state history."""
        self._state_history.clear()
        self._confirmed_state.clear()
        self._pallet_history.clear()
