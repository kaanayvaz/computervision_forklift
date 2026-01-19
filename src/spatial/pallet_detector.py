"""
Spatial logic for pallet-on-forklift detection.

Uses rule-based spatial analysis to determine if a forklift is carrying a pallet
based on:
- IoU (Intersection over Union) between bounding boxes
- Containment ratio (how much of pallet is inside forklift bbox)
- Vertical position (pallet should be in fork zone)
"""

from typing import Optional

from core.entities import Detection, TrackedObject, BoundingBox
from core.utils import get_logger, load_config, compute_iou, compute_containment

logger = get_logger(__name__)


class SpatialAnalyzer:
    """
    Rule-based spatial analyzer for forklift-pallet associations.
    
    Determines if a forklift is carrying a pallet based on spatial relationships
    between their bounding boxes. Uses configurable thresholds for IoU, containment,
    and vertical position.
    
    Args:
        config: Dictionary with spatial rule parameters, or path to rules.yaml.
        
    Configuration Parameters:
        - pallet_iou_threshold: Minimum IoU to consider association (default: 0.3)
        - pallet_containment_threshold: Minimum containment ratio (default: 0.5)
        - vertical_offset_max: Maximum vertical offset in pixels (default: 50)
        - fork_zone_ratio: Lower portion of forklift bbox where forks are (default: 0.4)
        
    Example:
        >>> analyzer = SpatialAnalyzer(config)
        >>> is_carrying, confidence = analyzer.is_carrying_pallet(forklift, pallets)
    """
    
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = {}
        
        # Handle nested config structure
        spatial_config = config.get("spatial", config)
        
        # STRICT thresholds - only detect carrying when pallet is ACTUALLY on forklift
        # Background/ground pallets should NOT trigger false positives
        self.iou_threshold = spatial_config.get("pallet_iou_threshold", 0.15)
        self.containment_threshold = spatial_config.get("pallet_containment_threshold", 0.35)
        self.vertical_offset_max = spatial_config.get("vertical_offset_max", 30)  # Strict vertical limit
        self.fork_zone_ratio = spatial_config.get("fork_zone_ratio", 0.5)  # Lower 50% only
        
        # STRICT: High minimum requirements for "carrying" detection
        # These prevent false positives from nearby ground pallets
        self.min_iou_required = 0.10  # At least 10% overlap required
        self.min_containment_required = 0.30  # At least 30% of pallet inside forklift bbox
        
        # SIZE-BASED FALLBACK - DISABLED to prevent false positives
        # Only enable if you're certain there are no background pallets
        self.size_increase_threshold = spatial_config.get("size_increase_threshold", 0.20)
        self.height_increase_threshold = spatial_config.get("height_increase_threshold", 0.10)
        self.aspect_ratio_change_threshold = 0.25
        self.enable_size_fallback = spatial_config.get("enable_size_fallback", False)  # OFF by default
        self.track_size_history: dict[int, list[tuple[float, float, float]]] = {}
        self.baseline_sizes: dict[int, tuple[float, float]] = {}
        
        logger.info(
            f"SpatialAnalyzer initialized (STRICT): "
            f"iou={self.iou_threshold}, min_iou={self.min_iou_required}, "
            f"containment={self.containment_threshold}, min_containment={self.min_containment_required}, "
            f"fork_zone={self.fork_zone_ratio}"
        )
    
    def is_carrying_pallet(
        self,
        forklift: TrackedObject | Detection | BoundingBox,
        pallets: list[Detection],
        track_id: Optional[int] = None
    ) -> tuple[bool, float, Optional[Detection]]:
        """
        Determine if forklift is carrying a pallet.
        
        Uses two detection methods:
        1. Primary: Pallet detection overlap with forklift bbox
        2. Fallback: Size increase detection (when pallet model misses elevated pallets)
        
        Args:
            forklift: Forklift as TrackedObject, Detection, or BoundingBox.
            pallets: List of pallet detections in current frame.
            track_id: Optional track ID for size-based detection tracking.
            
        Returns:
            Tuple of (is_carrying, confidence, associated_pallet).
            - is_carrying: True if forklift is likely carrying a pallet.
            - confidence: Confidence score 0-1.
            - associated_pallet: The Detection object of the carried pallet, or None.
        """
        # Get forklift bbox
        forklift_bbox = self._get_bbox(forklift)
        if forklift_bbox is None:
            return False, 0.0, None
        
        # Extract track_id from TrackedObject if not provided
        if track_id is None and isinstance(forklift, TrackedObject):
            track_id = forklift.track_id
        
        # PRIMARY DETECTION: Pallet overlap with forklift
        best_pallet: Optional[Detection] = None
        best_score = 0.0
        best_iou = 0.0
        best_containment = 0.0
        
        if pallets:
            for pallet in pallets:
                # Compute raw IoU and containment first
                forklift_tuple = forklift_bbox.as_tuple()
                pallet_tuple = pallet.bbox.as_tuple()
                
                iou = compute_iou(forklift_tuple, pallet_tuple)
                containment = compute_containment(pallet_tuple, forklift_tuple)
                
                # STRICT: Skip pallets that don't actually overlap with forklift
                if iou < self.min_iou_required:
                    continue
                
                if containment < self.min_containment_required:
                    continue
                
                score = self._compute_association_score(forklift_bbox, pallet.bbox)
                
                if score > best_score:
                    best_score = score
                    best_pallet = pallet
                    best_iou = iou
                    best_containment = containment
        
        # Check if primary detection found a pallet with STRICT criteria
        # The pallet must have significant overlap with the forklift bbox
        is_carrying_by_overlap = (
            best_score > 0.4 and  # Higher score threshold
            best_iou >= self.min_iou_required and
            best_containment >= self.min_containment_required
        )
        
        if is_carrying_by_overlap:
            logger.debug(
                f"Forklift carrying pallet (overlap): score={best_score:.2f}, "
                f"iou={best_iou:.2f}, containment={best_containment:.2f}"
            )
            return True, best_score, best_pallet
        
        # NO PROXIMITY FALLBACK - removed because it causes false positives
        # Ground pallets near forklift were being detected as "carried"
        
        # SIZE-BASED FALLBACK - only if enabled (disabled by default)
        if self.enable_size_fallback:
            is_carrying_by_size, size_confidence = self._detect_carrying_by_size(
                forklift_bbox, track_id
            )
            
            if is_carrying_by_size:
                logger.debug(
                    f"Forklift carrying pallet (size-based fallback): confidence={size_confidence:.2f}"
                )
                return True, size_confidence, None
        
        return False, 0.0, None
    
    def _detect_carrying_by_size(
        self,
        forklift_bbox: BoundingBox,
        track_id: Optional[int]
    ) -> tuple[bool, float]:
        """
        Detect pallet carrying based on forklift bbox size and shape changes.
        
        When a forklift picks up a pallet, its detection bbox typically:
        1. Grows in area (includes pallet)
        2. Changes aspect ratio (wider with pallet on forks)
        
        Args:
            forklift_bbox: Current forklift bounding box.
            track_id: Track ID for tracking size history.
            
        Returns:
            Tuple of (is_carrying, confidence).
        """
        if track_id is None:
            return False, 0.0
        
        # Calculate current bbox metrics
        current_area = forklift_bbox.width * forklift_bbox.height
        current_width = forklift_bbox.width
        current_height = forklift_bbox.height
        current_aspect = current_width / max(current_height, 1)  # width/height ratio
        
        # Initialize history if needed
        if track_id not in self.track_size_history:
            self.track_size_history[track_id] = []
        
        # Add current measurement to history (area, width, height)
        self.track_size_history[track_id].append((current_area, current_width, current_height))
        
        # Need at least 2 measurements to establish baseline
        history = self.track_size_history[track_id]
        if len(history) < 2:
            return False, 0.0
        
        # Calculate baseline from first few measurements (assumed to be "empty" state)
        if track_id not in self.baseline_sizes:
            if len(history) >= 3:
                baseline_areas = [h[0] for h in history[:3]]
                baseline_aspects = [h[1] / max(h[2], 1) for h in history[:3]]
                baseline_area = sum(baseline_areas) / 3
                baseline_aspect = sum(baseline_aspects) / 3
            else:
                baseline_area = history[0][0]
                baseline_aspect = history[0][1] / max(history[0][2], 1)
            self.baseline_sizes[track_id] = (baseline_area, baseline_aspect)
        
        baseline_area, baseline_aspect = self.baseline_sizes[track_id]
        
        if baseline_area <= 0:
            return False, 0.0
        
        # Calculate changes from baseline
        size_increase = (current_area - baseline_area) / baseline_area
        aspect_change = abs(current_aspect - baseline_aspect) / max(baseline_aspect, 0.1)
        
        # Detect carrying based on COMBINED signals (more conservative to reduce false positives)
        # Require EITHER: both metrics above threshold, OR one metric significantly above threshold
        is_carrying = False
        confidence = 0.0
        reason = ""
        
        # Check for combined signal (both metrics triggered) - most reliable
        if size_increase >= self.size_increase_threshold and aspect_change >= self.aspect_ratio_change_threshold:
            is_carrying = True
            confidence = min(0.90, 0.6 + size_increase + aspect_change * 0.5)
            reason = f"combined: size={size_increase:.1%}, aspect={aspect_change:.1%}"
        
        # Check for strong size signal (>15% increase is likely carrying)
        elif size_increase >= 0.15:
            is_carrying = True
            confidence = min(0.85, 0.5 + (size_increase - 0.15) * 3)
            reason = f"strong_size_increase={size_increase:.1%}"
        
        # Check for strong aspect change (>30% is likely carrying)
        elif aspect_change >= 0.30:
            is_carrying = True
            confidence = min(0.80, 0.5 + (aspect_change - 0.30) * 1.5)
            reason = f"strong_aspect_change={aspect_change:.1%}"
        
        if is_carrying:
            logger.debug(
                f"Track #{track_id}: Carrying detected by {reason} "
                f"(baseline_area={baseline_area:.0f}, current={current_area:.0f}, "
                f"baseline_aspect={baseline_aspect:.2f}, current_aspect={current_aspect:.2f})"
            )
            return True, confidence
        
        return False, 0.0
    
    def reset_size_tracking(self, track_id: Optional[int] = None) -> None:
        """Reset size tracking for a track or all tracks."""
        if track_id is None:
            self.track_size_history.clear()
            self.baseline_sizes.clear()
        else:
            self.track_size_history.pop(track_id, None)
            self.baseline_sizes.pop(track_id, None)

    
    def _get_bbox(
        self,
        obj: TrackedObject | Detection | BoundingBox
    ) -> Optional[BoundingBox]:
        """Extract BoundingBox from various object types."""
        if isinstance(obj, BoundingBox):
            return obj
        elif isinstance(obj, Detection):
            return obj.bbox
        elif isinstance(obj, TrackedObject):
            return obj.latest_bbox
        return None
    
    def _compute_association_score(
        self,
        forklift_bbox: BoundingBox,
        pallet_bbox: BoundingBox
    ) -> float:
        """
        Compute association score between forklift and pallet.
        
        Score is based on:
        1. IoU overlap (30% weight)
        2. Containment ratio (40% weight)
        3. Vertical position (30% weight)
        
        Returns:
            Score between 0 and 1, where higher = more likely associated.
        """
        forklift_tuple = forklift_bbox.as_tuple()
        pallet_tuple = pallet_bbox.as_tuple()
        
        # 1. IoU score
        iou = compute_iou(forklift_tuple, pallet_tuple)
        iou_score = min(iou / self.iou_threshold, 1.0)
        
        # 2. Containment score (how much of pallet is inside forklift bbox)
        containment = compute_containment(pallet_tuple, forklift_tuple)
        containment_score = min(containment / self.containment_threshold, 1.0)
        
        # 3. Vertical position score
        # Pallet should be in lower portion of forklift bbox (fork zone)
        position_score = self._compute_position_score(forklift_bbox, pallet_bbox)
        
        # Weighted combination
        total_score = (
            0.3 * iou_score +
            0.4 * containment_score +
            0.3 * position_score
        )
        
        return total_score
    
    def _compute_position_score(
        self,
        forklift_bbox: BoundingBox,
        pallet_bbox: BoundingBox
    ) -> float:
        """
        Compute score based on pallet position relative to forklift.
        
        For a pallet to be "on forks", it must be:
        1. Horizontally centered on the forklift (not off to the side)
        2. Vertically within the forklift bbox (not below = ground pallet)
        
        Returns:
            Score between 0 and 1.
        """
        pallet_center_x = pallet_bbox.centroid_x
        pallet_center_y = pallet_bbox.centroid_y
        
        # CHECK 1: Pallet must be horizontally within forklift bbox (with small margin)
        margin_x = forklift_bbox.width * 0.2  # 20% margin
        if not (forklift_bbox.x1 - margin_x <= pallet_center_x <= forklift_bbox.x2 + margin_x):
            return 0.0  # Pallet is too far left or right
        
        # CHECK 2: Pallet center must be INSIDE the forklift bbox vertically
        # This is the key check - ground pallets will have their center BELOW the forklift
        if pallet_center_y > forklift_bbox.y2:
            return 0.0  # Pallet center is below forklift = ground pallet, NOT carried
        
        if pallet_center_y < forklift_bbox.y1:
            return 0.0  # Pallet is above forklift = unlikely to be carried
        
        # Pallet is inside forklift bbox - score based on vertical position
        # Prefer pallets in the lower portion (fork zone)
        fork_zone_top = forklift_bbox.y1 + (1 - self.fork_zone_ratio) * forklift_bbox.height
        
        if pallet_center_y >= fork_zone_top:
            # Pallet is in fork zone - high score
            return 1.0
        else:
            # Pallet is in upper portion - lower score but still valid
            relative_pos = (pallet_center_y - forklift_bbox.y1) / forklift_bbox.height
            return 0.5 + 0.5 * relative_pos  # Score 0.5-1.0 based on position
    
    def get_relative_position(
        self,
        forklift_bbox: BoundingBox,
        pallet_bbox: BoundingBox
    ) -> str:
        """
        Get human-readable position relationship.
        
        Returns:
            One of: "on_forks", "nearby", "above", "below", "none"
        """
        iou = compute_iou(forklift_bbox.as_tuple(), pallet_bbox.as_tuple())
        
        if iou < 0.05:
            # Check if nearby
            distance = self._compute_bbox_distance(forklift_bbox, pallet_bbox)
            if distance < 100:  # pixels
                return "nearby"
            return "none"
        
        # Check vertical relationship
        pallet_center_y = pallet_bbox.centroid_y
        forklift_center_y = forklift_bbox.centroid_y
        
        fork_zone_top = forklift_bbox.y1 + (1 - self.fork_zone_ratio) * forklift_bbox.height
        
        if pallet_center_y < forklift_center_y and pallet_center_y < fork_zone_top:
            return "above"
        elif pallet_center_y > forklift_bbox.y2:
            return "below"
        else:
            # In fork zone with sufficient overlap
            if iou >= self.iou_threshold:
                return "on_forks"
            return "nearby"
    
    def _compute_bbox_distance(
        self,
        bbox1: BoundingBox,
        bbox2: BoundingBox
    ) -> float:
        """Compute distance between bbox centers."""
        cx1, cy1 = bbox1.center
        cx2, cy2 = bbox2.center
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    
    def analyze_frame(
        self,
        forklift_tracks: list[TrackedObject],
        pallet_detections: list[Detection]
    ) -> dict[int, tuple[bool, float, Optional[Detection]]]:
        """
        Analyze spatial relationships for all forklifts in a frame.
        
        Args:
            forklift_tracks: List of forklift TrackedObjects.
            pallet_detections: List of pallet Detections.
            
        Returns:
            Dictionary mapping track_id to (is_carrying, confidence, pallet).
        """
        results = {}
        
        # Copy pallet list so we can remove assigned pallets
        available_pallets = list(pallet_detections)
        
        for track in forklift_tracks:
            is_carrying, confidence, pallet = self.is_carrying_pallet(
                track, available_pallets
            )
            
            results[track.track_id] = (is_carrying, confidence, pallet)
            
            # Remove assigned pallet from available list
            if pallet is not None and pallet in available_pallets:
                available_pallets.remove(pallet)
        
        return results
