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
        
        # STRICT thresholds - pallet must actually overlap forklift, not just be nearby
        self.iou_threshold = spatial_config.get("pallet_iou_threshold", 0.15)
        self.containment_threshold = spatial_config.get("pallet_containment_threshold", 0.5)
        self.vertical_offset_max = spatial_config.get("vertical_offset_max", 30)  # Tighter
        self.fork_zone_ratio = spatial_config.get("fork_zone_ratio", 0.5)  # Forks in lower 50%
        
        # CRITICAL: Minimum requirements for "carrying" detection
        # These MUST be met regardless of score
        self.min_iou_required = 0.08  # Must have at least 8% overlap
        self.min_containment_required = 0.30  # Pallet 30%+ inside forklift bbox
        
        logger.info(
            f"SpatialAnalyzer initialized: "
            f"iou={self.iou_threshold}, "
            f"containment={self.containment_threshold}, "
            f"fork_zone={self.fork_zone_ratio}"
        )
    
    def is_carrying_pallet(
        self,
        forklift: TrackedObject | Detection | BoundingBox,
        pallets: list[Detection]
    ) -> tuple[bool, float, Optional[Detection]]:
        """
        Determine if forklift is carrying a pallet.
        
        Args:
            forklift: Forklift as TrackedObject, Detection, or BoundingBox.
            pallets: List of pallet detections in current frame.
            
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
        
        if not pallets:
            return False, 0.0, None
        
        # Find best matching pallet with STRICT criteria
        best_pallet: Optional[Detection] = None
        best_score = 0.0
        best_iou = 0.0
        best_containment = 0.0
        
        for pallet in pallets:
            # Compute raw IoU and containment first
            forklift_tuple = forklift_bbox.as_tuple()
            pallet_tuple = pallet.bbox.as_tuple()
            
            iou = compute_iou(forklift_tuple, pallet_tuple)
            containment = compute_containment(pallet_tuple, forklift_tuple)
            
            # STRICT: Skip pallets that don't actually overlap with forklift
            # This prevents nearby stacked pallets from being associated
            if iou < self.min_iou_required:
                continue  # No real overlap - skip this pallet
            
            if containment < self.min_containment_required:
                continue  # Pallet not inside forklift bbox - skip
            
            score = self._compute_association_score(forklift_bbox, pallet.bbox)
            
            if score > best_score:
                best_score = score
                best_pallet = pallet
                best_iou = iou
                best_containment = containment
        
        # STRICT: Determine if carrying - need high score AND real overlap
        is_carrying = (
            best_score > 0.6 and  # Higher threshold
            best_iou >= self.min_iou_required and
            best_containment >= self.min_containment_required
        )
        
        if is_carrying:
            logger.debug(
                f"Forklift carrying pallet: score={best_score:.2f}, iou={best_iou:.2f}, containment={best_containment:.2f}"
            )
        
        return is_carrying, best_score, best_pallet if is_carrying else None
    
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
        
        Pallet should be in the lower portion of forklift bbox (fork zone)
        to be considered "on forks".
        
        Returns:
            Score between 0 and 1.
        """
        # Define fork zone (lower portion of forklift bbox)
        fork_zone_top = forklift_bbox.y1 + (1 - self.fork_zone_ratio) * forklift_bbox.height
        fork_zone_bottom = forklift_bbox.y2
        
        # Check if pallet centroid is in fork zone
        pallet_center_y = pallet_bbox.centroid_y
        
        if pallet_center_y < fork_zone_top:
            # Pallet is above fork zone - unlikely carrying
            return 0.0
        elif pallet_center_y > fork_zone_bottom + self.vertical_offset_max:
            # Pallet is below forklift - ground pallet
            return 0.0
        else:
            # Pallet is in or near fork zone
            # Score based on how centered it is in the zone
            zone_center = (fork_zone_top + fork_zone_bottom) / 2
            distance_from_center = abs(pallet_center_y - zone_center)
            max_distance = (fork_zone_bottom - fork_zone_top) / 2 + self.vertical_offset_max
            
            if max_distance == 0:
                return 1.0
            
            return max(0.0, 1.0 - distance_from_center / max_distance)
    
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
