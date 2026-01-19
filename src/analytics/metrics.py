"""
Analytics and metrics calculation module.

Calculates utilization metrics, idle time analysis, and cost of waste.
"""

from dataclasses import dataclass, field
from typing import Optional

from core.entities import Activity, AnalyticsResult, ForkliftState
from core.utils import get_logger

logger = get_logger(__name__)


def calculate_utilization(
    activities: list[Activity],
    total_duration_seconds: Optional[float] = None,
    forklift_count: Optional[int] = None
) -> float:
    """
    Calculate forklift utilization percentage.
    
    Utilization = (active time / total available time) * 100
    
    For multiple forklifts, we calculate:
    - Per-forklift utilization based on each forklift's activity time
    - Fleet average utilization
    
    Args:
        activities: List of Activity objects.
        total_duration_seconds: Optional total duration (auto-calculated if None).
        forklift_count: Optional number of forklifts (auto-detected from activities if None).
        
    Returns:
        Utilization percentage 0-100.
    """
    if not activities:
        return 0.0
    
    # Group activities by track_id (forklift)
    activities_by_track: dict[int, list[Activity]] = {}
    for activity in activities:
        if activity.track_id not in activities_by_track:
            activities_by_track[activity.track_id] = []
        activities_by_track[activity.track_id].append(activity)
    
    # Detect forklift count from activities if not provided
    if forklift_count is None:
        forklift_count = len(activities_by_track)
    
    if forklift_count == 0:
        return 0.0
    
    # Calculate total duration from activities if not provided
    if total_duration_seconds is None:
        # Use the span from earliest to latest activity
        all_times = []
        for activity in activities:
            all_times.append(activity.start_frame)
            all_times.append(activity.end_frame)
        if all_times:
            # Rough estimate based on frames (assumes fps is embedded in activities)
            total_duration_seconds = sum(a.duration_seconds for a in activities)
        else:
            total_duration_seconds = 0.0
    
    if total_duration_seconds == 0:
        return 0.0
    
    # Calculate per-forklift utilization
    per_forklift_utilization = []
    
    for track_id, track_activities in activities_by_track.items():
        # Calculate active time for this forklift
        active_time = 0.0
        total_time = 0.0
        
        for activity in track_activities:
            total_time += activity.duration_seconds
            if activity.state in [
                ForkliftState.MOVING_EMPTY,
                ForkliftState.MOVING_LOADED,
                ForkliftState.LOADING,
                ForkliftState.UNLOADING
            ]:
                active_time += activity.duration_seconds
        
        # Per-forklift utilization = active_time / forklift's_tracked_time
        if total_time > 0:
            utilization = (active_time / total_time) * 100
            per_forklift_utilization.append(utilization)
            logger.debug(f"Track #{track_id}: {active_time:.1f}s active / {total_time:.1f}s total = {utilization:.1f}%")
    
    # Fleet average utilization
    if per_forklift_utilization:
        fleet_utilization = sum(per_forklift_utilization) / len(per_forklift_utilization)
    else:
        fleet_utilization = 0.0
    
    logger.info(f"Fleet utilization: {fleet_utilization:.1f}% (avg of {len(per_forklift_utilization)} forklifts)")
    
    return fleet_utilization


def calculate_idle_time(activities: list[Activity]) -> float:
    """
    Calculate total idle time in seconds.
    
    Args:
        activities: List of Activity objects.
        
    Returns:
        Total idle time in seconds.
    """
    return sum(
        a.duration_seconds
        for a in activities
        if a.state == ForkliftState.IDLE
    )


def calculate_cost_of_waste(
    idle_time_seconds: float,
    cost_per_hour: float = 75.0
) -> float:
    """
    Calculate cost of waste from idle time.
    
    Args:
        idle_time_seconds: Total idle time in seconds.
        cost_per_hour: Cost per hour of idle time (USD).
        
    Returns:
        Cost of waste in USD.
    """
    idle_hours = idle_time_seconds / 3600
    return idle_hours * cost_per_hour


def calculate_productive_ratio(activities: list[Activity]) -> float:
    """
    Calculate ratio of productive (loaded) movement vs total movement.
    
    Args:
        activities: List of Activity objects.
        
    Returns:
        Ratio 0-1 of loaded movement time.
    """
    loaded_time = sum(
        a.duration_seconds
        for a in activities
        if a.state == ForkliftState.MOVING_LOADED
    )
    
    total_moving = sum(
        a.duration_seconds
        for a in activities
        if a.state in [ForkliftState.MOVING_EMPTY, ForkliftState.MOVING_LOADED]
    )
    
    if total_moving == 0:
        return 0.0
    
    return loaded_time / total_moving


def generate_analytics(
    activities: list[Activity],
    total_duration_seconds: Optional[float] = None,
    cost_per_hour: float = 75.0,
    forklift_count: int = 1
) -> AnalyticsResult:
    """
    Generate comprehensive analytics from activities.
    
    Args:
        activities: List of Activity objects.
        total_duration_seconds: Total video/session duration.
        cost_per_hour: Cost per hour of idle time.
        forklift_count: Number of forklifts analyzed.
        
    Returns:
        AnalyticsResult with all metrics.
    """
    if total_duration_seconds is None:
        total_duration_seconds = sum(a.duration_seconds for a in activities)
    
    idle_time = calculate_idle_time(activities)
    
    # Calculate actual active time from non-idle activities
    active_time = sum(
        a.duration_seconds for a in activities
        if a.state in [
            ForkliftState.MOVING_EMPTY,
            ForkliftState.MOVING_LOADED,
            ForkliftState.LOADING,
            ForkliftState.UNLOADING
        ]
    )
    
    # Calculate utilization with forklift count for proper scaling
    utilization = calculate_utilization(activities, total_duration_seconds, forklift_count)
    cost_of_waste = calculate_cost_of_waste(idle_time, cost_per_hour)
    
    # Breakdown by state
    activities_by_state = {}
    for activity in activities:
        state_name = activity.state.value
        activities_by_state[state_name] = activities_by_state.get(state_name, 0) + 1
    
    # Idle breakdown categories
    idle_breakdown = categorize_idle_activities(activities)
    
    return AnalyticsResult(
        total_duration_seconds=total_duration_seconds,
        total_active_time_seconds=active_time,
        total_idle_time_seconds=idle_time,
        forklift_count=forklift_count,
        activity_count=len(activities),
        utilization_percentage=utilization,
        cost_of_waste=cost_of_waste,
        cost_per_idle_hour=cost_per_hour,
        activities_by_state=activities_by_state,
        idle_breakdown=idle_breakdown
    )


def categorize_idle_activities(
    activities: list[Activity]
) -> dict[str, float]:
    """
    Categorize idle activities by type.
    
    Categories:
    - idle_waiting: Standard idle time (< 60s)
    - extended_idle: Long idle periods (60s-300s)
    - significant_idle: Very long idle (> 300s)
    
    Returns:
        Dictionary mapping category to total seconds.
    """
    categories = {
        "idle_waiting": 0.0,
        "extended_idle": 0.0,
        "significant_idle": 0.0
    }
    
    for activity in activities:
        if activity.state != ForkliftState.IDLE:
            continue
        
        duration = activity.duration_seconds
        
        if duration >= 300:  # > 5 minutes
            categories["significant_idle"] += duration
        elif duration >= 60:  # 1-5 minutes
            categories["extended_idle"] += duration
        else:  # < 1 minute
            categories["idle_waiting"] += duration
    
    return categories


def generate_summary_report(analytics: AnalyticsResult) -> str:
    """
    Generate human-readable summary report.
    
    Args:
        analytics: AnalyticsResult object.
        
    Returns:
        Formatted summary string.
    """
    lines = [
        "=" * 60,
        "FORKLIFT ANALYTICS SUMMARY",
        "=" * 60,
        "",
        f"Duration Analyzed:    {analytics.total_duration_seconds / 60:.1f} minutes",
        f"Forklifts Tracked:    {analytics.forklift_count}",
        f"Activities Detected:  {analytics.activity_count}",
        "",
        "-" * 40,
        "UTILIZATION",
        "-" * 40,
        f"Utilization Rate:     {analytics.utilization_percentage:.1f}%",
        f"Active Time:          {analytics.total_active_time_seconds / 60:.1f} minutes",
        f"Idle Time:            {analytics.total_idle_time_seconds / 60:.1f} minutes",
        "",
        "-" * 40,
        "COST ANALYSIS",
        "-" * 40,
        f"Cost per Idle Hour:   ${analytics.cost_per_idle_hour:.2f}",
        f"Estimated Waste Cost: ${analytics.cost_of_waste:.2f}",
        "",
        "-" * 40,
        "IDLE BREAKDOWN",
        "-" * 40,
    ]
    
    for category, seconds in analytics.idle_breakdown.items():
        lines.append(f"  {category}: {seconds / 60:.1f} minutes")
    
    lines.extend([
        "",
        "-" * 40,
        "ACTIVITIES BY STATE",
        "-" * 40,
    ])
    
    for state, count in analytics.activities_by_state.items():
        lines.append(f"  {state}: {count} activities")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
