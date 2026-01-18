"""
Report generation module.

Generates JSON, CSV, and HTML reports from analytics data.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional

from core.entities import Activity, AnalyticsResult, ForkliftState
from core.utils import get_logger

logger = get_logger(__name__)


class Reporter:
    """
    Generate analytics reports in various formats.
    
    Supports:
    - JSON export for programmatic access
    - CSV export for spreadsheet analysis
    - Text summary for quick viewing
    """
    
    def __init__(self, output_dir: str | Path = "data/outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_json_report(
        self,
        analytics: AnalyticsResult,
        activities: list[Activity],
        video_name: str = "video",
        output_path: Optional[str | Path] = None
    ) -> dict:
        """
        Generate JSON report.
        
        Args:
            analytics: AnalyticsResult with metrics.
            activities: List of activities.
            video_name: Name of source video.
            output_path: Optional output file path.
            
        Returns:
            Report dictionary.
        """
        report = {
            "metadata": {
                "video_name": video_name,
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "summary": {
                "total_duration_seconds": analytics.total_duration_seconds,
                "total_duration_minutes": analytics.total_duration_seconds / 60,
                "forklift_count": analytics.forklift_count,
                "activity_count": analytics.activity_count,
                "utilization_percentage": round(analytics.utilization_percentage, 2),
                "total_active_time_seconds": analytics.total_active_time_seconds,
                "total_idle_time_seconds": analytics.total_idle_time_seconds,
                "cost_of_waste_usd": round(analytics.cost_of_waste, 2)
            },
            "idle_breakdown": analytics.idle_breakdown,
            "activities_by_state": analytics.activities_by_state,
            "activities": [
                {
                    "track_id": a.track_id,
                    "state": a.state.value,
                    "start_frame": a.start_frame,
                    "end_frame": a.end_frame,
                    "duration_seconds": round(a.duration_seconds, 2),
                    "is_value_added": a.is_value_added,
                    "description": a.description,
                    "zone": a.zone
                }
                for a in activities
            ]
        }
        
        if output_path:
            self._save_json(report, output_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = self.output_dir / f"{video_name}_{timestamp}.json"
            self._save_json(report, default_path)
        
        return report
    
    def _save_json(self, data: dict, path: str | Path) -> None:
        """Save dictionary to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"JSON report saved: {path}")
    
    def generate_csv_export(
        self,
        activities: list[Activity],
        output_path: Optional[str | Path] = None,
        video_name: str = "video"
    ) -> Path:
        """
        Export activities to CSV.
        
        Args:
            activities: List of activities.
            output_path: Optional output file path.
            video_name: Name of source video.
            
        Returns:
            Path to CSV file.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"{video_name}_{timestamp}.csv"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "track_id",
                "state",
                "start_frame",
                "end_frame",
                "duration_seconds",
                "is_value_added",
                "description",
                "zone"
            ])
            
            # Data rows
            for activity in activities:
                writer.writerow([
                    activity.track_id,
                    activity.state.value,
                    activity.start_frame,
                    activity.end_frame,
                    round(activity.duration_seconds, 2),
                    activity.is_value_added,
                    activity.description,
                    activity.zone
                ])
        
        logger.info(f"CSV export saved: {output_path}")
        return output_path
    
    def generate_summary(
        self,
        analytics: AnalyticsResult
    ) -> str:
        """
        Generate text summary.
        
        Args:
            analytics: AnalyticsResult with metrics.
            
        Returns:
            Formatted summary string.
        """
        from analytics.metrics import generate_summary_report
        return generate_summary_report(analytics)
    
    def save_summary(
        self,
        analytics: AnalyticsResult,
        output_path: Optional[str | Path] = None,
        video_name: str = "video"
    ) -> Path:
        """
        Save text summary to file.
        
        Args:
            analytics: AnalyticsResult with metrics.
            output_path: Optional output file path.
            video_name: Name of source video.
            
        Returns:
            Path to summary file.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"{video_name}_{timestamp}_summary.txt"
        else:
            output_path = Path(output_path)
        
        summary = self.generate_summary(analytics)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"Summary saved: {output_path}")
        return output_path
