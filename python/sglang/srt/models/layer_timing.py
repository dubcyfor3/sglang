"""Global layer timing recorder for profiling."""
import time
from collections import defaultdict
from typing import Dict, Optional

from sglang.srt.dllm.algorithm.utils import AverageMeter


_global_layer_timing_recorder: Optional["LayerTimingRecorder"] = None


class LayerTimingRecorder:
    """Global recorder for per-layer timing statistics."""
    
    def __init__(self):
        self.enabled = False
        self.timings: Dict[str, AverageMeter] = defaultdict(AverageMeter)
        self.current_layer_id: Optional[int] = None
    
    def enable(self):
        """Enable layer timing recording."""
        self.enabled = True
        self.reset()
    
    def disable(self):
        """Disable layer timing recording."""
        self.enabled = False
    
    def reset(self):
        """Reset all timing statistics."""
        self.timings.clear()
        self.current_layer_id = None
    
    def set_layer_id(self, layer_id: int):
        """Set the current layer ID for timing context."""
        self.current_layer_id = layer_id
    
    def record_timing(self, component_name: str, time_ms: float):
        """Record timing for a component."""
        if not self.enabled:
            return
        
        if self.current_layer_id is not None:
            key = f"layer_{self.current_layer_id}_{component_name}"
        else:
            key = component_name
        
        self.timings[key].update(time_ms)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all recorded components."""
        stats = {}
        for key, meter in self.timings.items():
            stats[key] = {
                'avg': meter.avg,
                'sum': meter.sum,
                'count': meter.count,
                'val': meter.val,
            }
        return stats
    
    def get_aggregated_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics aggregated by component type (attention, mlp, moe)."""
        aggregated = defaultdict(AverageMeter)
        
        for key, meter in self.timings.items():
            # Extract component name from key like "layer_0_attention" -> "attention"
            # or "layer_5_mlp" -> "mlp"
            if '_' in key:
                # Try to extract component name (last part after last underscore)
                parts = key.split('_')
                if len(parts) >= 2:
                    # Check if it's a layer-specific key (format: layer_X_component)
                    if parts[0] == 'layer' and len(parts) >= 3:
                        component_name = '_'.join(parts[2:])  # Get everything after layer_X
                    else:
                        # Fallback: use last part
                        component_name = parts[-1]
                else:
                    component_name = key
            else:
                component_name = key
            
            # Aggregate by component name
            # Update the aggregated meter with all values from this meter
            # Since AverageMeter doesn't store individual values, we approximate
            # by updating with the average value, weighted by count
            if meter.count > 0:
                aggregated[component_name].sum += meter.sum
                aggregated[component_name].count += meter.count
                # Recalculate average
                if aggregated[component_name].count > 0:
                    aggregated[component_name].avg = aggregated[component_name].sum / aggregated[component_name].count
                aggregated[component_name].val = meter.val  # Keep last value
        
        # Convert to dict format
        stats = {}
        for component_name, meter in aggregated.items():
            stats[component_name] = {
                'avg': meter.avg,
                'sum': meter.sum,
                'count': meter.count,
                'val': meter.val,
            }
        return stats
    
    def get_statistics_string(self) -> str:
        """Get formatted statistics string."""
        if not self.timings:
            return "No timing data recorded."
        
        lines = ["Layer Timing Statistics:"]
        lines.append(f"{'Component':<50} {'Avg (ms)':<12} {'Count':<10}")
        lines.append("-" * 75)
        
        # Sort by layer_id and component name
        sorted_keys = sorted(self.timings.keys())
        for key in sorted_keys:
            meter = self.timings[key]
            lines.append(f"{key:<50} {meter.avg:<12.3f} {meter.count:<10}")
        
        return "\n".join(lines)


def get_global_layer_timing_recorder() -> LayerTimingRecorder:
    """Get the global layer timing recorder."""
    global _global_layer_timing_recorder
    if _global_layer_timing_recorder is None:
        _global_layer_timing_recorder = LayerTimingRecorder()
    return _global_layer_timing_recorder


def set_global_layer_timing_recorder(recorder: LayerTimingRecorder):
    """Set the global layer timing recorder."""
    global _global_layer_timing_recorder
    _global_layer_timing_recorder = recorder
