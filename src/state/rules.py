"""
Rule definitions for state classification.

Contains named rule functions that can be used for explainable classification.
Each rule returns a tuple of (applies: bool, state: ForkliftState, reason: str).
"""

from typing import Tuple, Optional

from core.entities import ForkliftState


def rule_idle_no_pallet(
    velocity: float,
    is_carrying: bool,
    threshold: float = 2.0
) -> Tuple[bool, Optional[ForkliftState], str]:
    """
    Rule: Forklift is idle if stationary without pallet.
    
    Conditions:
    - Velocity < threshold
    - Not carrying pallet
    """
    if velocity < threshold and not is_carrying:
        return True, ForkliftState.IDLE, "Stationary without pallet"
    return False, None, ""


def rule_moving_empty(
    velocity: float,
    is_carrying: bool,
    threshold: float = 2.0
) -> Tuple[bool, Optional[ForkliftState], str]:
    """
    Rule: Forklift is moving empty if in motion without pallet.
    
    Conditions:
    - Velocity >= threshold
    - Not carrying pallet
    """
    if velocity >= threshold and not is_carrying:
        return True, ForkliftState.MOVING_EMPTY, "Moving without load"
    return False, None, ""


def rule_moving_loaded(
    velocity: float,
    is_carrying: bool,
    threshold: float = 2.0
) -> Tuple[bool, Optional[ForkliftState], str]:
    """
    Rule: Forklift is transporting if in motion with pallet.
    
    Conditions:
    - Velocity >= threshold
    - Carrying pallet
    """
    if velocity >= threshold and is_carrying:
        return True, ForkliftState.MOVING_LOADED, "Transporting load"
    return False, None, ""


def rule_loading(
    velocity: float,
    is_carrying: bool,
    was_carrying: bool,
    threshold: float = 2.0
) -> Tuple[bool, Optional[ForkliftState], str]:
    """
    Rule: Forklift is loading if stationary and just picked up pallet.
    
    Conditions:
    - Velocity < threshold
    - Now carrying pallet
    - Previously not carrying
    """
    if velocity < threshold and is_carrying and not was_carrying:
        return True, ForkliftState.LOADING, "Picking up load"
    return False, None, ""


def rule_unloading(
    velocity: float,
    is_carrying: bool,
    was_carrying: bool,
    threshold: float = 2.0
) -> Tuple[bool, Optional[ForkliftState], str]:
    """
    Rule: Forklift is unloading if stationary and just put down pallet.
    
    Conditions:
    - Velocity < threshold
    - Not carrying pallet
    - Previously carrying
    """
    if velocity < threshold and not is_carrying and was_carrying:
        return True, ForkliftState.UNLOADING, "Putting down load"
    return False, None, ""


# Ordered list of rules for evaluation
CLASSIFICATION_RULES = [
    rule_loading,
    rule_unloading,
    rule_moving_loaded,
    rule_moving_empty,
    rule_idle_no_pallet,
]


def apply_rules(
    velocity: float,
    is_carrying: bool,
    was_carrying: bool = False,
    threshold: float = 2.0
) -> Tuple[ForkliftState, str]:
    """
    Apply all rules in order and return first matching state.
    
    Args:
        velocity: Current velocity in pixels/frame.
        is_carrying: Whether currently carrying pallet.
        was_carrying: Whether previously carrying pallet.
        threshold: Velocity threshold for idle detection.
        
    Returns:
        Tuple of (state, reason) for explainability.
    """
    for rule_func in CLASSIFICATION_RULES:
        # Check if rule needs was_carrying parameter
        if rule_func in [rule_loading, rule_unloading]:
            applies, state, reason = rule_func(
                velocity, is_carrying, was_carrying, threshold
            )
        else:
            applies, state, reason = rule_func(velocity, is_carrying, threshold)
        
        if applies and state is not None:
            return state, reason
    
    # Default case
    return ForkliftState.UNKNOWN, "No matching rule"
