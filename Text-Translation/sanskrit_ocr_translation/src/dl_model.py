"""
DHCD (Devanagari Handwritten Character Dataset) class mappings and metadata.

This module provides the static mapping of the 46 DHCD character classes,
which include 36 consonants/conjuncts (indices 0-35) and 10 Devanagari digits (indices 36-45).
"""

from typing import Dict, Optional

DHCD_MAP: Dict[int, Dict[str, str]] = {
    0: {"devanagari": "क", "label": "ka", "name": "character_1_ka"},
    1: {"devanagari": "ख", "label": "kha", "name": "character_2_kha"},
    2: {"devanagari": "ग", "label": "ga", "name": "character_3_ga"},
    3: {"devanagari": "घ", "label": "gha", "name": "character_4_gha"},
    4: {"devanagari": "ङ", "label": "nga", "name": "character_5_kna"},
    5: {"devanagari": "च", "label": "cha", "name": "character_6_cha"},
    6: {"devanagari": "छ", "label": "chha", "name": "character_7_chha"},
    7: {"devanagari": "ज", "label": "ja", "name": "character_8_ja"},
    8: {"devanagari": "झ", "label": "jha", "name": "character_9_jha"},
    9: {"devanagari": "ञ", "label": "nya", "name": "character_10_yna"},
    10: {"devanagari": "ट", "label": "ta_hard", "name": "character_11_taamatar"},
    11: {"devanagari": "ठ", "label": "tha_hard", "name": "character_12_thaa"},
    12: {"devanagari": "ड", "label": "da_hard", "name": "character_13_daa"},
    13: {"devanagari": "ढ", "label": "dha_hard", "name": "character_14_dhaa"},
    14: {"devanagari": "ण", "label": "na_hard", "name": "character_15_adna"},
    15: {"devanagari": "त", "label": "ta", "name": "character_16_tabala"},
    16: {"devanagari": "थ", "label": "tha", "name": "character_17_tha"},
    17: {"devanagari": "द", "label": "da", "name": "character_18_da"},
    18: {"devanagari": "ध", "label": "dha", "name": "character_19_dha"},
    19: {"devanagari": "न", "label": "na", "name": "character_20_na"},
    20: {"devanagari": "प", "label": "pa", "name": "character_21_pa"},
    21: {"devanagari": "फ", "label": "pha", "name": "character_22_pha"},
    22: {"devanagari": "ब", "label": "ba", "name": "character_23_ba"},
    23: {"devanagari": "भ", "label": "bha", "name": "character_24_bha"},
    24: {"devanagari": "म", "label": "ma", "name": "character_25_ma"},
    25: {"devanagari": "य", "label": "ya", "name": "character_26_yaw"},
    26: {"devanagari": "र", "label": "ra", "name": "character_27_ra"},
    27: {"devanagari": "ल", "label": "la", "name": "character_28_la"},
    28: {"devanagari": "व", "label": "va", "name": "character_29_waw"},
    29: {"devanagari": "श", "label": "sha", "name": "character_30_motosaw"},
    30: {"devanagari": "ष", "label": "shha", "name": "character_31_petchiryakha"},
    31: {"devanagari": "स", "label": "sa", "name": "character_32_patalosaw"},
    32: {"devanagari": "ह", "label": "ha", "name": "character_33_ha"},
    33: {"devanagari": "क्ष", "label": "ksha", "name": "character_34_chhya"},
    34: {"devanagari": "त्र", "label": "tra", "name": "character_35_tra"},
    35: {"devanagari": "ज्ञ", "label": "gya", "name": "character_36_gya"},
    36: {"devanagari": "०", "label": "0", "name": "digit_0"},
    37: {"devanagari": "१", "label": "1", "name": "digit_1"},
    38: {"devanagari": "२", "label": "2", "name": "digit_2"},
    39: {"devanagari": "३", "label": "3", "name": "digit_3"},
    40: {"devanagari": "४", "label": "4", "name": "digit_4"},
    41: {"devanagari": "५", "label": "5", "name": "digit_5"},
    42: {"devanagari": "६", "label": "6", "name": "digit_6"},
    43: {"devanagari": "७", "label": "7", "name": "digit_7"},
    44: {"devanagari": "८", "label": "8", "name": "digit_8"},
    45: {"devanagari": "९", "label": "9", "name": "digit_9"},
}

NUM_CLASSES: int = len(DHCD_MAP)


def get_character_info(class_idx: int) -> Optional[Dict[str, str]]:
    """
    Retrieve metadata for a DHCD character class by its index.

    Args:
        class_idx (int): The 0-based class index (0 to 45).

    Returns:
        Optional[Dict[str, str]]: Dictionary with 'devanagari', 'label', and 'name',
                                  or None if index is out of range.
    """
    return DHCD_MAP.get(class_idx)


def get_devanagari_glyph(class_idx: int) -> Optional[str]:
    """
    Get the Devanagari character string for a given class index.

    Args:
        class_idx (int): The 0-based class index (0 to 45).

    Returns:
        Optional[str]: Devanagari glyph or None if index is out of range.
    """
    info = DHCD_MAP.get(class_idx)
    return info["devanagari"] if info else None


def get_class_index_by_name(folder_name: str) -> Optional[int]:
    """
    Lookup class index by DHCD folder identifier (e.g. 'character_1_ka' -> 0).

    Args:
        folder_name (str): The dataset folder name.

    Returns:
        Optional[int]: Class index or None if not found.
    """
    for idx, info in DHCD_MAP.items():
        if info["name"] == folder_name:
            return idx
    return None
